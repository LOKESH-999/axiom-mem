//! # Static Free-Idx-Map — Free Index Manager
//!
//! A compact, bitmap-based index allocator that tracks free and used slots in a
//! pre-allocated memory pool.  
//!
//! Each bit represents one slot:
//! - `1` → free
//! - `0` → occupied
//!
//! Optimized for high-performance allocation with:
//! - Branchless index lookup
//! - Constant-time allocation/unchecked_retire
//! - Minimal metadata overhead

/// Bitmap-based index manager for tracking free/used blocks within a fixed-capacity pool.
///
/// Each bit in the `bitmap` represents one block:
/// - `1` → free
/// - `0` → occupied
///
/// The allocator maintains a `free_list` of `u16` indices, where each entry
/// corresponds to a 64-bit segment (map) in the bitmap. This acts as a
/// fast-access stack of candidate segments for allocation.
///
/// # Layout
///
/// For `n_block` total blocks:
///
/// - Number of bitmap entries = `ceil(n_block / 64)`
/// - Each entry manages 64 blocks
/// - If `n_block` is not a multiple of 64, the final entry is masked so that
///   bits beyond capacity are always treated as unavailable
///
/// # Allocation Model
///
/// Allocation operates on a stack of bitmap segments (`free_list`):
///
/// - The allocator always selects the top segment:  
///   `map_idx = free_list[curr_idx]`
/// - A free bit is located using `trailing_zeros`
/// - No global scanning or fallback is performed
///
/// This guarantees constant-time allocation with minimal branching.
///
/// # Terminal Segment (Map 0)
///
/// Map index `0` acts as a permanent fallback segment:
///
/// - `free_list[0]` is initialized to `0`
/// - It is never removed or overwritten
/// - It remains present even when it no longer contains free slots
///
/// ## Invariant
///
/// Within the active freelist region `[0..=curr_idx]`:
///
/// - All entries are expected to refer to non-empty bitmap segments
/// - **Except map `0`**, which is allowed to be empty (`bitmap[0] == 0`)
///
/// This occurs because the allocator never removes the final remaining entry.
///
/// # NULL_IDX Semantics
///
/// [`NULL_IDX`](Self::NULL_IDX) is returned when:
///
/// - The selected segment (`free_list[curr_idx]`) has no free bits
///
/// In particular:
///
/// - When `curr_idx == 0`, allocation is attempted only from map `0`
/// - If `bitmap[0] == 0`, allocation returns `NULL_IDX`
///
/// This does **not** imply that all bitmap segments are exhausted.
/// It only indicates that the current active segment has no capacity.
///
/// # Design Trade-off
///
/// The allocator prioritizes:
///
/// - constant-time allocation
/// - predictable latency
/// - minimal branching
/// - cache-local access patterns
///
/// at the cost of:
///
/// - not guaranteeing global completeness of `free_list`
/// - allowing a stale terminal segment (map `0`)
///
/// # Safety Guarantees
///
/// - Never returns an invalid or out-of-bounds index
/// - Returned indices always correspond to valid set bits in the bitmap
///
/// # Example
///
/// ```text
/// n_block = 130
/// len = ceil(130 / 64) = 3
///
/// bitmap[0] = 0xFFFF_FFFF_FFFF_FFFF (64 free)
/// bitmap[1] = 0xFFFF_FFFF_FFFF_FFFF (64 free)
/// bitmap[2] = 0xFFFF_FFFF_FFFF_FC00 (2 valid bits set)
/// ```
pub struct FreeIdxManager {
    /// Bitmap representing free (1) or used (0) block states.
    bitmap: Box<[u64]>,
    /// Precomputed list of 64-bit map indices for faster traversal.
    free_list: Box<[u16]>,
    /// Indicates the curr_pointing index in `free_list`.
    curr_idx: u16,
    /// Capacity of the Allocated/total slots
    capacity: u32,
    /// Count of Occupaid slots
    occupaid:u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreeMapError {
    /// Returned when the requested index falls outside the allocated memory pool.
    OutOfBounds,
    /// Returned when attempting to retire an index that is already marked as free.
    DoubleFree,
    /// Returned if the `rebuild` length is less than or equal to the current capacity.
    ShrinkNotSupported,
    /// Returned if the `rebuild` length exceeds `FreeIdxManager::MAX_BLOCK`.
    ExceedsMaxCapacity,
}

impl std::fmt::Display for FreeMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            FreeMapError::OutOfBounds => write!(f, "Index out of bounds (SegFault prevented)"),
            FreeMapError::DoubleFree => write!(f, "Attempted to free an already free index"),
            FreeMapError::ShrinkNotSupported => write!(
                f,
                "Rebuilding to a smaller or equal capacity is not supported"
            ),
            FreeMapError::ExceedsMaxCapacity => {
                write!(f, "Requested capacity exceeds MAX_BLOCK limits")
            }
        }
    }
}

impl std::error::Error for FreeMapError {}

/// # Terminal Segment (Map 0)
///
/// Map index `0` acts as a permanent fallback segment in the allocator.
///
/// ## Properties
///
/// - `free_list[0]` is initialized to `0`
/// - It is never removed or overwritten
/// - It remains present even when `curr_idx == 0`
///
/// ## Invariant
///
/// Within the active freelist region `[0..=curr_idx]`:
///
/// - All entries must correspond to non-empty bitmap segments
/// - **Except map 0**, which is allowed to be empty (`bitmap[0] == 0`)
///
/// This occurs because the allocator never removes the final remaining entry.
///
/// ## Allocation Semantics
///
/// Allocation always operates on the top of the freelist:
///
/// ```text
/// map_idx = free_list[curr_idx]
/// ```
///
/// - Only this map is consulted
/// - No scanning or fallback is performed
///
/// When `curr_idx == 0`:
///
/// - The allocator operates solely on map 0
/// - If `bitmap[0] == 0`, [`NULL_IDX`](Self::NULL_IDX) is returned
///
/// ## Design Rationale
///
/// This design:
///
/// - guarantees constant-time allocation
/// - avoids branching and scanning
///
/// at the cost of:
///
/// - allowing a stale fallback entry (map 0)
/// - not enforcing global completeness of the freelist
///
/// ## Safety
///
/// - Never returns an invalid or stale index
/// - Returned indices always correspond to set bits in the bitmap
impl FreeIdxManager {

    /// Number of bits to right-shift for dividing by 64 (`2^6`).
    pub const DIV_BY: u32 = 6;

    /// Number of bits per bitmap entry.
    pub const MAP_WIDTH: u32 = 64;

    /// Mask with all 64 bits set.
    const MASK_SET_MAP: u64 = u64::MAX;

    /// Mask representing the lower 6 bits (`0..63`) used for bit indexing.
    const MASK_64: u64 = 0b111111;

    /// Maximum number of blocks supported by the manager.
    ///
    /// 4,194,112 = 2^22 -192 blocks → supports large pool sizes with compact metadata.
    pub const MAX_BLOCK: u32 = 4_194_112;

    /// The maximum number of 64-bit chunks the `free_list` can safely track.
    ///
    /// Set to `u16::MAX - 2` (65,533) to leave exactly one buffer slot at the
    /// end of the `free_list` array (max length 65,534). This guarantees the
    /// `curr_idx + 1` buffer write inside `unchecked_retire` will never overflow
    /// the `u16` integer limit or panic the array.
    pub const MAX_MAP_ENTRIES: u16 = u16::MAX - 2;

    /// Sentinel value used to indicate no valid free index.
    pub const NULL_IDX: u32 = u32::MAX;

    /// Creates a new free index manager for the given number of blocks.
    ///
    /// # Parameters
    /// - `n_block`: total number of blocks managed.
    ///
    /// # Panics
    /// - If `n_block == 0`
    /// - If `n_block > MAX_BLOCK`
    ///
    /// # Behavior
    /// - Initializes all bits to `1` (all blocks free).
    /// - Applies a final-bit mask if `n_block` is not a multiple of 64.
    pub fn new(n_block: u32) -> Self {
        assert!(n_block > 0, "Number of blocks must be greater than zero");
        assert!(n_block <= Self::MAX_BLOCK, "Exceeded maximum block count");

        // Each u64 entry represents 64 blocks.
        let len = ((n_block + 63) >> Self::DIV_BY) as u16;

        // Initialize free list with linear indices of each bitmap entry.
        let free_list = (0..(len + 1)).collect::<Box<_>>();

        // Initially, all blocks are free (all bits = 1).
        let mut bitmap = (0..len).map(|_| Self::MASK_SET_MAP).collect::<Box<_>>();

        // Calculate remaining blocks in the last 64-bit segment.
        let reminder = n_block as u64 & Self::MASK_64;

        // If not a perfect multiple of 64, mask off invalid tail bits.
        if reminder != 0 {
            let end_mask = Self::MASK_SET_MAP << (Self::MAP_WIDTH - reminder as u32);
            bitmap[(len - 1) as usize] = end_mask;
        }

        Self {
            free_list,
            bitmap,
            curr_idx: len - 1,
            capacity: n_block,
            occupaid:0
        }
    }

    /// Returns the index of a free slot, or `u32::MAX` if all blocks are full.
    ///
    /// This function performs a **branchless lookup**:
    /// - Finds the first available bit (`trailing_zeros`).
    /// - Clears that bit (marks block as used).
    /// - Updates `curr_idx` if this bitmap chunk becomes empty.
    /// - Returns the global block index, or `u32::MAX` if none are free.
    /// 
    /// # Terminal Case Behavior
    ///
    /// When `curr_idx == 0`, the allocator operates on `free_list[0]`,
    /// which is always map 0.
    ///
    /// Map 0 is allowed to be empty (`bitmap[0] == 0`).
    ///
    /// In this case:
    ///
    /// - `get_free_idx` returns `NULL_IDX`
    /// - No further maps are consulted
    ///
    /// This is a deliberate design trade-off for constant-time allocation.
    ///
    #[inline(always)]
    pub const fn get_free_idx(&mut self) -> u32 {
        // NOTE:
        // free_list[0] is a permanent fallback (map 0).
        // It may be empty (bitmap[0] == 0) and is not removed.


        // Load current bitmap index from the freelist
        let map_idx = self.free_list[self.curr_idx as usize];

        // Find first free bit in this 64-bit map (returns 64 if none)
        let map_res = self.bitmap[map_idx as usize].trailing_zeros();

        // Mark that bit as occupied
        self.bitmap[map_idx as usize] &= !(1u64.wrapping_shl(map_res));

        // Move to previous freelist entry if this map is now full
        self.curr_idx -= (self.bitmap[map_idx as usize] == 0) as u16 & (self.curr_idx != 0) as u16;

        // Compute global block index or return `u32::MAX` (branchless)
        let free_idx = ((-((map_res != 64) as i32)).cast_unsigned()
            & ((Self::MAP_WIDTH.saturating_sub(map_res).saturating_sub(1))
                + (map_idx as u32 * Self::MAP_WIDTH)))
            | (-((map_res == 64) as i32)).cast_unsigned();
        self.occupaid += (free_idx != Self::NULL_IDX) as u32;
        free_idx
    }

    
    /// # Safety
    /// Caller must ensure:
    /// - `idx` < total number of blocks managed.
    /// - The same `idx` is not released twice without a reallocation.
    ///   Violating either may corrupt the bitmap state or freelist tracking.
    #[inline(always)]
    pub unsafe fn unchecked_retire(&mut self, id: u32) {
        let idx = id;
        // dividing it by 64 inorder to find map-index.
        let map_idx = Self::get_map_idx(idx);
        // if `self.bitmap[map_idx as usize] == 0` then we need to add a marking in `free_list`.
        let is_to_add = (self.bitmap[map_idx as usize] == 0) & (map_idx != 0) ;
        // let shoudl_subtract = is_to_add & self.free_list[self.curr_idx as usize] == 0;
        
        let has_added = unsafe { !self.unchecked_is_free(idx) };
        self.occupaid -= has_added as u32;

        // marking the `idx` in its `bitmap` slot.
        self.bitmap[map_idx as usize] |=
            1u64.wrapping_shl((Self::MAP_WIDTH as u64 - ((idx + 1) as u64 & Self::MASK_64)) as u32);
        
        // this acts as the buffer write.
        // And at the time of initilization we do cleverly add one extra slot to act as buff so we dont write in uninitilized memeory.
        self.free_list[self.curr_idx as usize + 1] = map_idx as u16;
        // if `self.bitmap[map_idx as usize] == 0` then the above will be valid by we increment `curr_idx` else it acts as dead buff
        self.curr_idx += is_to_add as u16;

    }

    /// checks if a specific slot index is free in the bitmap.
    ///
    /// # Safety
    /// - The caller **must ensure** that `idx` is within the valid range of the bitmap.
    ///   Passing an out-of-bounds index will result in undefined behavior.
    ///
    /// # Parameters
    /// - `idx`: The slot index to check.
    ///
    /// # Returns
    /// - `true` if the slot at `idx` is free.
    /// - `false` if the slot is occupied.
    ///
    /// # Details
    /// The bitmap (`u64` array) tracks free slots:
    /// - Computes `map_idx` by dividing `idx` by 64 (`idx >> DIV_BY`) to locate the `u64` block.
    /// - Computes a `mask` to isolate the corresponding bit.
    /// - Checks if the bit is set (free).
    #[inline(always)]
    pub unsafe fn unchecked_is_free(&self, idx: u32) -> bool {
        let map_idx = Self::get_map_idx(idx);
        let mask =
            1u64.wrapping_shl((Self::MAP_WIDTH as u64 - ((idx + 1) as u64 & Self::MASK_64)) as u32);
        self.bitmap[map_idx as usize] & mask == mask
    }

    /// Computes the physical `bitmap` array index for a given logical block index.
    ///
    /// This function uses a fast bitwise right-shift (`>> DIV_BY`) to effectively
    /// divide the given logical block index by 64. This determines exactly which
    /// `u64` chunk in the `bitmap` array holds the bit corresponding to the `idx`.
    ///
    /// # Parameters
    /// * `idx`: The logical block index to translate.
    ///
    /// # Returns
    /// * `u32`: The index of the `u64` chunk within the `bitmap` array. (Note: You may
    ///   need to cast this to `usize` when actually indexing into the array).
    ///
    /// # Warning: No Bounds Checking
    /// This function performs a pure mathematical translation and does **not** verify
    /// if the provided `idx` is within the allocated capacity of the memory pool.
    ///
    /// If you pass an `idx` that exceeds the total number of managed blocks, this
    /// function will compute a map index that is **out-of-bounds**. Using this returned
    /// index directly to access the `bitmap` array without prior validation will result
    /// in a panic, or potentially undefined behavior/segmentation faults if used within
    /// an `unsafe` block.
    #[inline(always)]
    pub const fn get_map_idx(idx: u32) -> u32 {
        idx >> Self::DIV_BY
    }

    /// Safely checks if a specific slot index is free.
    ///
    /// Returns `FreeMapError::OutOfBounds` if the index equals or exceeds the allocated logical capacity.
    #[inline]
    pub fn is_free(&self, idx: u32) -> Result<bool, FreeMapError> {
        // Prevent SegFault: Ensure we strictly respect the logical limits
        if idx >= self.capacity {
            return Err(FreeMapError::OutOfBounds);
        }

        // INVARIANT MAINTAINED: map_idx is strictly within self.bitmap bounds.
        Ok(unsafe { self.unchecked_is_free(idx) })
    }

    /// Safely retires a slot index back to the free pool.
    ///
    /// Returns:
    /// - `FreeMapError::OutOfBounds` if the index equals or exceeds the logical capacity limits.
    /// - `FreeMapError::DoubleFree` if the bit is already set to 1 (free).
    #[inline]
    pub fn retire(&mut self, idx: u32) -> Result<(), FreeMapError> {
        // 1. Bounds Check
        let is_free = self.is_free(idx)?;

        // 2. Double-Free Check
        if is_free {
            return Err(FreeMapError::DoubleFree);
        }

        // INVARIANTS MAINTAINED:
        // - Index is within bounds.
        // - Index is currently occupied (0), so freeing it will not double-count.
        unsafe { self.unchecked_retire(idx) };

        Ok(())
    }

    /// Returns the capacity
    #[inline(always)]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the free slots
    #[inline(always)]
    pub const fn free_slots(&self)-> u32{
        self.capacity - self.occupaid
    }
    /// Expands the capacity of the memory pool to a new length.
    ///
    /// This safely handles unmasking previously out-of-bounds bits, expanding
    /// the physical allocations if necessary, and re-applying the strict logical
    /// boundary mask—all while keeping the `free_list` stack perfectly synchronized.
    pub fn rebuild(&mut self, new_cap: u32) -> Result<(), FreeMapError> {
        if new_cap <= self.capacity() {
            return Err(FreeMapError::ShrinkNotSupported);
        }
        if new_cap > Self::MAX_BLOCK {
            return Err(FreeMapError::ExceedsMaxCapacity);
        }

        let old_map_idx = Self::get_map_idx(self.capacity - 1) as usize;
        let new_map_idx = Self::get_map_idx(new_cap - 1) as usize;

        // --- STEP 1: UNMASK THE OLD BOUNDARY ---
        let old_reminder = Self::MASK_64 & (self.capacity as u64);
        if old_reminder != 0 {
            // creating mask to fill new slots
            let rem_mask = Self::MASK_SET_MAP >> old_reminder;

            // Safely restore chunk to free_list if it was completely full
            if self.bitmap[old_map_idx] == 0 && old_map_idx != 0 {
                self.free_list[self.curr_idx as usize + 1] = old_map_idx as u16;
                self.curr_idx += 1;
            }

            // Unlock all previously padded bits in this segment
            self.bitmap[old_map_idx] |= rem_mask;
        }

        // --- STEP 2: EXPAND PHYSICAL ARRAYS (IF CROSSING SEGMENTS) ---
        if new_map_idx > old_map_idx {
            let new_map_len = new_map_idx + 1;

            // Zero-copy swap out the box, convert to vec, resize, and box it again
            let mut new_bitmap = std::mem::replace(&mut self.bitmap, Box::new([])).into_vec();
            new_bitmap.resize(new_map_len, Self::MASK_SET_MAP);
            self.bitmap = new_bitmap.into_boxed_slice();

            // Same for the free_list (+1 for the buffer write slot)
            let mut new_freelist = std::mem::replace(&mut self.free_list, Box::new([])).into_vec();
            new_freelist.resize(new_map_len + 1, 0);
            self.free_list = new_freelist.into_boxed_slice();

            // Push the newly created physical chunks onto the freelist stack
            for i in (old_map_idx + 1)..new_map_len {
                assert!(
                    i <= Self::MAX_MAP_ENTRIES as usize,
                    "No of entries must be bounded in free-list"
                );
                self.free_list[self.curr_idx as usize + 1] = i as u16;
                self.curr_idx += 1;
            }
        }

        // --- STEP 3: APPLY NEW CAPACITY & BOUNDARY MASK ---
        self.capacity = new_cap;

        let new_reminder = Self::MASK_64 & (self.capacity as u64);
        if new_reminder != 0 {
            // Apply the strict boundary mask for the NEW capacity
            let shift_amt = Self::MAP_WIDTH - (new_reminder as u32);
            let end_mask = Self::MASK_SET_MAP << shift_amt;
            self.bitmap[new_map_idx] &= end_mask;
        }

        Ok(())
    }
}

unsafe impl Send for FreeIdxManager {}
unsafe impl Sync for FreeIdxManager {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_internal_consistency(mgr: &FreeIdxManager) {
        let mut seen = std::collections::HashSet::new();

        // Check freelist entries
        for i in 0..=mgr.curr_idx as usize {
            let idx = mgr.free_list[i] as usize;

            assert!(idx < mgr.bitmap.len(), "freelist OOB idx={}", idx);
            assert!(seen.insert(idx), "duplicate freelist entry: {}", idx);
            assert!(
                mgr.bitmap[idx] != 0 || (idx == 0),
                "freelist contains empty map in active region:{idx},{},{:?},{:?},{}",mgr.bitmap[idx],mgr.bitmap,mgr.free_list,mgr.curr_idx
            );
        }

        // Check bitmap ↔ freelist consistency
        for i in 0..=mgr.curr_idx as usize {
            let idx = mgr.free_list[i] as usize;

            assert!(
                mgr.bitmap[idx] != 0 || (idx == 0),
                "freelist contains empty map in active region: {}",
                idx
            );
        }
    }

    #[test]
    fn test_force_map0_loss_scenario() {
        let mut mgr = FreeIdxManager::new(128);

        let mut allocated = Vec::new();

        // Step 1: fully exhaust
        loop {
            let idx = mgr.get_free_idx();
            if idx == u32::MAX { break; }
            allocated.push(idx);
        }

        // Step 2: free ONLY map 0
        for &idx in &allocated {
            if FreeIdxManager::get_map_idx(idx) == 0 {
                unsafe { mgr.unchecked_retire(idx); }
            }
        }

        // At this point:
        // bitmap[0] != 0
        // others = 0

        // Step 3: aggressively overwrite freelist with other maps
        // This is the key missing piece
        for _ in 0..50 {
            for &idx in &allocated {
                if FreeIdxManager::get_map_idx(idx) != 0 {
                    unsafe { mgr.unchecked_retire(idx); }
                }
            }
        }

        // Step 4: collapse stack
        while mgr.curr_idx > 0 {
            mgr.get_free_idx();
        }

        // Now we are in the dangerous state:
        // curr_idx == 0
        // bitmap[0] != 0
        // freelist[0] might not be 0

        let top = mgr.free_list[0] as usize;

        if mgr.bitmap[0] != 0 {
            // If top is not 0 → we LOST map 0
            if top != 0 {
                let idx = mgr.get_free_idx();

                assert!(
                    idx != u32::MAX,
                    "BUG: map 0 lost, allocator returned NULL"
                );
            }
        }
    }

    #[test]
    fn test_freelist_covers_at_least_one_non_empty_map() {
        let mut mgr = FreeIdxManager::new(256);

        for _ in 0..5000 {
            let idx = mgr.get_free_idx();

            if idx != u32::MAX {
                unsafe { mgr.unchecked_retire(idx); }
            }

            // check: if any bitmap has free space → freelist must reflect at least one
            let any_free = mgr.bitmap.iter().any(|&b| b != 0);

            if any_free {
                let top = mgr.free_list[mgr.curr_idx as usize] as usize;

                assert!(
                    mgr.bitmap[top] != 0 || mgr.curr_idx == 0,
                    "freelist lost all non-empty maps"
                );
            }
        }
    }

    #[test]
    fn test_rebuild_no_duplicate_map0_on_partial_expansion() {
        // Create a pool small enough to fit inside a single partial map (Map 0)
        let mut mgr = FreeIdxManager::new(10);
        
        // Exhaust the pool completely (bitmap[0] becomes 0)
        for _ in 0..10 {
            mgr.get_free_idx();
        }
        
        // Rebuild within the exact same chunk
        mgr.rebuild(50).unwrap();
        
        // The fix (&& old_map_idx != 0) prevents 0 from being pushed onto the stack again.
        // We verify the active freelist region contains NO duplicate entries.
        let mut seen = std::collections::HashSet::new();
        for i in 0..=mgr.curr_idx as usize {
            let val = mgr.free_list[i];
            assert!(seen.insert(val), "Duplicate map index found in freelist: {}", val);
        }
        
        // Because we stayed in Map 0, curr_idx must still be exactly 0
        assert_eq!(mgr.curr_idx, 0);
    }

    #[test]
    fn test_rebuild_no_duplicate_map0_on_cross_segment_expansion() {
        // Create a pool small enough to fit inside a single partial map (Map 0)
        let mut mgr = FreeIdxManager::new(10);
        
        // Exhaust the pool completely
        for _ in 0..10 {
            mgr.get_free_idx();
        }
        
        // Rebuild across a chunk boundary (forces the creation of Map 1)
        mgr.rebuild(100).unwrap();
        
        // Verify no duplicate entries were pushed during the boundary unmasking
        let mut seen = std::collections::HashSet::new();
        for i in 0..=mgr.curr_idx as usize {
            let val = mgr.free_list[i];
            assert!(seen.insert(val), "Duplicate map index found in freelist: {}", val);
        }
        
        // Since we expanded into Map 1, the stack must be exactly [0, 1]
        assert_eq!(mgr.curr_idx, 1);
        assert_eq!(mgr.free_list[0], 0);
        assert_eq!(mgr.free_list[1], 1);
    }

    #[test]
    fn test_rebuild_unmask_old_segment_not_reachable() {
        let mut mgr = FreeIdxManager::new(65);

        // Exhaust everything
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // At this point:
        // bitmap[0] = 0
        // bitmap[1] = 0

        // Rebuild → increases capacity inside SAME segment
        mgr.rebuild(100).unwrap();

        // Now:
        // bitmap[1] has new free bits
        // but may NOT be in freelist

        let mut found = false;

        for _ in 0..10 {
            let idx = mgr.get_free_idx();
            if idx != FreeIdxManager::NULL_IDX {
                found = true;
                break;
            }
        }

        assert!(
            found,
            "rebuild failed: new free bits in old segment are not reachable"
        );
    }

    #[test]
    fn test_rebuild_stale_top_returns_null() {
        let mut mgr = FreeIdxManager::new(64);

        // Exhaust everything
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // Now:
        // curr_idx = 0
        // bitmap[0] = 0

        // Rebuild → add new maps
        mgr.rebuild(128).unwrap();

        // If top is still stale (0), this will fail
        let idx = mgr.get_free_idx();

        assert!(
            idx != FreeIdxManager::NULL_IDX,
            "rebuild failed: stale top caused NULL despite free space"
        );
    }

    #[test]
    fn test_rebuild_top_must_be_valid() {
        let mut mgr = FreeIdxManager::new(128);

        // Create fragmentation
        for _ in 0..128 {
            let _ = mgr.get_free_idx();
        }

        // Free only some
        for i in (0..128).step_by(5) {
            unsafe { mgr.unchecked_retire(i) };
        }

        // Rebuild
        mgr.rebuild(256).unwrap();

        let top = mgr.free_list[mgr.curr_idx as usize] as usize;

        assert!(
            mgr.bitmap[top] != 0 || mgr.curr_idx == 0,
            "top of freelist is stale after rebuild"
        );
    }

    #[test]
    fn test_rebuild_old_segment_reuse() {
        let mut mgr = FreeIdxManager::new(70);

        // Exhaust all
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // Rebuild expands inside last segment
        mgr.rebuild(120).unwrap();

        // Expect allocation from newly unlocked region
        let idx = mgr.get_free_idx();

        assert!(
            idx != FreeIdxManager::NULL_IDX,
            "old segment free bits not reused after rebuild"
        );
    }

    #[test]
    fn test_rebuild_freelist_covers_new_maps() {
        let mut mgr = FreeIdxManager::new(64);

        // Exhaust
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // Rebuild
        mgr.rebuild(256).unwrap();

        // Check: at least one active freelist entry must be valid
        let mut found_valid = false;

        for i in 0..=mgr.curr_idx as usize {
            let idx = mgr.free_list[i] as usize;
            if mgr.bitmap[idx] != 0 {
                found_valid = true;
                break;
            }
        }

        assert!(
            found_valid,
            "freelist does not cover any non-empty bitmap after rebuild"
        );
    }

    #[test]
    fn test_rebuild_after_full_exhaustion() {
        let mut mgr = FreeIdxManager::new(128);

        // Exhaust
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // Rebuild
        mgr.rebuild(512).unwrap();

        // Try multiple allocations
        let mut success = 0;

        for _ in 0..10 {
            if mgr.get_free_idx() != FreeIdxManager::NULL_IDX {
                success += 1;
            }
        }

        assert!(
            success > 0,
            "allocator failed after rebuild from fully exhausted state"
        );
    }

    #[test]
    fn test_minimal_two_map_failure() {
        let mut mgr = FreeIdxManager::new(128);

        // exhaust all
        while mgr.get_free_idx() != u32::MAX {}

        // free only map 0 entries
        for i in 0..64 {
            unsafe { mgr.unchecked_retire(i); }
        }

        // now allocator SHOULD return valid index
        let idx = mgr.get_free_idx();

        assert!(
            idx != u32::MAX,
            "minimal repro: allocator missed map 0"
        );
    }

    #[test]
    fn test_cleanup_fails_when_curr_idx_zero_and_map_missing() {
        let mut mgr = FreeIdxManager::new(128);

        let mut allocated = Vec::new();

        // exhaust all
        loop {
            let idx = mgr.get_free_idx();
            if idx == u32::MAX { break; }
            allocated.push(idx);
        }

        // free only map 0
        for &idx in &allocated {
            if FreeIdxManager::get_map_idx(idx) == 0 {
                unsafe { mgr.unchecked_retire(idx); }
            }
        }

        // force freelist pollution by freeing other maps repeatedly
        for _ in 0..5 {
            for &idx in &allocated {
                if FreeIdxManager::get_map_idx(idx) != 0 {
                    unsafe { mgr.unchecked_retire(idx); }
                }
            }
        }

        // Now curr_idx may be 0 but freelist[0] ≠ 0

        if mgr.curr_idx == 0 {
            let idx = mgr.get_free_idx();

            assert!(
                idx != u32::MAX,
                "cleanup failed: NULL returned while free space exists"
            );
        }
    }
    #[test]
    fn test_duplicate_map_entries_lead_to_stale_active() {
        let mut mgr = FreeIdxManager::new(128);

        let mut allocated = Vec::new();

        // allocate some
        for _ in 0..50 {
            let idx = mgr.get_free_idx();
            if idx == u32::MAX { break; }
            allocated.push(idx);
        }

        // free repeatedly from same map to force duplicates
        for _ in 0..10 {
            for &idx in &allocated {
                if FreeIdxManager::get_map_idx(idx) == 1 {
                    unsafe { mgr.unchecked_retire(idx); }
                }
            }
        }

        // exhaust map 1 again
        loop {
            let idx = mgr.get_free_idx();
            if idx == u32::MAX { break; }
        }

        // Now freelist likely contains stale entries of map 1
        let top = mgr.free_list[mgr.curr_idx as usize] as usize;

        assert!(
            mgr.bitmap[top] != 0 || mgr.curr_idx == 0,
            "stale map present in active region"
        );
    }

    #[test]
    fn test_lost_map_returns_null_even_when_space_exists() {
        let mut mgr = FreeIdxManager::new(128);

        // Step 1: exhaust everything
        let mut allocated = Vec::new();
        loop {
            let idx = mgr.get_free_idx();
            if idx == u32::MAX { break; }
            allocated.push(idx);
        }

        // Step 2: free ONLY from map 0
        // (force situation where only map 0 has space)
        for &idx in &allocated {
            let map_idx = FreeIdxManager::get_map_idx(idx);
            if map_idx == 0 {
                unsafe { mgr.unchecked_retire(idx); }
            }
        }

        // Now:
        // bitmap[0] != 0
        // other maps may be 0
        // freelist may have lost map 0

        let idx = mgr.get_free_idx();

        assert!(
            idx != u32::MAX,
            "allocator returned NULL while map 0 still has free slots"
        );
    }

    #[test]
    fn test_internal_invariant_basic() {
        let mut mgr = FreeIdxManager::new(128);

        for _ in 0..128 {
            let _ = mgr.get_free_idx();
            assert_internal_consistency(&mgr);
        }

        for i in 0..128 {
            unsafe { mgr.unchecked_retire(i) };
            assert_internal_consistency(&mgr);
        }
    }

    #[test]
    fn test_no_duplicate_freelist_entries_after_mass_free() {
        let mut mgr = FreeIdxManager::new(256);

        let mut allocated = Vec::new();
        for _ in 0..256 {
            allocated.push(mgr.get_free_idx());
        }

        for idx in allocated {
            unsafe { mgr.unchecked_retire(idx) };
        }

        let mut seen = std::collections::HashSet::new();

        for i in 0..=mgr.curr_idx as usize {
            let v = mgr.free_list[i];
            assert!(seen.insert(v), "duplicate freelist entry detected: {}", v);
        }
    }

    #[test]
    fn test_adversarial_pattern() {
        let mut mgr = FreeIdxManager::new(256);
        let mut allocated = Vec::new();

        for i in 0..10_000 {
            if i % 3 != 0 {
                let idx = mgr.get_free_idx();
                if idx != u32::MAX {
                    allocated.push(idx);
                }
            } else if !allocated.is_empty() {
                let idx = allocated.remove(0);
                unsafe { mgr.unchecked_retire(idx) };
            }

            assert_internal_consistency(&mgr);
        }
    }

    #[test]
    fn test_rebuild_after_fragmentation() {
        let mut mgr = FreeIdxManager::new(128);

        let mut allocated = Vec::new();
        for _ in 0..128 {
            allocated.push(mgr.get_free_idx());
        }

        // Free some random subset
        for i in (0..128).step_by(3) {
            unsafe { mgr.unchecked_retire(allocated[i]) };
        }

        assert_internal_consistency(&mgr);

        // Rebuild
        mgr.rebuild(256).unwrap();

        // Validate after rebuild
        assert_internal_consistency(&mgr);

        // Continue usage
        for _ in 0..200 {
            let _ = mgr.get_free_idx();
            assert_internal_consistency(&mgr);
        }
    }

    #[test]
    fn test_freelist_near_capacity_limit() {
        let max_blocks = FreeIdxManager::MAX_BLOCK - 1;
        let mut mgr = FreeIdxManager::new(max_blocks);

        // allocate a chunk
        for _ in 0..1000 {
            let _ = mgr.get_free_idx();
        }

        // free a chunk
        for i in 0..1000 {
            unsafe { mgr.unchecked_retire(i) };
        }

        assert_internal_consistency(&mgr);
    }

    #[test]
    fn test_repeated_alloc_free_same_region() {
        let mut mgr = FreeIdxManager::new(64);

        for _ in 0..10_000 {
            let idx = mgr.get_free_idx();
            if idx != u32::MAX {
                unsafe { mgr.unchecked_retire(idx) };
            }
        }

        assert_internal_consistency(&mgr);
    }

    #[test]
    fn test_safe_api_out_of_bounds() {
        let mut mgr = FreeIdxManager::new(130);

        // Valid bounds check
        assert!(mgr.is_free(129).is_ok());

        // Exactly on the boundary (130) should fail
        assert_eq!(mgr.is_free(130), Err(FreeMapError::OutOfBounds));
        assert_eq!(mgr.retire(130), Err(FreeMapError::OutOfBounds));

        // Way out of bounds should fail
        assert_eq!(mgr.is_free(999), Err(FreeMapError::OutOfBounds));
        assert_eq!(mgr.retire(999), Err(FreeMapError::OutOfBounds));
    }

    #[test]
    fn test_safe_api_double_free() {
        let mut mgr = FreeIdxManager::new(64);

        // At initialization, all blocks are free. Retiring immediately should error.
        assert_eq!(mgr.retire(0), Err(FreeMapError::DoubleFree));
        assert_eq!(mgr.retire(63), Err(FreeMapError::DoubleFree));

        // Allocate a block
        let idx = mgr.get_free_idx();
        assert_eq!(idx, 63);

        // First retire should succeed
        assert_eq!(mgr.retire(idx), Ok(()));

        // Second retire of the same index should fail
        assert_eq!(mgr.retire(idx), Err(FreeMapError::DoubleFree));
    }

    #[test]
    fn test_safe_api_lifecycle() {
        let mut mgr = FreeIdxManager::new(32);

        // Verify capacity getter
        assert_eq!(mgr.capacity(), 32);

        // Initially free
        assert_eq!(mgr.is_free(5), Ok(true));

        // Allocate until we get index 5
        let mut allocs = vec![];
        for _ in 0..6 {
            allocs.push(mgr.get_free_idx());
        }
        print!("{:?}", mgr.bitmap);

        // Index 5 should now be occupied
        assert_eq!(mgr.is_free(32 - 5), Ok(false));

        // Retire it safely
        assert_eq!(mgr.retire(32 - 5), Ok(()));

        // Should be free again
        assert_eq!(mgr.is_free(32 - 5), Ok(true));

        // Next allocation should prioritize the newly freed index 5
        assert_eq!(mgr.get_free_idx(), 32 - 5);
    }

    #[test]
    fn test_rebuild_errors() {
        let mut mgr = FreeIdxManager::new(100);

        // Error: Shrink (equal to current)
        assert_eq!(mgr.rebuild(100), Err(FreeMapError::ShrinkNotSupported));

        // Error: Shrink (less than current)
        assert_eq!(mgr.rebuild(50), Err(FreeMapError::ShrinkNotSupported));

        // Error: Exceed Absolute MAX
        assert_eq!(
            mgr.rebuild(FreeIdxManager::MAX_BLOCK + 1),
            Err(FreeMapError::ExceedsMaxCapacity)
        );
    }

    #[test]
    fn test_rebuild_chained_expansions() {
        let mut mgr = FreeIdxManager::new(10);
        let idx1 = mgr.get_free_idx(); // Should be 9

        // Rebuild across the first 64-bit boundary
        mgr.rebuild(70).unwrap();
        let idx2 = mgr.get_free_idx(); // Should be 69

        // Rebuild again into a third chunk
        mgr.rebuild(150).unwrap();
        let idx3 = mgr.get_free_idx(); // Should be 149

        // Verify the original state survived the multiple memory reallocations
        assert_eq!((idx1, idx2, idx3), (9, 69, 149));
        assert_eq!(mgr.capacity(), 150);

        // Ensure boundary of the final expansion is strict
        assert_eq!(mgr.is_free(148), Ok(true));
        assert_eq!(mgr.is_free(150), Err(FreeMapError::OutOfBounds));
    }

    #[test]
    fn test_rebuild_to_absolute_max() {
        let mut mgr = FreeIdxManager::new(100);

        // Consume all initial slots
        for _ in 0..100 {
            mgr.get_free_idx();
        }
        assert_eq!(mgr.get_free_idx(), FreeIdxManager::NULL_IDX);

        // Nuke the capacity to the maximum allowed limit
        assert_eq!(mgr.rebuild(FreeIdxManager::MAX_BLOCK), Ok(()));
        assert_eq!(mgr.capacity(), FreeIdxManager::MAX_BLOCK);

        // Next allocation should cross perfectly into the newly available space
        let idx = mgr.get_free_idx();
        assert_eq!(idx, FreeIdxManager::MAX_BLOCK - 1);

        // Check the absolute final valid index is correctly masked and available
        assert_eq!(mgr.is_free(FreeIdxManager::MAX_BLOCK - 2), Ok(true));

        // Ensure we haven't leaked out of bounds on the massive array
        assert_eq!(
            mgr.is_free(FreeIdxManager::MAX_BLOCK),
            Err(FreeMapError::OutOfBounds)
        );
    }

    #[test]
    fn test_rebuild_complex_state_preservation() {
        let mut mgr = FreeIdxManager::new(64);

        // 1. Fully allocate
        let mut allocs = vec![];
        for _ in 0..64 {
            allocs.push(mgr.get_free_idx());
        }

        // 2. Introduce scattered fragmentation
        mgr.retire(5).unwrap();
        mgr.retire(15).unwrap();
        mgr.retire(63).unwrap();

        // 3. Rebuild to 128
        mgr.rebuild(128).unwrap();

        // We now have 64 new slots (64..127) plus 3 freed slots (5, 15, 63).
        // Total available = 67. We will allocate exactly 67 times.
        let mut new_allocs = vec![];
        for _ in 0..67 {
            let idx = mgr.get_free_idx();
            assert_ne!(idx, FreeIdxManager::NULL_IDX);
            new_allocs.push(idx);
        }

        // 4. Ensure the pool is perfectly full again
        assert_eq!(mgr.get_free_idx(), FreeIdxManager::NULL_IDX);

        assert!(
            new_allocs.contains(&5),
            "Failed to track fragmented index 5"
        );
        assert!(
            new_allocs.contains(&15),
            "Failed to track fragmented index 15"
        );
        assert!(
            new_allocs.contains(&63),
            "Failed to track fragmented index 63"
        );
    }

    #[test]
    fn test_freeidx_init_min_block() {
        let m = FreeIdxManager::new(1);
        assert_eq!(m.bitmap.len(), 1);
        assert_eq!(m.free_list.len(), 1 + 1);
        assert_eq!(m.curr_idx, 0);

        // Only the top bit should be set (free)
        let expected_mask = FreeIdxManager::MASK_SET_MAP << (FreeIdxManager::MAP_WIDTH - 1);
        assert_eq!(m.bitmap[0], expected_mask);
        assert_eq!(m.bitmap[0], 0b1 << (FreeIdxManager::MAP_WIDTH - 1));
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_63_block() {
        let m = FreeIdxManager::new(63);
        assert_eq!(m.bitmap.len(), 1);
        assert_eq!(m.free_list.len(), 2);
        assert_eq!(m.curr_idx, 0);

        // Only the top bit should be set (free)
        let expected_mask = u64::MAX << (64 - 63);
        assert_eq!(m.bitmap[0], expected_mask);
        assert_eq!(
            m.bitmap[0],
            0b1111111111111111111111111111111111111111111111111111111111111110
        );
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_init_exact_multiple_64() {
        let m = FreeIdxManager::new(128);
        assert_eq!(m.bitmap.len(), 2);
        assert_eq!(m.free_list.len(), 2 + 1);
        assert_eq!(m.curr_idx, 1);

        // Both bitmaps should be full (all bits set)
        assert!(m.bitmap.iter().all(|&b| b == u64::MAX));
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_init_partial_map() {
        let m = FreeIdxManager::new(70);
        assert_eq!(m.bitmap.len(), 2);

        // First map is full
        assert_eq!(m.bitmap[0], u64::MAX);

        // Remainder = 6 → top 6 bits set
        let expected_end = u64::MAX << (64 - 6);
        assert_eq!(m.bitmap[1], expected_end);
        assert_eq!(m.bitmap[1], 0b111111 << 58);
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        assert_eq!(m.curr_idx, 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_init_max_limit() {
        let n = FreeIdxManager::MAX_BLOCK - 1;
        let m = FreeIdxManager::new(n);
        assert!(m.bitmap.len() > 0);
        assert_eq!(m.bitmap.len(), ((n + 63) >> 6) as usize);
        assert_eq!(m.curr_idx, 65532);
        for b in m.bitmap[..(m.bitmap.len() - 1)].iter() {
            assert_eq!(*b, u64::MAX);
        }
        assert_eq!(*m.bitmap.last().unwrap(), u64::MAX - 1);
        println!("{:b}", m.bitmap.last().unwrap());
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    #[should_panic]
    fn test_freeidx_zero_block_panics() {
        let _ = FreeIdxManager::new(0);
    }

    #[test]
    #[should_panic]
    fn test_freeidx_overflow_panics() {
        let _ = FreeIdxManager::new(FreeIdxManager::MAX_BLOCK + 1);
    }

    #[test]
    fn test_bitmap_all_ones_at_start() {
        let m = FreeIdxManager::new(128);
        for b in m.bitmap.iter() {
            // Every bit should be 1 → all free
            assert_eq!(*b, u64::MAX);
        }
        assert_eq!(m.bitmap.len(), 2);
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        assert_eq!(m.curr_idx, 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }
    /// Helper to count remaining free bits across all bitmaps.
    fn count_free_bits(f: &FreeIdxManager) -> u32 {
        f.bitmap.iter().map(|x| x.count_ones()).sum()
    }

    #[test]
    fn test_initialization_masks_last_entry_correctly() {
        // Case 1: exactly multiple of 64 → all u64s should be full
        let f1 = FreeIdxManager::new(128);
        assert_eq!(f1.bitmap.len(), 2);
        assert!(f1.bitmap.iter().all(|&b| b == u64::MAX));

        // Case 2: not multiple of 64 → last entry must be masked
        let f2 = FreeIdxManager::new(130);
        assert_eq!(f2.bitmap.len(), 3);
        let last_mask = f2.bitmap[2];
        // 130 % 64 = 2, so last mask keeps only top 2 bits set
        let expected_mask = u64::MAX << (64 - 2);
        assert_eq!(last_mask, expected_mask);
    }

    #[test]
    fn test_allocate_until_full_then_returns_u32_max() {
        let mut mgr = FreeIdxManager::new(64);
        let mut results = vec![];

        // Should allocate 64 valid indices: 0..63
        for _ in 0..64 {
            let idx = mgr.get_free_idx();
            assert!(idx < 64, "idx={}", idx);
            results.push(idx);
        }

        // All bits consumed → next must be u32::MAX
        let idx = mgr.get_free_idx();
        assert_eq!(idx, u32::MAX);

        // No free bits left
        assert_eq!(count_free_bits(&mgr), 0);
    }

    #[test]
    fn test_multi_map_progression() {
        // 128 blocks => 2 u64 entries
        let mut mgr = FreeIdxManager::new(128);
        let mut seen = vec![];

        // Allocate all → should go from 0..127
        for _ in 0..128 {
            let idx = mgr.get_free_idx();
            assert!(idx < 128);
            seen.push(idx);
        }

        for _ in 0..1000 {
            // Must now return u32::MAX
            assert_eq!(mgr.get_free_idx(), u32::MAX);
            // Total bits now all consumed
            assert_eq!(count_free_bits(&mgr), 0);
        }
    }

    #[test]
    fn test_partial_map_behavior() {
        // 70 blocks → 2 u64 entries, only 6 bits in second entry valid
        let mut mgr = FreeIdxManager::new(70);
        let total_before = count_free_bits(&mgr);
        assert_eq!(total_before, 70);

        // Allocate all → should drain all bits
        for _ in 0..70 {
            let idx = mgr.get_free_idx();
            println!("idx:{idx}");
            assert!(idx < 70, "invalid idx={}", idx);
        }

        // Next call → u32::MAX
        assert_eq!(mgr.get_free_idx(), u32::MAX);
        assert_eq!(count_free_bits(&mgr), 0);
    }

    #[test]
    fn test_curr_idx_moves_downward_as_maps_fill() {
        // 128 blocks → 2 maps → curr_idx starts = 2
        let mut mgr = FreeIdxManager::new(128);
        assert_eq!(mgr.curr_idx, 1);

        // Fill first map fully (64 blocks)
        for _ in 0..64 {
            mgr.get_free_idx();
        }
        // It should have moved down once (still not zero)
        assert_eq!(mgr.curr_idx, 0);

        // Fill second map
        for _ in 0..64 {
            mgr.get_free_idx();
        }
        // Fully drained → curr_idx should now be 0
        assert_eq!(mgr.curr_idx, 0);

        // Next call returns u32::MAX
        assert_eq!(mgr.get_free_idx(), u32::MAX);
    }

    #[test]
    fn allocates_and_releases_correctly() {
        let mut mgr = FreeIdxManager::new(128);

        // Collect all allocations
        let mut indices = Vec::new();
        for _ in 0..128 {
            let idx = mgr.get_free_idx();
            assert!(idx != u32::MAX, "Should return a valid free index");
            indices.push(idx);
        }

        // After all allocated, next call should return NULL_IDX (u32::MAX)
        assert_eq!(mgr.get_free_idx(), FreeIdxManager::NULL_IDX);

        // Release one block and allocate again — should reuse the freed one
        unsafe { mgr.unchecked_retire(indices[5]) }
        println!("IDXS:{:?}", indices);
        let reused = mgr.get_free_idx();
        assert_eq!(reused, indices[5], "Released index should be reused first");
    }

    #[test]
    fn fills_multiple_bitmaps_correctly() {
        // 130 blocks => 3 bitmaps (64 + 64 + 2)
        let mut mgr = FreeIdxManager::new(130);

        // Allocate all
        let mut allocated = Vec::new();
        for _ in 0..130 {
            let idx = mgr.get_free_idx();
            assert_ne!(idx, u32::MAX);
            allocated.push(idx);
        }

        // All full now
        assert_eq!(mgr.get_free_idx(), u32::MAX);

        // Release last one and ensure it reappears
        unsafe { mgr.unchecked_retire(allocated[129]) }
        let idx = mgr.get_free_idx();
        assert_eq!(idx, allocated[129]);
    }

    #[test]
    fn multiple_release_and_reuse_order() {
        let mut mgr = FreeIdxManager::new(64);

        let mut allocs = Vec::new();
        for _ in 0..64 {
            allocs.push(mgr.get_free_idx());
        }
        assert_eq!(mgr.get_free_idx(), u32::MAX);

        // Free 10 arbitrary blocks
        for &i in &allocs[10..20] {
            println!("RELEASE_ID:{i}");
            unsafe { mgr.unchecked_retire(i) }
        }

        // Should allocate from freed slots
        for expected in &allocs[10..20] {
            let got = mgr.get_free_idx();
            assert_eq!(got, *expected, "Should reuse freed index {:?}", expected);
        }
    }

    #[test]
    fn release_respects_curr_idx_buffering() {
        let mut mgr = FreeIdxManager::new(64);

        // Fill all
        let mut allocated = Vec::new();
        for _ in 0..64 {
            allocated.push(mgr.get_free_idx());
        }

        // curr_idx should now point to last (0)
        assert_eq!(mgr.curr_idx, 0);

        // Release last one (should re-add freelist entry safely)
        unsafe { mgr.unchecked_retire(allocated[63]) }

        // curr_idx should'nt incremented (buffer write worked)
        assert_eq!(mgr.curr_idx, 0);

        // Next get_free_idx should reuse it
        let idx = mgr.get_free_idx();
        assert_eq!(idx, allocated[63]);
    }

    fn alloc_all(mgr: &mut FreeIdxManager, n: usize) -> Vec<u32> {
        let mut res = Vec::new();
        for _ in 0..n {
            let idx = mgr.get_free_idx();
            assert_ne!(idx, FreeIdxManager::NULL_IDX);
            res.push(idx);
        }
        res
    }

    #[test]
    fn alloc_release_realloc_basic() {
        let mut mgr = FreeIdxManager::new(64);
        let all = alloc_all(&mut mgr, 64);

        // fully allocated
        assert_eq!(mgr.get_free_idx(), FreeIdxManager::NULL_IDX);

        // unchecked_retire one block safely
        unsafe { mgr.unchecked_retire(all[10]) }

        // should reuse it again
        let reused = mgr.get_free_idx();
        assert_eq!(reused, all[10]);
    }

    #[test]
    fn release_updates_bitmap_and_curr_idx() {
        let mut mgr = FreeIdxManager::new(128);
        let all = alloc_all(&mut mgr, 128);

        let prev_curr = mgr.curr_idx;
        unsafe { mgr.unchecked_retire(all[50]) }

        // bitmap[map_idx] must have at least one bit set again
        let idx = all[50] + 1;
        let map_idx = idx >> FreeIdxManager::DIV_BY;
        assert!(mgr.bitmap[map_idx as usize] != 0);

        // since map_idx != 0, curr_idx may increment
        assert!(mgr.curr_idx >= prev_curr);
    }

    #[test]
    fn release_does_not_overflow_free_list_buffer() {
        let mut mgr = FreeIdxManager::new(64);
        let all = alloc_all(&mut mgr, 64);

        // unchecked_retire last few safely — ensures we never touch out of bound free_list slot
        for &idx in all.iter().rev().take(5) {
            unsafe { mgr.unchecked_retire(idx) }
        }

        assert!(
            (mgr.curr_idx as usize) < mgr.free_list.len(),
            "curr_idx must never exceed free_list length"
        );
    }

    #[test]
    fn releasing_multiple_blocks_refills_in_reverse_order() {
        let mut mgr = FreeIdxManager::new(32);
        let all = alloc_all(&mut mgr, 32);

        unsafe {
            mgr.unchecked_retire(all[10]);
            mgr.unchecked_retire(all[11]);
            mgr.unchecked_retire(all[12]);
        }

        // order of reuse depends on free_list stack behavior — validate any freed ones are reused first
        let mut new_allocs = Vec::new();
        for _ in 0..3 {
            let idx = mgr.get_free_idx();
            new_allocs.push(idx);
        }

        assert!(
            new_allocs.iter().all(|i| all[10..13].contains(i)),
            "must reuse released indices first"
        );
    }

    #[test]
    fn releasing_first_block_does_not_increment_curr_idx_due_to_map_idx_zero() {
        let mut mgr = FreeIdxManager::new(64);
        let all = alloc_all(&mut mgr, 64);

        println!(
            "{:?},\n{:?},\n{:?}",
            mgr.bitmap, mgr.free_list, mgr.curr_idx
        );
        let prev = mgr.curr_idx;
        unsafe { mgr.unchecked_retire(all[0]) }

        // map_idx == 0 => curr_idx should NOT increment
        assert_eq!(mgr.curr_idx, prev);
    }

    #[test]
    fn test_occupancy_tracking_consistency() {
        let mut mgr = FreeIdxManager::new(128);

        let mut allocated = Vec::new();

        for _ in 0..50 {
            let idx = mgr.get_free_idx();
            if idx != FreeIdxManager::NULL_IDX {
                allocated.push(idx);
            }
        }

        let free_before = mgr.free_slots();

        // retire half
        for &idx in &allocated[..25] {
            mgr.retire(idx).unwrap();
        }

        let free_after = mgr.free_slots();

        assert!(
            free_after > free_before,
            "free_slots did not increase after retire"
        );

        // reallocate
        for _ in 0..25 {
            let _ = mgr.get_free_idx();
        }

        assert_eq!(
            mgr.free_slots(),
            free_before,
            "free_slots inconsistent after reuse cycle"
        );
    }

    #[test]
    fn test_rebuild_failure_does_not_mutate_state() {
        let mut mgr = FreeIdxManager::new(128);

        // create some state
        for _ in 0..20 {
            let _ = mgr.get_free_idx();
        }

        let bitmap_before = mgr.bitmap.clone();
        let freelist_before = mgr.free_list.clone();
        let curr_before = mgr.curr_idx;

        // invalid rebuild (shrink)
        let res = mgr.rebuild(64);

        assert!(res.is_err());

        assert_eq!(mgr.bitmap, bitmap_before, "bitmap mutated on failure");
        assert_eq!(mgr.free_list, freelist_before, "freelist mutated on failure");
        assert_eq!(mgr.curr_idx, curr_before, "curr_idx mutated on failure");
    }

    #[test]
    #[should_panic]
    fn test_unchecked_retire_out_of_bounds_panics() {
        let mut mgr = FreeIdxManager::new(64);

        unsafe {
            mgr.unchecked_retire(9999); // UB or segfault
        }
    }

    #[test]
    fn test_rebuild_within_same_segment_preserves_access() {
        let mut mgr = FreeIdxManager::new(10);

        // exhaust
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // rebuild within same u64 segment
        mgr.rebuild(50).unwrap();

        let idx = mgr.get_free_idx();

        assert!(
            idx != FreeIdxManager::NULL_IDX,
            "rebuild within same segment failed to expose new capacity"
        );
    }

    #[test]
    fn test_rebuild_top_is_immediately_valid() {
        let mut mgr = FreeIdxManager::new(64);

        // exhaust everything
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // rebuild adds new maps
        mgr.rebuild(256).unwrap();

        let top = mgr.free_list[mgr.curr_idx as usize] as usize;

        assert!(
            mgr.bitmap[top] != 0 || mgr.curr_idx == 0,
            "top is stale immediately after rebuild"
        );
    }

    #[test]
    fn test_terminal_map_zero_after_rebuild() {
        let mut mgr = FreeIdxManager::new(64);

        // exhaust everything
        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        // ensure map 0 is empty
        assert_eq!(mgr.bitmap[0], 0);

        // rebuild
        mgr.rebuild(128).unwrap();

        // allocator should now find new space
        let idx = mgr.get_free_idx();

        assert!(
            idx != FreeIdxManager::NULL_IDX,
            "allocator stuck on empty map 0 after rebuild"
        );
    }

    #[test]
    fn test_rebuild_ensures_reachable_capacity() {
        let mut mgr = FreeIdxManager::new(64);

        while mgr.get_free_idx() != FreeIdxManager::NULL_IDX {}

        mgr.rebuild(256).unwrap();

        let mut found = false;

        for _ in 0..10 {
            if mgr.get_free_idx() != FreeIdxManager::NULL_IDX {
                found = true;
                break;
            }
        }

        assert!(
            found,
            "rebuild produced capacity but allocator cannot reach it"
        );
    }

    
}
