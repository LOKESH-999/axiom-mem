/// `DynFreeIdxManagerV1` — Dynamic Free-Index Manager with Bitmap + Freelist Hybrid.
///
/// This structure manages allocation and tracking of free indices within a
/// fixed-capacity range. It combines a **bitmap** for O(1) occupancy tracking
/// with a **freelist** for O(1) reuse of freed indices.
///
/// # Overview
///
/// Each bit in `bitmap` represents a resource slot:
/// - `0` → occupied
/// - `1` → free
///
/// The bitmap is segmented into `u64` blocks (each representing 64 slots),
/// aligned to 64-bit boundaries for word-sized atomic operations.  
/// When the number of slots (`n_block`) is **not a multiple of 64**, the manager
/// **rounds up** to the next 64-bit boundary. This ensures that every allocation
/// operation remains word-aligned without requiring tail masking except in the
/// initialization phase.
///
/// Internally, the manager rounds up the total block count to the next 64-bit
/// boundary:
///
/// ```rust
/// let n_block = 6u32.next_multiple_of(64);
/// assert_eq!(n_block,64);
/// ```
///
/// This means the allocator **always initializes a fully aligned 64-bit bitmap,**
/// even if the requested capacity isn’t a multiple of 64.  
/// No tail masking is required — all bits in the final `u64` entry are valid
/// and represent usable slots.
///
/// # Memory Layout
///
/// ```text
/// Bitmap (64-bit aligned)
/// ┌────────────┬────────────┬────────────┬────────────┐
/// │ block[0]   │ block[1]   │ ...        │ block[n]   │
/// └────────────┴────────────┴────────────┴────────────┘
///   ↑ each bit = one slot
///
/// Example (for 130 requested slots):
/// - Rounds up to 192 (3 × 64) total slots.
/// - All 192 bits are valid and managed.
/// - The extra 62 bits beyond 130 are usable and tracked normally.
/// ```
///
/// This approach ensures uniformity in bitmap arithmetic and allows all
/// operations (`get_free_idx`, `retire`, `is_free`, `grow`) to work purely on
/// 64-bit aligned words without conditional masks or partial block handling.
/// 
/// # Core Fields
///
/// - `bitmap: Vec<u64>`  
///   Tracks the availability of indices.  
///   - Each `u64` word encodes 64 index states.
///   - Bits set to `1` are free; `0` are allocated.
///   - Rounded up to the next multiple of 64.
///
/// - `free_list: Vec<u16>`  
///   Stores indices of bitmap blocks (`map_idx`) that still contain free bits.  
///   This allows the allocator to skip full blocks and jump directly to those
///   with available capacity.  
///   The freelist acts like a stack: allocation pulls from the top (`curr_idx`),
///   and freeing a block pushes it back if it transitions from full → partially free.
///
/// - `capacity: u32`  
///   Total number of managed slots (rounded up to multiple of 64).
///
/// - `curr_idx: u16`  
///   Current freelist head — points to the active bitmap block index.
///
/// - `max_free_list_idx: u16`  
///   Upper bound of freelist indices; aids in growing or bounds validation.
///
/// # Rounding Semantics
///
/// Every new instance of the manager applies 64-bit alignment:
/// ```rust
/// let n_block = 32u32.next_multiple_of(64);
/// ```
///
/// This ensures:
/// - All bitmap operations are word-aligned.
/// - No unaligned partial-block writes.
/// - Easier vectorized or atomic word updates.
///
/// # Example
///
/// ```rust
/// use axiom_mem::buff_manager::dynamic_free_idx_map_v1::DynFreeIdxManagerV1;
/// let mut mgr = DynFreeIdxManagerV1::new(130);
///
/// let a = mgr.get_free_idx(); // alloc slot 191
/// let b = mgr.get_free_idx(); // alloc slot 190
///
/// unsafe { mgr.retire(a); }  // free slot 191
///
/// assert!(mgr.is_free(a));
/// assert!(!mgr.is_free(b));
/// ```
///
/// # Invariants
///
/// - All `bitmap[i]` are valid `u64` blocks aligned to `MAP_WIDTH = 64`.
/// - `free_list` always has one extra “buffer” entry to prevent OOB writes.
/// - `curr_idx` never exceeds `max_free_list_idx`.
/// - `capacity` is always a multiple of 64 (rounded up).
///
/// Violating any of these invariants (e.g., double-retiring an index, or freeing
/// an invalid ID) results in undefined allocator state.
///
/// # Complexity
///
/// - Allocation: **O(1)** average  
/// - Deallocation: **O(1)**  
/// - Capacity growth: **O(Δ)** where Δ = number of new blocks
///
/// # Safety
///
/// Some methods (`retire`, `is_free`) are `unsafe` because they depend on caller
/// guarantees about index validity and unique free/alloc pairs.
///
/// # Internal Notes (for maintainers)
///
/// - `DIV_BY = 6` and `MAP_WIDTH = 64` define the base-2 logarithmic relationship.
/// - The bitmap uses **MSB-first bit numbering** for consistent trailing-zero scans.
/// - Tail bits beyond `capacity` in the final block are masked off during init.
/// - The freelist intentionally keeps one extra slot as a buffer to avoid write races.
///
/// # Example Internal State (128-slot pool)
///
/// ```text
/// bitmap[0] = 1111111111111111111111111111111111111111111111111111111111111111
/// bitmap[1] = 1111111111111111111111111111111111111111111111111111111111111111
/// curr_idx  = 1
/// free_list = [0, 1, _buf_]
/// ```
///
/// After allocating 5 slots:
///
/// ```text
/// bitmap[0] = 1111111111111111111111111111111111111111111111111111111111100000
/// ```
///
/// After retiring one:
///
/// ```text
/// bitmap[0] = 1111111111111111111111111111111111111111111111111111111111100001
/// ```
///
/// This design ensures both **constant-time allocation/deallocation**
/// and **bit-level precision** with predictable 64-bit alignment behavior.
pub struct DynFreeIdxManagerV1 {
    bitmap: Vec<u64>,
    free_list: Vec<u16>,
    capacity: u32,
    curr_idx: u16,
    max_free_list_idx: u16,
}

impl DynFreeIdxManagerV1 {
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
    /// 4,194,112 = 2^22 blocks → supports large pool sizes with compact metadata.
    pub const MAX_BLOCK: u32 = 4_194_112;

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
        let n_block = n_block.next_multiple_of(Self::MAP_WIDTH);

        assert!(n_block > 0, "Number of blocks must be greater than zero");
        assert!(n_block <= Self::MAX_BLOCK, "Exceeded maximum block count");

        // Each u64 entry represents 64 blocks.
        let len = ((n_block + 63) >> Self::DIV_BY) as u16;

        // Initialize free list with linear indices of each bitmap entry.
        let free_list = (0..(len + 1)).collect::<Vec<_>>();

        // Initially, all blocks are free (all bits = 1).
        let mut bitmap = (0..len).map(|_| Self::MASK_SET_MAP).collect::<Vec<_>>();

        // Calculate remaining blocks in the last 64-bit segment.
        let reminder = n_block as u64 & Self::MASK_64;

        // If not a perfect multiple of 64, mask off invalid tail bits.
        if reminder != 0 {
            let end_mask = Self::MASK_SET_MAP << (Self::MAP_WIDTH - reminder as u32);
            bitmap[(len - 1) as usize] = end_mask;
        }
        let max_free_list_idx = (free_list.len() - 1) as u16;
        let curr_idx = (bitmap.len() - 1) as u16;
        Self {
            bitmap,
            free_list,
            capacity: n_block,
            curr_idx,
            max_free_list_idx,
        }
    }

    /// Returns the index of a free slot, or `u32::MAX` if all blocks are full.
    ///
    /// This function performs a **branchless lookup**:
    /// - Finds the first available bit (`trailing_zeros`).
    /// - Clears that bit (marks block as used).
    /// - Updates `curr_idx` if this bitmap chunk becomes empty.
    /// - Returns the global block index, or `u32::MAX` if none are free.
    pub fn get_free_idx(&mut self) -> u32 {
        // Load current bitmap index from the freelist
        let map_idx = self.free_list[self.curr_idx as usize];

        // Find first free bit in this 64-bit map (returns 64 if none)
        let map_res = self.bitmap[map_idx as usize].trailing_zeros();
        // println!("MAP_RES:{}",map_res);

        // Mark that bit as occupied
        self.bitmap[map_idx as usize] &= !(1u64 << map_res);

        // Move to previous freelist entry if this map is now full
        self.curr_idx -= (self.bitmap[map_idx as usize] == 0) as u16 & (self.curr_idx != 0) as u16;

        // Compute global block index or return `u32::MAX` (branchless)
        ((-((map_res != 64) as i32)).cast_unsigned()
            & ((Self::MAP_WIDTH - map_res - 1) + (map_idx as u32 * Self::MAP_WIDTH)))
            | (-((map_res == 64) as i32)).cast_unsigned()
    }

    /// # Safety
    /// Caller must ensure:
    /// - `idx` < total number of blocks managed.
    /// - The same `idx` is not released twice without a reallocation.
    ///   Violating either may corrupt the bitmap state or freelist tracking.
    pub unsafe fn retire(&mut self, id: u32) {
        let idx = id;
        // dividing it by 64 inorder to find map-index.
        let map_idx = idx >> Self::DIV_BY;
        // if `self.bitmap[map_idx as usize] == 0` then we need to add a marking in `free_list`.
        let is_to_add = (self.bitmap[map_idx as usize] == 0) & (map_idx != 0);
        // marking the `idx` in its `bitmap` slot.
        self.bitmap[map_idx as usize] |=
            1u64 << (Self::MAP_WIDTH as u64 - ((idx + 1) as u64 & Self::MASK_64));
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
    pub fn is_free(&self, idx: u32) -> bool {
        // dividing it by 64 inorder to find map-index.
        let map_idx = idx >> Self::DIV_BY;
        let mask = 1u64 << (Self::MAP_WIDTH as u64 - ((idx + 1) as u64 & Self::MASK_64));
        self.bitmap[map_idx as usize] & mask == mask
    }

    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    // NOTE (for maintainers):
    // This function is internal and assumes the developer ensures that
    // `delta + max_free_list_idx` does not overflow `u16::MAX`.
    // Violating this invariant will corrupt the freelist structure.
    // This is not a runtime guarantee — it is a contract for anyone
    // rewriting or extending these core allocator routines.
    pub fn calculate_n_slots(&self, delta: u32) -> u16 {
        let new_delta = delta.next_multiple_of(Self::MAP_WIDTH);
        if self.capacity.wrapping_add(new_delta) <= Self::MAX_BLOCK {
            return (new_delta >> Self::DIV_BY) as u16;
        }
        panic!(" `new_delta` must be <= `Self::MAX_BLOCK` ")
    }

    // NOTE (for maintainers):
    // This function is internal and assumes the developer ensures that
    // `delta + max_free_list_idx` does not overflow `u16::MAX`.
    // Violating this invariant will corrupt the freelist structure.
    // This is not a runtime guarantee — it is a contract for anyone
    // rewriting or extending these core allocator routines.
    pub fn grow(&mut self, delta: u16) {
        for _ in 0..delta {
            self.bitmap.push(Self::MASK_SET_MAP);
            self.max_free_list_idx += 1;
            self.free_list.push(self.max_free_list_idx);
            self.curr_idx += 1;
            self.bitmap[self.curr_idx as usize] = Self::MASK_SET_MAP;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

       #[test]
    fn test_freeidx_init_min_block() {
        let m = DynFreeIdxManagerV1::new(1);
        // Rounded up to 64
        assert_eq!(m.bitmap.len(), 1);
        assert_eq!(m.free_list.len(), 2);
        assert_eq!(m.curr_idx, 0);
        assert_eq!(m.bitmap[0], u64::MAX);
        assert_eq!(m.capacity, 64);
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_63_block() {
        let m = DynFreeIdxManagerV1::new(63);
        // Rounded up to 64
        assert_eq!(m.bitmap.len(), 1);
        assert_eq!(m.free_list.len(), 2);
        assert_eq!(m.curr_idx, 0);
        assert_eq!(m.bitmap[0], u64::MAX);
        assert_eq!(m.capacity, 64);
    }

    #[test]
    fn test_freeidx_init_exact_multiple_64() {
        let m = DynFreeIdxManagerV1::new(128);
        assert_eq!(m.bitmap.len(), 2);
        assert_eq!(m.free_list.len(), 3);
        assert_eq!(m.curr_idx, 1);
        assert!(m.bitmap.iter().all(|&b| b == u64::MAX));
        assert_eq!(m.capacity, 128);
    }

    #[test]
    fn test_freeidx_init_partial_map_rounded() {
        let m = DynFreeIdxManagerV1::new(70);
        // Rounded up to 128 (2 × 64)
        assert_eq!(m.bitmap.len(), 2);
        assert_eq!(m.capacity, 128);
        assert!(m.bitmap.iter().all(|&b| b == u64::MAX));
        assert_eq!(m.curr_idx, 1);
    }

    #[test]
    fn test_freeidx_init_max_limit() {
        let n = DynFreeIdxManagerV1::MAX_BLOCK - 1;
        let m = DynFreeIdxManagerV1::new(n);
        assert_eq!(m.bitmap.len(), ((n.next_multiple_of(64) + 63) >> 6) as usize);
        assert!(m.bitmap.iter().all(|&b| b == u64::MAX));
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        assert!(m.curr_idx <= m.bitmap.len() as u16 - 1);
    }

    #[test]
    #[should_panic]
    fn test_freeidx_zero_block_panics() {
        let _ = DynFreeIdxManagerV1::new(0);
    }

    #[test]
    #[should_panic]
    fn test_freeidx_overflow_panics() {
        let _ = DynFreeIdxManagerV1::new(DynFreeIdxManagerV1::MAX_BLOCK + 1);
    }

    #[test]
    fn test_bitmap_all_ones_at_start() {
        let m = DynFreeIdxManagerV1::new(128);
        assert!(m.bitmap.iter().all(|&b| b == u64::MAX));
        assert_eq!(m.capacity, 128);
        assert_eq!(m.curr_idx, 1);
    }

    fn count_free_bits(f: &DynFreeIdxManagerV1) -> u32 {
        f.bitmap.iter().map(|x| x.count_ones()).sum()
    }

    #[test]
    fn test_initialization_no_masking() {
        // Rounded up to full 64-bit words
        let f1 = DynFreeIdxManagerV1::new(130);
        // 130 → 192 total (3×64)
        assert_eq!(f1.bitmap.len(), 3);
        assert!(f1.bitmap.iter().all(|&b| b == u64::MAX));
        assert_eq!(f1.capacity, 192);
    }

    #[test]
    fn test_allocate_until_full_then_returns_u32_max() {
        let mut mgr = DynFreeIdxManagerV1::new(64);
        for _ in 0..64 {
            let idx = mgr.get_free_idx();
            assert!(idx < 64);
        }
        assert_eq!(mgr.get_free_idx(), u32::MAX);
        assert_eq!(count_free_bits(&mgr), 0);
    }

    #[test]
    fn test_multi_map_progression() {
        let mut mgr = DynFreeIdxManagerV1::new(128);
        for _ in 0..128 {
            assert!(mgr.get_free_idx() < 128);
        }
        assert_eq!(mgr.get_free_idx(), u32::MAX);
        assert_eq!(count_free_bits(&mgr), 0);
    }

    #[test]
    fn test_partial_map_behavior_rounded() {
        // 70 requested → 128 capacity
        let mut mgr = DynFreeIdxManagerV1::new(70);
        assert_eq!(count_free_bits(&mgr), 128);

        // Allocate all 128
        for _ in 0..128 {
            let idx = mgr.get_free_idx();
            assert!(idx < 128);
        }

        assert_eq!(mgr.get_free_idx(), u32::MAX);
        assert_eq!(count_free_bits(&mgr), 0);
    }

    #[test]
    fn test_curr_idx_moves_downward_as_maps_fill() {
        let mut mgr = DynFreeIdxManagerV1::new(128);
        assert_eq!(mgr.curr_idx, 1);

        for _ in 0..64 {
            mgr.get_free_idx();
        }
        assert_eq!(mgr.curr_idx, 0);

        for _ in 0..64 {
            mgr.get_free_idx();
        }
        assert_eq!(mgr.curr_idx, 0);
        assert_eq!(mgr.get_free_idx(), u32::MAX);
    }

    #[test]
    fn allocates_and_releases_correctly() {
        let mut mgr = DynFreeIdxManagerV1::new(128);
        let mut indices = Vec::new();

        for _ in 0..128 {
            indices.push(mgr.get_free_idx());
        }

        assert_eq!(mgr.get_free_idx(), DynFreeIdxManagerV1::NULL_IDX);

        unsafe { mgr.retire(indices[5]) }
        let reused = mgr.get_free_idx();
        assert_eq!(reused, indices[5]);
    }

    #[test]
    fn fills_multiple_bitmaps_correctly() {
        let n_block = 130;
        // 130 → 192 capacity (3 maps)
        let mut mgr = DynFreeIdxManagerV1::new(n_block);
        let total_capacity = n_block.next_multiple_of(DynFreeIdxManagerV1::MAP_WIDTH);
        let mut allocated = Vec::new();

        for _ in 0..total_capacity {
            let idx = mgr.get_free_idx();
            assert_ne!(idx, u32::MAX);
            allocated.push(idx);
        }

        assert_eq!(mgr.get_free_idx(), u32::MAX);

        unsafe {
            mgr.retire(allocated[total_capacity as usize - 1]);
        }
        let idx = mgr.get_free_idx();
        assert_eq!(idx, allocated[total_capacity as usize - 1]);
    }

    #[test]
    fn multiple_release_and_reuse_order() {
        let mut mgr = DynFreeIdxManagerV1::new(64);
        let mut allocs = Vec::new();
        for _ in 0..64 {
            allocs.push(mgr.get_free_idx());
        }
        assert_eq!(mgr.get_free_idx(), u32::MAX);

        for &i in &allocs[10..20] {
            unsafe { mgr.retire(i) }
        }

        for expected in &allocs[10..20] {
            let got = mgr.get_free_idx();
            assert_eq!(got, *expected);
        }
    }

    #[test]
    fn release_respects_curr_idx_buffering() {
        let mut mgr = DynFreeIdxManagerV1::new(64);
        let mut allocated = Vec::new();

        for _ in 0..64 {
            allocated.push(mgr.get_free_idx());
        }

        assert_eq!(mgr.curr_idx, 0);
        unsafe { mgr.retire(allocated[63]) }
        assert_eq!(mgr.curr_idx, 0);

        let idx = mgr.get_free_idx();
        assert_eq!(idx, allocated[63]);
    }

    fn alloc_all(mgr: &mut DynFreeIdxManagerV1, n: usize) -> Vec<u32> {
        let mut res = Vec::new();
        for _ in 0..n {
            let idx = mgr.get_free_idx();
            assert_ne!(idx, DynFreeIdxManagerV1::NULL_IDX);
            res.push(idx);
        }
        res
    }

    #[test]
    fn alloc_release_realloc_basic() {
        let mut mgr = DynFreeIdxManagerV1::new(64);
        let all = alloc_all(&mut mgr, 64);

        // fully allocated
        assert_eq!(mgr.get_free_idx(), DynFreeIdxManagerV1::NULL_IDX);

        // retire one block safely
        unsafe { mgr.retire(all[10]) }

        // should reuse it again
        let reused = mgr.get_free_idx();
        assert_eq!(reused, all[10]);
    }

    #[test]
    fn release_updates_bitmap_and_curr_idx() {
        let mut mgr = DynFreeIdxManagerV1::new(128);
        let all = alloc_all(&mut mgr, 128);

        let prev_curr = mgr.curr_idx;
        unsafe { mgr.retire(all[50]) }

        // bitmap[map_idx] must have at least one bit set again
        let idx = all[50] + 1;
        let map_idx = idx >> DynFreeIdxManagerV1::DIV_BY;
        assert!(mgr.bitmap[map_idx as usize] != 0);

        // since map_idx != 0, curr_idx may increment
        assert!(mgr.curr_idx >= prev_curr);
    }

    #[test]
    fn release_does_not_overflow_free_list_buffer() {
        let mut mgr = DynFreeIdxManagerV1::new(64);
        let all = alloc_all(&mut mgr, 64);

        // retire last few safely — ensures we never touch out of bound free_list slot
        for &idx in all.iter().rev().take(5) {
            unsafe { mgr.retire(idx) }
        }

        assert!(
            (mgr.curr_idx as usize) < mgr.free_list.len(),
            "curr_idx must never exceed free_list length"
        );
    }

    #[test]
    fn releasing_multiple_blocks_refills_in_reverse_order() {
        let mut mgr = DynFreeIdxManagerV1::new(32);
        let all = alloc_all(&mut mgr, 32);

        unsafe {
            mgr.retire(all[10]);
            mgr.retire(all[11]);
            mgr.retire(all[12]);
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
        let mut mgr = DynFreeIdxManagerV1::new(64);
        let all = alloc_all(&mut mgr, 64);

        println!(
            "{:?},\n{:?},\n{:?}",
            mgr.bitmap, mgr.free_list, mgr.curr_idx
        );
        let prev = mgr.curr_idx;
        unsafe { mgr.retire(all[0]) }

        // map_idx == 0 => curr_idx should NOT increment
        assert_eq!(mgr.curr_idx, prev);
    }

    // ============================================================
    // 🧩 TESTS FOR NEW FUNCTIONALITIES (capacity, calculate_n_slots, grow)
    // ============================================================

    #[test]
    fn test_capacity_returns_correct_value() {
        let mgr = DynFreeIdxManagerV1::new(128);
        assert_eq!(mgr.capacity(), 128);

        let mgr2 = DynFreeIdxManagerV1::new(4096);
        assert_eq!(mgr2.capacity(), 4096);
    }

    #[test]
    fn test_calculate_n_slots_alignment_and_bounds() {
        let mgr = DynFreeIdxManagerV1::new(128);

        // 1 → rounds up to next multiple of 64 → 64/64 = 1
        assert_eq!(mgr.calculate_n_slots(1), 1);

        // 63 → rounds up to 64 → 1 slot
        assert_eq!(mgr.calculate_n_slots(63), 1);

        // 64 → already multiple → 1 slot
        assert_eq!(mgr.calculate_n_slots(64), 1);

        // 65 → rounds up to 128 → 128/64 = 2 slots
        assert_eq!(mgr.calculate_n_slots(65), 2);

        // 128 → 128/64 = 2
        assert_eq!(mgr.calculate_n_slots(128), 2);
    }

    #[test]
    #[should_panic(expected = "`new_delta` must be <= `Self::MAX_BLOCK`")]
    fn test_calculate_n_slots_panics_on_exceeding_max_block() {
        let mgr = DynFreeIdxManagerV1::new(1);
        // Cause panic by exceeding max block
        let _ = mgr.calculate_n_slots(DynFreeIdxManagerV1::MAX_BLOCK);
    }

    #[test]
    fn test_grow_adds_additional_slots() {
        let mut mgr = DynFreeIdxManagerV1::new(128);
        let initial_bitmap_len = mgr.bitmap.len();
        let initial_free_list_len = mgr.free_list.len();
        let initial_max_idx = mgr.max_free_list_idx;

        // Grow by 4 → adds 4 new u64 maps
        mgr.grow(4);

        // Bitmap should have increased by 4
        assert_eq!(mgr.bitmap.len(), initial_bitmap_len + 4);
        assert_eq!(mgr.free_list.len(), initial_free_list_len + 4);

        // max_free_list_idx should have incremented by 4
        assert_eq!(mgr.max_free_list_idx, initial_max_idx + 4);

        // New entries should all be fully free
        for new_idx in initial_bitmap_len..mgr.bitmap.len() {
            assert_eq!(mgr.bitmap[new_idx], DynFreeIdxManagerV1::MASK_SET_MAP);
        }
    }

    #[test]
    fn test_grow_does_not_corrupt_existing_maps() {
        let mut mgr = DynFreeIdxManagerV1::new(64);
        let before = mgr.bitmap.clone();

        // Grow by 2 → adds 2 extra maps
        mgr.grow(2);

        // Existing maps remain untouched
        assert_eq!(mgr.bitmap[0], before[0]);
        // Newly grown maps are initialized as full (free)
        assert_eq!(mgr.bitmap[1], u64::MAX);
        assert_eq!(mgr.bitmap[2], u64::MAX);
    }
}
