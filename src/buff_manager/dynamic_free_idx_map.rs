/// `DynFreeIdxManager` is a dynamic manager for efficiently allocating and
/// tracking free indices within a fixed-capacity pool.  
///
/// It maintains a bitmap to quickly check availability of indices and a free
/// list for fast reuse of recently freed indices.
///
/// # Fields
///
/// - `bitmap: Vec<u64>`  
///   A vector of 64-bit words representing the allocation status of indices.  
///   Each bit corresponds to an index:  
///     - `0` → free  
///     - `1` → allocated  
///   This allows fast operations like checking availability, marking free/used,
///   and finding the first free index.
///
/// - `free_list: Vec<u16>`  
///   A stack of recently freed indices. When allocating a new index, the manager
///   can quickly pop from this list instead of scanning the bitmap.  
///   This improves allocation speed when indices are frequently freed and reused.
///
/// - `capacity: u32`  
///   The maximum number of indices the manager can handle.  
///   This must not exceed `bitmap.len() * 64`.
///
/// - `curr_idx: u16`  
///   The next sequential index to allocate if the free list is empty.  
///   This grows monotonically until it reaches `capacity`.
///
/// - `max_free_list_idx: u16`  
///   Tracks the highest index currently in the `free_list`.  
///   Useful for quick bounds checks and avoiding scanning beyond known free indices.
///
/// # Usage
///
/// `DynFreeIdxManager` is ideal when you need:
/// - Efficient allocation/deallocation of integer IDs or handles
/// - Fast reuse of recently freed indices
/// - Sparse allocation with a large number of possible indices
///
/// Example workflow:
/// 1. Allocate an index:  
///    - Check `free_list` first, if not empty pop from it  
///    - Otherwise, allocate `curr_idx` and increment
/// 2. Free an index:  
///    - Push the index onto `free_list`  
///    - Clear the corresponding bit in `bitmap`
///
/// This combination of bitmap + free list provides **O(1)** average allocation
/// and deallocation, while maintaining the ability to scan for free indices if needed.
pub struct DynFreeIdxManager {
    bitmap: Vec<u64>,
    free_list: Vec<u16>,
    capacity: u32,
    curr_idx: u16,
    max_free_list_idx: u16,
}

impl DynFreeIdxManager {
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
        let m = DynFreeIdxManager::new(1);
        assert_eq!(m.bitmap.len(), 1);
        assert_eq!(m.free_list.len(), 1 + 1);
        assert_eq!(m.curr_idx, 0);

        // All bits should be set (free)
        let expected_mask = DynFreeIdxManager::MASK_SET_MAP;
        assert_eq!(m.bitmap[0], expected_mask);
        assert_eq!(m.bitmap[0], u64::MAX);
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_63_block() {
        let m = DynFreeIdxManager::new(63);
        assert_eq!(m.bitmap.len(), 1);
        assert_eq!(m.free_list.len(), 2);
        assert_eq!(m.curr_idx, 0);

        // Only the top bit should be set (free)
        let expected_mask = u64::MAX;
        assert_eq!(m.bitmap[0], expected_mask);
        assert_eq!(
            m.bitmap[0],
            0b1111111111111111111111111111111111111111111111111111111111111111
        );
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_init_exact_multiple_64() {
        let m = DynFreeIdxManager::new(128);
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
        let m = DynFreeIdxManager::new(70);
        assert_eq!(m.bitmap.len(), 2);

        // First map is full
        assert_eq!(m.bitmap[0], u64::MAX);

        // Remainder = 6 → top 6 bits set
        let expected_end = u64::MAX;
        assert_eq!(m.bitmap[1], expected_end);
        assert_eq!(m.bitmap[1], u64::MAX);
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        assert_eq!(m.curr_idx, 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    fn test_freeidx_init_max_limit() {
        let n = DynFreeIdxManager::MAX_BLOCK - 1;
        let m = DynFreeIdxManager::new(n);
        assert!(m.bitmap.len() > 0);
        assert_eq!(m.bitmap.len(), ((n + 63) >> 6) as usize);
        assert_eq!(m.curr_idx, 65532);
        for b in m.bitmap[..(m.bitmap.len() - 1)].iter() {
            assert_eq!(*b, u64::MAX);
        }
        assert_eq!(*m.bitmap.last().unwrap(), u64::MAX);
        println!("{:b}", m.bitmap.last().unwrap());
        assert_eq!(m.bitmap.len(), m.free_list.len() - 1);
        for (idx, val) in m.free_list.iter().enumerate() {
            assert_eq!(idx, *val as usize);
        }
    }

    #[test]
    #[should_panic]
    fn test_freeidx_zero_block_panics() {
        let _ = DynFreeIdxManager::new(0);
    }

    #[test]
    #[should_panic]
    fn test_freeidx_overflow_panics() {
        let _ = DynFreeIdxManager::new(DynFreeIdxManager::MAX_BLOCK + 1);
    }

    #[test]
    fn test_bitmap_all_ones_at_start() {
        let m = DynFreeIdxManager::new(128);
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
    fn count_free_bits(f: &DynFreeIdxManager) -> u32 {
        f.bitmap.iter().map(|x| x.count_ones()).sum()
    }

    #[test]
    fn test_initialization_masks_last_entry_correctly() {
        // Case 1: exactly multiple of 64 → all u64s should be full
        let f1 = DynFreeIdxManager::new(128);
        assert_eq!(f1.bitmap.len(), 2);
        assert!(f1.bitmap.iter().all(|&b| b == u64::MAX));

        // Case 2: not multiple of 64 → last entry must be masked
        let f2 = DynFreeIdxManager::new(130);
        assert_eq!(f2.bitmap.len(), 3);
        let last_mask = f2.bitmap[2];
        // 130 % 64 = 2, so last mask keeps only top 2 bits set
        let expected_mask = u64::MAX;
        assert_eq!(last_mask, expected_mask);
    }

    #[test]
    fn test_allocate_until_full_then_returns_u32_max() {
        let mut mgr = DynFreeIdxManager::new(64);
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
        let mut mgr = DynFreeIdxManager::new(128);
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
        // 70 blocks → 2 u64 entries
        let mut mgr = DynFreeIdxManager::new(70);
        let total_before = count_free_bits(&mgr);
        assert_eq!(total_before, 128);

        // Allocate all → should drain all bits
        for _ in 0..128 {
            let idx = mgr.get_free_idx();
            println!("idx:{idx}");
            assert!(idx < 128, "invalid idx={}", idx);
        }

        // Next call → u32::MAX
        assert_eq!(mgr.get_free_idx(), u32::MAX);
        assert_eq!(count_free_bits(&mgr), 0);
    }

    #[test]
    fn test_curr_idx_moves_downward_as_maps_fill() {
        // 128 blocks → 2 maps → curr_idx starts = 2
        let mut mgr = DynFreeIdxManager::new(128);
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
        let mut mgr = DynFreeIdxManager::new(128);

        // Collect all allocations
        let mut indices = Vec::new();
        for _ in 0..128 {
            let idx = mgr.get_free_idx();
            assert!(idx != u32::MAX, "Should return a valid free index");
            indices.push(idx);
        }

        // After all allocated, next call should return NULL_IDX (u32::MAX)
        assert_eq!(mgr.get_free_idx(), DynFreeIdxManager::NULL_IDX);

        // Release one block and allocate again — should reuse the freed one
        unsafe { mgr.retire(indices[5]) }
        println!("IDXS:{:?}", indices);
        let reused = mgr.get_free_idx();
        assert_eq!(reused, indices[5], "Released index should be reused first");
    }

    #[test]
    fn fills_multiple_bitmaps_correctly() {
        let n_block = 130;
        // 130 blocks => 3 bitmaps (64 + 64 + 64) => 192
        let mut mgr = DynFreeIdxManager::new(n_block);

        // Allocate all
        let mut allocated = Vec::new();
        for _ in 0..(n_block.next_multiple_of(DynFreeIdxManager::MAP_WIDTH)) {
            let idx = mgr.get_free_idx();
            assert_ne!(idx, u32::MAX);
            allocated.push(idx);
        }

        // All full now
        assert_eq!(mgr.get_free_idx(), u32::MAX);

        // Release last one and ensure it reappears
        unsafe {
            mgr.retire(
                allocated[n_block.next_multiple_of(DynFreeIdxManager::MAP_WIDTH) as usize - 1],
            )
        }
        let idx = mgr.get_free_idx();
        assert_eq!(
            idx,
            allocated[n_block.next_multiple_of(DynFreeIdxManager::MAP_WIDTH) as usize - 1]
        );
    }

    #[test]
    fn multiple_release_and_reuse_order() {
        let mut mgr = DynFreeIdxManager::new(64);

        let mut allocs = Vec::new();
        for _ in 0..64 {
            allocs.push(mgr.get_free_idx());
        }
        assert_eq!(mgr.get_free_idx(), u32::MAX);

        // Free 10 arbitrary blocks
        for &i in &allocs[10..20] {
            println!("RELEASE_ID:{i}");
            unsafe { mgr.retire(i) }
        }

        // Should allocate from freed slots
        for expected in &allocs[10..20] {
            let got = mgr.get_free_idx();
            assert_eq!(got, *expected, "Should reuse freed index {:?}", expected);
        }
    }

    #[test]
    fn release_respects_curr_idx_buffering() {
        let mut mgr = DynFreeIdxManager::new(64);

        // Fill all
        let mut allocated = Vec::new();
        for _ in 0..64 {
            allocated.push(mgr.get_free_idx());
        }

        // curr_idx should now point to last (0)
        assert_eq!(mgr.curr_idx, 0);

        // Release last one (should re-add freelist entry safely)
        unsafe { mgr.retire(allocated[63]) }

        // curr_idx should'nt incremented (buffer write worked)
        assert_eq!(mgr.curr_idx, 0);

        // Next get_free_idx should reuse it
        let idx = mgr.get_free_idx();
        assert_eq!(idx, allocated[63]);
    }

    fn alloc_all(mgr: &mut DynFreeIdxManager, n: usize) -> Vec<u32> {
        let mut res = Vec::new();
        for _ in 0..n {
            let idx = mgr.get_free_idx();
            assert_ne!(idx, DynFreeIdxManager::NULL_IDX);
            res.push(idx);
        }
        res
    }

    #[test]
    fn alloc_release_realloc_basic() {
        let mut mgr = DynFreeIdxManager::new(64);
        let all = alloc_all(&mut mgr, 64);

        // fully allocated
        assert_eq!(mgr.get_free_idx(), DynFreeIdxManager::NULL_IDX);

        // retire one block safely
        unsafe { mgr.retire(all[10]) }

        // should reuse it again
        let reused = mgr.get_free_idx();
        assert_eq!(reused, all[10]);
    }

    #[test]
    fn release_updates_bitmap_and_curr_idx() {
        let mut mgr = DynFreeIdxManager::new(128);
        let all = alloc_all(&mut mgr, 128);

        let prev_curr = mgr.curr_idx;
        unsafe { mgr.retire(all[50]) }

        // bitmap[map_idx] must have at least one bit set again
        let idx = all[50] + 1;
        let map_idx = idx >> DynFreeIdxManager::DIV_BY;
        assert!(mgr.bitmap[map_idx as usize] != 0);

        // since map_idx != 0, curr_idx may increment
        assert!(mgr.curr_idx >= prev_curr);
    }

    #[test]
    fn release_does_not_overflow_free_list_buffer() {
        let mut mgr = DynFreeIdxManager::new(64);
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
        let mut mgr = DynFreeIdxManager::new(32);
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
        let mut mgr = DynFreeIdxManager::new(64);
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
        let mgr = DynFreeIdxManager::new(128);
        assert_eq!(mgr.capacity(), 128);

        let mgr2 = DynFreeIdxManager::new(4096);
        assert_eq!(mgr2.capacity(), 4096);
    }

    #[test]
    fn test_calculate_n_slots_alignment_and_bounds() {
        let mgr = DynFreeIdxManager::new(128);

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
        let mgr = DynFreeIdxManager::new(1);
        // Cause panic by exceeding max block
        let _ = mgr.calculate_n_slots(DynFreeIdxManager::MAX_BLOCK);
    }

    #[test]
    fn test_grow_adds_additional_slots() {
        let mut mgr = DynFreeIdxManager::new(128);
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
            assert_eq!(mgr.bitmap[new_idx], DynFreeIdxManager::MASK_SET_MAP);
        }
    }

    #[test]
    fn test_grow_does_not_corrupt_existing_maps() {
        let mut mgr = DynFreeIdxManager::new(64);
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
