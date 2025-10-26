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
//! - Constant-time allocation/retire
//! - Minimal metadata overhead

/// Bitmap-based index manager for tracking free/used blocks within the pool.
///
/// Each bit in the `bitmap` corresponds to a block:
/// - `1` → free block
/// - `0` → occupied block
///
/// The bitmap is accompanied by a `free_list` of `u16` indices, each mapping to
/// a 64-bit chunk in the bitmap. This allows faster iteration or caching of
/// indices with available space.
///
/// # Layout
/// For `n_block` total blocks:
/// - Number of 64-bit entries = `ceil(n_block / 64)`
/// - Each entry covers 64 blocks.
/// - If `n_block` is not a multiple of 64, the final entry is masked
///   so that bits beyond `n_block` are cleared (treated as unavailable).
///
/// # Example
/// ```text
/// n_block = 130
/// len = ceil(130 / 64) = 3
///
/// bitmap[0] = 0xFFFF_FFFF_FFFF_FFFF (64 free)
/// bitmap[1] = 0xFFFF_FFFF_FFFF_FFFF (64 free)
/// bitmap[2] = 0xFFFF_FFFF_FFFF_FC00 (only 2 valid bits set)
/// ```
pub struct FreeIdxManager {
    /// Bitmap representing free (1) or used (0) block states.
    bitmap: Box<[u64]>,
    /// Precomputed list of 64-bit map indices for faster traversal.
    free_list: Box<[u16]>,
    /// Indicates the curr_pointing index in `free_list`.
    curr_idx: u16,
}

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
        }
    }

    /// Returns the index of a free slot, or `u32::MAX` if all blocks are full.
    ///
    /// This function performs a **branchless lookup**:
    /// - Finds the first available bit (`trailing_zeros`).
    /// - Clears that bit (marks block as used).
    /// - Updates `curr_idx` if this bitmap chunk becomes empty.
    /// - Returns the global block index, or `u32::MAX` if none are free.
    pub const fn get_free_idx(&mut self) -> u32 {
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
    pub const unsafe fn retire(&mut self, id: u32) {
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
    pub const unsafe fn is_free(&self, idx: u32) -> bool {
        let map_idx = idx >> Self::DIV_BY;
        let mask = 1u64 << (Self::MAP_WIDTH as u64 - ((idx + 1) as u64 & Self::MASK_64));
        self.bitmap[map_idx as usize] & mask == mask
    }
}

unsafe impl Send for FreeIdxManager {}
unsafe impl Sync for FreeIdxManager {}

#[cfg(test)]
mod tests {
    use super::*;

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
        unsafe { mgr.retire(indices[5]) }
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
        unsafe { mgr.retire(allocated[129]) }
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
        let mut mgr = FreeIdxManager::new(64);

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

        // retire one block safely
        unsafe { mgr.retire(all[10]) }

        // should reuse it again
        let reused = mgr.get_free_idx();
        assert_eq!(reused, all[10]);
    }

    #[test]
    fn release_updates_bitmap_and_curr_idx() {
        let mut mgr = FreeIdxManager::new(128);
        let all = alloc_all(&mut mgr, 128);

        let prev_curr = mgr.curr_idx;
        unsafe { mgr.retire(all[50]) }

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
        let mut mgr = FreeIdxManager::new(32);
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
        let mut mgr = FreeIdxManager::new(64);
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
}
