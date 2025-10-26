pub struct DynFreeIdxManager {
    bitmap: Vec<u64>,
    free_list: Vec<u16>,
    capacity: u32,
    curr_idx: u16,
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
        let curr_idx = (free_list.len() - 1) as u16;
        Self {
            bitmap,
            free_list,
            capacity: n_block,
            curr_idx,
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

    pub fn grow(&mut self, delta: u16) {
        unimplemented!("Grow funcanalities not yet implemented delta:{}", delta)
    }
}
