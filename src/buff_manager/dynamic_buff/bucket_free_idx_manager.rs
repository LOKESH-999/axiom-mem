

pub struct BucketFreeIdxManager{
    slots:Vec<u128>,
    empty_slots:Vec<u16>,
    empty_slot_idx:u16,
}

impl BucketFreeIdxManager{
    pub const DEFAULT_BASE_SLOTS:u16 = 4;
    pub const SET_MASK:u128 = u128::MAX;
    pub const NULL_IDX:u32 = u32::MAX;
    pub const MAX_SLOTS:u16 = 65_533;
    pub const MASK_LEN:u32 = 128;
    pub const DEFAULT_GROW_CAP:u16 = 4;
    pub fn new()->Self{
        Self::new_with(Self::DEFAULT_BASE_SLOTS)
    }

    pub fn new_with(n_slots:u16)->Self{
        assert!(n_slots > 0, "N Slots Must be greater then 0" );
        assert!(n_slots  < Self::MAX_SLOTS,"Slots cant be graterthen ");
        let slots = vec![Self::SET_MASK;n_slots as usize];
        let mut empty_slots = (0..n_slots).collect::<Vec<_>>();
        // this acts as buffer for branchless ops
        empty_slots.push(0);
        Self{
            slots,
            empty_slot_idx:n_slots - 1,
            empty_slots
        }
    }

    pub fn get_free_idx(&mut self)->u32{
        todo!()
    }
}