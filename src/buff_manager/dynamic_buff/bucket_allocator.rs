use std::ptr::NonNull;



struct SlotMap<T>{
    mask:u64,
    slot_base_ptr:NonNull<T>
}
impl<T> SlotMap<T>{
    const SET_MASK:u64 = u64::MAX;
    const MASK_LEN:u32 = 64;
    const fn new(slot_base_ptr:NonNull<T>)->Self{
        Self{
            mask:Self::SET_MASK,
            slot_base_ptr
        }
    }

    const unsafe fn is_free_unchecked(&self,id:u32) -> bool{
        let m = 1u64 << id;
        (self.mask & m) == m
    }

    const fn is_free(&self,id:u32)->bool{
        assert!(id < Self::MASK_LEN,"ID Should be less then `Self::MASK_LEN`");
        unsafe {self.is_free_unchecked(id)}
    }

    const unsafe fn get_free_idx_unchecked(&mut self)->u32{
        let idx =self.mask.trailing_zeros();
        let mask = 1u64.wrapping_shl(idx);
        self.mask &= !mask;
        idx
    }

    const fn get_free_idx(&mut self)-> Option<u32>{
        let idx =unsafe {self.get_free_idx_unchecked()};
        if idx < Self::MASK_LEN{
            Some(idx)
        }else {
            None
        }
    }

    const unsafe fn release_unchecked(&mut self,id:u32){
        let m = 1u64 << id;
        self.mask |= m;
    }

    unsafe fn idx_from_ptr_unchecked(&self,ptr:NonNull<T>)->u32{
        unsafe { ptr.offset_from(self.slot_base_ptr) as u32}
    }

    unsafe fn release_by_ptr(&mut self,ptr:NonNull<T>){
        let idx = unsafe { self.idx_from_ptr_unchecked(ptr) };
        self.mask |= 1u64.wrapping_shl(idx);
    }
}

pub struct BucketAllocatorManager<T>{
    slots:Vec<SlotMap<T>>,

}