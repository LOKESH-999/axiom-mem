use std::{mem::transmute, ptr::NonNull};

#[repr(transparent)]
pub struct Frame<T: ?Sized> {
    ptr: NonNull<T>,
}

pub struct Buffer<T: ?Sized> {
    buff_ptr: NonNull<Frame<T>>,
    frame_size: u32,
    buff_len: u32,
    n_empty: u32,
}

/// Start and end are inclusive Range
struct Bound {
    start: u32,
    end: u32,
}

struct LvlBitMasking {
    n_lvl: usize,
    lvl_arr: Frame<Bound>,
    lvl_bitmask_arr: &'static [u64],
}

impl LvlBitMasking {
    pub(super) fn new(size: u64) -> Self {
        todo!()
    }

    fn generate_bit_masking(size: u64) {
        let mut remining_size = size / 64;
        let rem = size >> 6 & 0b111111;
        let mut lvl_size = vec![];
        while remining_size > 0 {
            // remaining_size /= 64
            remining_size >>= 6;
            let rem = remining_size & 0b111111;
            // equvalent to if rem>0{ c = remaining_size + 1}else{remaining_size}
            // we can directly use `&` with is_rem bcoz we only add `1` whis is same as when we convert bool to u64
            let is_rem = (rem > 0) as u64;
            let c = remining_size + (1 & is_rem);
            lvl_size.push(c);
        }
    }
}
