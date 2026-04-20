use std::{
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::slot_pool::static_pool::fixed_slot_manager::FixedPool;

pub struct Slot<'ax, T> {
    pub(crate) ptr: NonNull<MaybeUninit<T>>,
    pub(crate) pool_ref: &'ax FixedPool<T>,
    pub(crate) idx: u32,
}

impl<'ax, T> Slot<'ax, T> {
    pub fn into_inner(self) -> T {
        unsafe {
            let data = (*self.ptr.as_ptr()).assume_init_read();
            self.pool_ref.unchecked_retire(self.idx);
            mem::forget(self);
            data
        }
    }
}

impl<T> Deref for Slot<'_, T> {
    type Target = T;

    /// Returns a shared reference to the underlying Slot.
    ///
    /// # Safety
    /// Guaranteed safe because the Slot is initialized.
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref().assume_init_ref() }
    }
}

impl<T> DerefMut for Slot<'_, T> {
    /// Returns a mutable reference to the underlying Slot.
    ///
    /// # Safety
    /// Safe because the Slot is uniquely borrowed through `&mut self`.
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut().assume_init_mut() }
    }
}

impl<T> Drop for Slot<'_, T> {
    /// Drops the Slot and returns the slot to the pool.
    ///
    /// # Safety
    /// Safe because Rust ensures that the handle is dropped exactly once.
    fn drop(&mut self) {
        unsafe {
            self.ptr.as_mut().assume_init_drop();
            self.pool_ref.unchecked_retire(self.idx);
        }
    }
}
