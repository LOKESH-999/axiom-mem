use std::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub trait SlotPool<T>:Deallocate {
    /// Handle returned by the pool.
    ///
    /// Must:
    /// - Deref/DerefMut to T
    /// - On Drop → correctly release slot
    type Handle<'a>: Deref<Target = T> + DerefMut
    where
        Self: 'a;

    // ========================
    // Core Allocation & Drop
    // ========================
    fn alloc(&self, value: T) -> Result<Self::Handle<'_>, T>;

    /// # Safety
    /// - ptr must belong to this pool
    /// - must not double-retire
    unsafe fn retire_by_ptr(&self, ptr: NonNull<T>);

    /// # Safety
    /// - idx must be valid
    /// - no double retire
    unsafe fn unchecked_retire_by_idx(&self, idx: u32);

    // ========================
    // Introspection (required)
    // ========================

    fn capacity(&self) -> usize;

    fn available(&self) -> usize;
}
pub trait Grow {
    fn grow(&self, new_capacity: usize) -> Result<(), ()>;
}

pub trait Deallocate {
    fn deallocate(&mut self);
}