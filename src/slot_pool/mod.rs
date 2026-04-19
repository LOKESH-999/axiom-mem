pub mod objects;
pub mod static_pool;
pub mod traits;

use std::ops::{Deref, DerefMut};

pub trait ObjectPool<T> {
    /// Handle type returned by the pool.
    ///
    /// Must:
    /// - Deref to `T`
    /// - DerefMut to `T`
    /// - manage lifecycle (drop → retire if applicable)
    type Handle<'a>: Deref<Target = T> + DerefMut
    where
        Self: 'a;

    /// Attempts to allocate and initialize an object in the pool.
    ///
    /// # Returns
    /// - `Ok(handle)` if allocation succeeds
    /// - `Err(value)` if allocation fails (e.g., full pool)
    ///
    /// # Semantics
    /// - Allocation is pool-specific (may be local, segmented, etc.)
    /// - No guarantee of global completeness
    fn alloc(&self, value: T) -> Result<Self::Handle<'_>, T>;
}

use std::ptr::NonNull;

pub trait Retire<T> {
    /// Retires an object by pointer.
    ///
    /// # Safety
    /// - `ptr` must belong to this pool
    /// - must not be double-retired
    unsafe fn retire_by_ptr(&self, ptr: NonNull<T>);

    /// Faster variant without checks.
    ///
    /// # Safety
    /// - same as above, but no validation
    unsafe fn retire_by_ptr_unchecked(&self, ptr: NonNull<T>);
}

pub trait IndexedPool {
    /// Returns index for pointer.
    ///
    /// # Safety
    /// - pointer must belong to pool
    unsafe fn get_idx_by_ptr_unchecked(&self, ptr: NonNull<u8>) -> u32;

    /// Checks if index is free.
    fn is_free_idx(&self, idx: u32) -> bool;

    /// Unsafe fast path
    unsafe fn is_free_idx_unchecked(&self, idx: u32) -> bool;
}

pub trait Capacity {
    /// Total capacity of pool
    fn capacity(&self) -> usize;

    /// Currently available slots (optional accuracy)
    fn available(&self) -> usize;
}

pub trait Grow {
    /// Attempts to grow the pool.
    ///
    /// # Returns
    /// - Ok if growth succeeds
    /// - Err if not supported or failed
    fn grow(&self, new_capacity: usize) -> Result<(), ()>;
}

pub trait RawAccess<T>: ObjectPool<T> {
    type PoolRef<'a>
    where
        Self: 'a;

    /// Convert handle → raw pointer
    ///
    /// # Safety
    /// Caller must ensure proper lifetime + retirement
    unsafe fn into_raw(handle: Self::Handle<'_>) -> NonNull<T>;

    /// Reconstruct handle from raw pointer
    ///
    /// # Safety
    /// - pointer must be valid
    /// - must belong to this pool
    unsafe fn from_raw(ptr: NonNull<T>, pool: Self::PoolRef<'_>) -> Self::Handle<'_>;
}

pub trait Reset {
    /// Clears entire pool at once
    ///
    /// Used for arena-style allocators
    fn reset(&self);
}
