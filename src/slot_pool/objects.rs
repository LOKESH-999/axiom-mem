use std::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub trait StaticObject<T>: Deref<Target = T> + DerefMut {
    /// Returns the internal slot index.
    ///
    /// NOTE:
    /// This is mostly for debugging / introspection.
    /// Do NOT use for lifecycle management.
    fn id(&self) -> u32;

    /// Extract inner value and retire slot.
    ///
    /// Consumes self → prevents double free.
    fn into_inner(self) -> T;

    /// Convert into raw pointer (escape hatch)
    ///
    /// # Safety
    /// - Caller must manually retire
    unsafe fn into_raw_ptr(self) -> NonNull<T>;

    /// Reconstruct from raw pointer
    ///
    /// # Safety
    /// - ptr must belong to pool
    /// - must not already be retired
    unsafe fn from_raw_parts(ptr: NonNull<T>, pool: &Self::Pool) -> Self
    where
        Self: Sized;

    /// Associated pool type
    type Pool;
}

// pub struct Object<T