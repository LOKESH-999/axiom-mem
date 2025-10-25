use std::{
    alloc::{Layout, alloc, dealloc},
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::buff_manager::static_free_idx_map::FreeIdxManager;

/// Fixed-capacity object pool for type `T`.
///
/// # Overview
/// `ObjectPoolManager` pre-allocates a contiguous memory region that can hold up to `size`
/// instances of `T`. Objects are allocated and deallocated without invoking the global allocator
/// for each object. Allocation returns an [`Object`] handle which ensures proper destruction
/// and slot reclamation.
///
/// # Safety
/// - Performs manual memory management using `UnsafeCell`, `NonNull`, and `MaybeUninit<T>`.
/// - Rust lifetimes enforce that no [`Object`] can outlive its pool.
/// - **However**, FFI or raw pointer misuse can easily cause *undefined behavior*.
/// - The pool is **not Send or Sync**; concurrent access must be externally synchronized.
///
/// # Features
/// - Fixed-size allocation; no resizing or reallocation.
/// - Constant-time allocation and deallocation.
/// - Automatic cleanup of live objects on drop.
pub struct ObjectPoolManager<T> {
    /// Internal free index manager for tracking unused slots.
    free_idx_map: UnsafeCell<FreeIdxManager>,

    /// Base pointer to pre-allocated memory containing `MaybeUninit<T>`.
    base_ptr: NonNull<MaybeUninit<T>>,

    /// Maximum number of elements the pool can hold.
    size: u32,

    // Internal marker for compiler bookkeeping; This is to indicate !Send & !Sync
    _marker: PhantomData<Cell<()>>,
}
/// A handle representing a single initialized object in an `ObjectPoolManager`.
///
/// This handle provides ergonomic access to the object via dereferencing,
/// and automatically returns the slot to the pool when dropped.
///
/// # Features
/// - Dereferences to `&T` or `&mut T`.
/// - On drop, calls the destructor of `T` and releases the slot back to the pool.
/// - Can be converted to/from a raw tuple `(NonNull<T>, id, &pool)` for advanced use.
///
/// # Safety
/// The internal pointer must always refer to a valid slot initialized by the pool.
/// Violating this invariant is **undefined behavior**.
pub struct Object<'o, T> {
    /// Pointer to the object inside the pool.
    ptr: NonNull<MaybeUninit<T>>,

    /// Reference to the pool that owns this object.
    pool_ref: &'o ObjectPoolManager<T>,

    /// Slot index of this object in the pool.
    id: u32,

    /// Marker to indicate `!Send` and `!Sync`.
    _marker: PhantomData<Cell<()>>,
}

impl<'o, T> Object<'o, T> {
    /// Consumes the handle and extracts the inner value.
    ///
    /// Moves out the value from the pool slot and releases it back to the free list.
    /// After this call, the handle is **moved** and cannot be used again.
    ///
    /// # Safety
    /// - The internal pointer must point to a valid, initialized object in the pool.
    /// - Safe in normal usage because the pool ensures the object is valid.
    pub const fn into_inner(self) -> T {
        unsafe {
            let data = (*self.ptr.as_ptr()).assume_init_read();
            self.pool_ref.retire(self.id);
            mem::forget(self);
            data
        }
    }

    /// Returns the slot index of this object in the pool.
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// Converts the `Object` into a raw tuple `(NonNull<T>, id, &pool)`.
    ///
    /// # Safety
    /// - Bypasses the usual safety guarantees of the pool.
    /// - `ptr` must be valid and initialized.
    /// - Use only for low-level manipulation; improper use may cause undefined behavior.
    pub const unsafe fn into_tuple(self) -> (NonNull<T>, u32, &'o ObjectPoolManager<T>) {
        let tup = (self.ptr.cast(), self.id, self.pool_ref);
        mem::forget(self);
        tup
    }

    /// Reconstructs an `Object` from a raw tuple `(NonNull<T>, id, &pool)`.
    ///
    /// # Safety
    /// - `ptr` must point to a valid, initialized object in the pool.
    /// - `id` must correspond to the slot index of the object.
    /// - Violating these conditions is undefined behavior.
    pub const unsafe fn from_tuple(
        ptr: NonNull<T>,
        id: u32,
        pool_ref: &'o ObjectPoolManager<T>,
    ) -> Self {
        Object {
            ptr: ptr.cast(),
            pool_ref,
            id,
            _marker: PhantomData,
        }
    }
}

impl<T> Deref for Object<'_, T> {
    type Target = T;

    /// Returns a shared reference to the underlying object.
    ///
    /// # Safety
    /// Guaranteed safe because the object is initialized.
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref().assume_init_ref() }
    }
}

impl<T> DerefMut for Object<'_, T> {
    /// Returns a mutable reference to the underlying object.
    ///
    /// # Safety
    /// Safe because the object is uniquely borrowed through `&mut self`.
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut().assume_init_mut() }
    }
}

impl<T> Drop for Object<'_, T> {
    /// Drops the object and returns the slot to the pool.
    ///
    /// # Safety
    /// Safe because Rust ensures that the handle is dropped exactly once.
    fn drop(&mut self) {
        unsafe {
            self.ptr.as_mut().assume_init_drop();
            self.pool_ref.retire(self.id);
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Object<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Object")
            .field("id", &self.id)
            .field("data", &**self)
            .finish()
    }
}

impl<T> ObjectPoolManager<T> {
    /// Creates a new object pool with capacity `size`.
    ///
    /// # Panics
    /// Panics if memory allocation fails.
    pub fn new(size: u32) -> Self {
        let free_idx_map = UnsafeCell::new(FreeIdxManager::new(size));
        let base_ptr =
            NonNull::new(unsafe { alloc(Self::layout(size)) as _ }).expect("Allocation failed");
        Self {
            free_idx_map,
            base_ptr,
            size,
            _marker: PhantomData,
        }
    }

    /// Returns the memory layout for `size` elements of `MaybeUninit<T>`.
    const fn layout(size: u32) -> Layout {
        match Layout::array::<MaybeUninit<T>>(size as usize) {
            Ok(l) => l,
            Err(_) => panic!("Invalid layout for ObjectPoolManager"),
        }
    }

    /// Allocates and initializes a new object in the pool.
    ///
    /// Returns `Ok(Object)` if successful, or `Err(data)` if the pool is full.
    ///
    /// # Safety
    /// - Writes directly to raw memory.
    /// - Modifies the free index map via `UnsafeCell`.
    pub fn pop_free(&self, data: T) -> Result<Object<'_, T>, T> {
        unsafe {
            let idx = (&mut *self.free_idx_map.get()).get_free_idx();
            if idx != FreeIdxManager::NULL_IDX {
                let ptr = self.base_ptr.add(idx as usize);
                (*ptr.as_ptr()).write(data);
                return Ok(Object {
                    pool_ref: self,
                    ptr,
                    id: idx,
                    _marker: PhantomData,
                });
            }
            Err(data)
        }
    }

    /// Releases a slot back to the pool.
    ///
    /// # Safety
    /// - Only call with a valid slot index previously returned by `pop_free`.
    /// - Double retire or invalid ID is *undefined behavior*.
    const unsafe fn retire(&self, id: u32) {
        unsafe { (&mut *self.free_idx_map.get()).retire(id) }
    }
}

impl<T> Drop for ObjectPoolManager<T> {
    /// Deallocates the memory backing the pool.
    ///
    /// # Safety
    /// Must not have any live [`Object`] handles; Rust lifetimes ensure this for safe Rust usage.
    fn drop(&mut self) {
        unsafe { dealloc(self.base_ptr.as_ptr() as _, Self::layout(self.size)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_allocation_and_deref() {
        let pool = ObjectPoolManager::new(2);
        let obj = pool.pop_free(42).unwrap();
        assert_eq!(*obj, 42);

        let mut obj_mut = pool.pop_free(10).unwrap();
        *obj_mut += 5;
        assert_eq!(*obj_mut, 15);
    }

    #[test]
    fn into_inner_consumes_object() {
        let pool = ObjectPoolManager::new(1);
        let obj = pool.pop_free(99).unwrap();
        let val = obj.into_inner();
        assert_eq!(val, 99);
        // Pool slot should now be free
        let obj2 = pool.pop_free(100).unwrap();
        assert_eq!(obj2.id, 0);
    }

    #[test]
    fn multiple_allocations_and_releases() {
        let pool = ObjectPoolManager::new(3);
        let o1 = pool.pop_free(1).unwrap();
        let o2 = pool.pop_free(2).unwrap();
        let o3 = pool.pop_free(3).unwrap();
        assert!(pool.pop_free(4).is_err()); // pool full

        drop(o2); // retire middle slot
        let o4 = pool.pop_free(40).unwrap();
        assert_eq!(o4.id, 1); // should reuse released slot

        drop(o1);
        drop(o3);
        drop(o4);

        // All slots should be free
        let mut ids = Vec::new();
        let mut objs = Vec::new();
        for i in 0..3 {
            let obj = pool.pop_free(i * 10).unwrap();
            ids.push(obj.id);
            objs.push(obj);
        }
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn pool_exhaustion() {
        let pool = ObjectPoolManager::new(2);
        let _a = pool.pop_free(10).unwrap();
        let _b = pool.pop_free(20).unwrap();
        assert!(pool.pop_free(30).is_err(), "Pool should be exhausted");
    }

    #[test]
    fn drop_release_cycle() {
        let pool = ObjectPoolManager::new(2);
        {
            let o1 = pool.pop_free(1).unwrap();
            let _o2 = pool.pop_free(2).unwrap();
            assert!(pool.pop_free(3).is_err());
            drop(o1);
            let o3 = pool.pop_free(3).unwrap();
            assert_eq!(o3.id, 1);
        }
        // Both slots should now be free
        let mut ids = Vec::new();
        let mut objs = Vec::new();
        for i in 0..2 {
            let obj = pool.pop_free(i).unwrap();
            ids.push(obj.id);
            objs.push(obj);
        }
        ids.sort();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn stress_allocation_and_reuse() {
        let pool = ObjectPoolManager::new(5);
        let mut handles = Vec::new();

        // Allocate all slots
        for i in 0..5 {
            let obj = pool.pop_free(i).unwrap();
            handles.push(obj);
        }
        assert!(pool.pop_free(99).is_err());

        // Drop all
        handles.clear();

        // Reallocate and check ids reused
        let mut new_ids = Vec::new();
        let mut objs = Vec::new();
        for i in 0..5 {
            let obj = pool.pop_free(i * 10).unwrap();
            new_ids.push(obj.id);
            objs.push(obj);
        }
        new_ids.sort();
        assert_eq!(new_ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn deref_mut_and_modify() {
        let pool = ObjectPoolManager::new(2);
        let mut obj = pool.pop_free(5).unwrap();
        *obj += 10;
        assert_eq!(*obj, 15);

        let obj2 = pool.pop_free(20).unwrap();
        assert_eq!(*obj2, 20);
    }

    #[test]
    fn into_inner_and_slot_reuse() {
        let pool = ObjectPoolManager::new(2);
        let obj = pool.pop_free(123).unwrap();
        let obj2 = pool.pop_free(456).unwrap();
        let val = obj.into_inner();
        assert_eq!(val, 123);
        assert_eq!(obj2.id, 0);
    }

    #[test]
    fn drop_multiple_cycles() {
        let pool = ObjectPoolManager::new(2);
        {
            let a = pool.pop_free(1).unwrap();
            let b = pool.pop_free(2).unwrap();
            assert!(pool.pop_free(3).is_err());
            drop(a);
            let c = pool.pop_free(3).unwrap();
            assert_eq!(c.id, 1);
            drop(b);
            drop(c);
        }
        // Both slots free again
        let mut ids = Vec::new();
        let mut objs = Vec::new();
        for i in 0..2 {
            let obj = pool.pop_free(i * 10).unwrap();
            ids.push(obj.id);
            objs.push(obj);
        }
        assert_eq!(ids, vec![1, 0]);
    }

    #[test]
    fn full_pool_allocation_and_exhaustion() {
        let pool = ObjectPoolManager::new(1);
        let obj = pool.pop_free(7).unwrap();
        assert!(pool.pop_free(8).is_err());
        drop(obj);
        assert!(pool.pop_free(9).is_ok());
    }
    #[test]
    fn exhaustion_and_reuse() {
        let pool = ObjectPoolManager::new(3);

        let o1 = pool.pop_free(1).unwrap();
        let o2 = pool.pop_free(2).unwrap();
        let _o3 = pool.pop_free(3).unwrap();
        assert!(pool.pop_free(4).is_err());

        drop(o2);
        drop(o1);

        let r1 = pool.pop_free(10).unwrap();
        let r2 = pool.pop_free(20).unwrap();

        let mut ids = vec![r1.id, r2.id];
        ids.sort();
        assert_eq!(ids, vec![1, 2], "Freed slots should be reused first");
    }

    /// Stress test: repeatedly allocate and drop to catch memory reuse bugs.
    #[test]
    fn repeated_allocate_drop_cycles() {
        let pool = ObjectPoolManager::new(2);

        for cycle in 0..10 {
            let a = pool.pop_free(cycle).unwrap();
            let b = pool.pop_free(cycle + 100).unwrap();
            assert!(pool.pop_free(999).is_err());
            drop(a);
            drop(b);
        }

        let a = pool.pop_free(500).unwrap();
        let b = pool.pop_free(501).unwrap();
        assert!(pool.pop_free(502).is_err());
        drop(a);
        drop(b);
    }

    /// Test `into_inner` releases slot correctly for reuse.
    #[test]
    fn into_inner_reuse() {
        let pool = ObjectPoolManager::new(2);
        let obj1 = pool.pop_free(42).unwrap();
        let _val = obj1.into_inner(); // consumes obj1, slot should free

        let obj2 = pool.pop_free(100).unwrap();
        assert_eq!(obj2.id, 1, "Slot freed by into_inner should be reused");
    }

    /// Mutation isolation: modifying one object should not affect another.
    #[test]
    fn deref_mut_isolation() {
        let pool = ObjectPoolManager::new(2);
        let mut a = pool.pop_free(1).unwrap();
        let mut b = pool.pop_free(2).unwrap();

        *a += 100;
        *b += 200;

        assert_eq!(*a, 101);
        assert_eq!(*b, 202);
    }

    /// Stress memory safety by filling pool, dropping, reallocating, and checking values.
    #[test]
    fn stress_memory_safety() {
        let pool = ObjectPoolManager::new(5);
        let mut handles = vec![];

        // Fill pool
        for i in 0..5 {
            handles.push(pool.pop_free(i * 10).unwrap());
        }

        assert!(pool.pop_free(999).is_err());

        handles.reverse();
        // Drop some in non-sequential order
        drop(handles.swap_remove(2));
        drop(handles.swap_remove(0));

        // Reallocate and check ids reused
        let a = pool.pop_free(111).unwrap();
        let b = pool.pop_free(222).unwrap();
        let mut ids = vec![a.id, b.id];
        ids.sort();
        assert_eq!(
            ids,
            vec![0, 2],
            "Freed slots must be reused regardless of drop order"
        );
    }

    /// Ensure no double retire occurs and pool handles properly after multiple drops.
    #[test]
    fn double_drop_simulation() {
        let pool = ObjectPoolManager::new(2);

        let a = pool.pop_free(1).unwrap();
        let b = pool.pop_free(2).unwrap();
        drop(a);
        drop(b);

        let c = pool.pop_free(3).unwrap();
        let d = pool.pop_free(4).unwrap();

        assert_eq!(vec![c.id, d.id], vec![1, 0]);
    }

    /// Edge case: allocate, immediately drop, and allocate again multiple times.
    #[test]
    fn rapid_allocate_drop() {
        let pool = ObjectPoolManager::new(1);

        for i in 0..10 {
            let obj = pool.pop_free(i).unwrap();
            assert_eq!(obj.id, 0, "Single slot pool should always reuse same index");
            drop(obj);
        }
    }

    /// Ensure object destruction calls destructor correctly.
    #[test]
    fn drop_destructor_called() {
        use std::cell::Cell;

        #[derive(Debug)]
        struct Tracker<'a> {
            dropped: &'a Cell<u32>,
        }

        impl Drop for Tracker<'_> {
            fn drop(&mut self) {
                self.dropped.set(self.dropped.get() + 1);
            }
        }

        let dropped = Cell::new(0);
        let pool = ObjectPoolManager::new(2);

        {
            let t1 = pool.pop_free(Tracker { dropped: &dropped }).unwrap();
            let t2 = pool.pop_free(Tracker { dropped: &dropped }).unwrap();
            assert_eq!(dropped.get(), 0);
            drop(t1);
            assert_eq!(dropped.get(), 1);
            drop(t2);
            assert_eq!(dropped.get(), 2);
        }

        // Pool destroyed, no double-drop
    }

    #[test]
    fn test_into_tuple_and_from_tuple() {
        let pool = ObjectPoolManager::new(1);
        let obj = pool.pop_free(55).unwrap();

        // Convert to raw tuple
        let (ptr, idx, pool_ref) = unsafe { obj.into_tuple() };
        assert_eq!(idx, 0);

        // Reconstruct object from tuple
        let reconstructed = unsafe { Object::from_tuple(ptr, idx, pool_ref) };
        assert_eq!(*reconstructed, 55);

        // Dropping reconstructed object should free slot
        drop(reconstructed);
        let obj2 = pool.pop_free(99).unwrap();
        assert_eq!(obj2.id(), 0);
    }
}
