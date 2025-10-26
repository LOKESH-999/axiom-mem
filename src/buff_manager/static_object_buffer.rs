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

    /// Consumes the [`Object`] and returns a raw pointer to the value.
    ///
    /// # Safety
    /// - The caller must guarantee that this pointer is **used and dropped**
    ///   within the lifetime of the [`ObjectPoolManager`] that owns it.
    /// - The pool will not automatically reclaim or drop the value when this pointer
    ///   is no longer used; the user must call respective [`ObjectPoolManager::retire_by_ptr()`]
    ///   manually when done.
    ///
    /// # Example
    /// ```
    /// use axiom_mem::buff_manager::ObjectPoolManager;
    /// let pool = ObjectPoolManager::new(8);
    /// let obj = pool.pop_free(42).unwrap();
    /// let raw = unsafe { obj.into_raw_ptr() };
    ///
    /// // use raw pointer ...
    /// unsafe { pool.retire_by_ptr(raw); }
    /// ```
    pub const unsafe fn into_raw_ptr(self) -> NonNull<T> {
        let ptr = self.ptr.cast();
        mem::forget(self);
        ptr
    }

    /// Reconstructs an `Object` from a raw pointer and its owning pool.
    ///
    /// # Safety
    /// - `ptr` must point to a valid object currently allocated in `pool_ref`.
    /// - The returned `Object` assumes ownership of this slot and will release it
    ///   when dropped.
    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: NonNull<T>, pool_ref: &'o ObjectPoolManager<T>) -> Self {
        let id = unsafe { pool_ref.get_idx_by_ptr_unchecked(ptr) };
        Self {
            id,
            pool_ref,
            ptr: ptr.cast(),
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

    /// Computes the index of a pointer within the pool without bounds checking.
    ///
    /// # Safety
    /// - The pointer must have been obtained via [`Object::into_raw_ptr`] which releated to this pool.
    /// - The pointer `ptr` must point to an initialized object inside this pool.
    /// - It must have been originally allocated via this pool.
    /// - The caller must ensure `ptr >= base_ptr` and within bounds.
    ///
    /// # Debug Mode
    /// - In debug builds, asserts that the pointer lies after `base_ptr`.
    ///
    /// # Example
    /// ```
    /// use axiom_mem::buff_manager::ObjectPoolManager;
    /// let pool = ObjectPoolManager::new(4);
    /// let obj = pool.pop_free(10).unwrap();
    /// let raw = unsafe { obj.into_raw_ptr() };
    /// let idx = unsafe { pool.get_idx_by_ptr_unchecked(raw) };
    /// assert_eq!(idx, 3);
    /// unsafe { pool.retire_by_ptr(raw); }
    /// ```
    #[inline(always)]
    pub unsafe fn get_idx_by_ptr_unchecked(&self, ptr: NonNull<T>) -> u32 {
        // Checking `ptr` > `self.base_ptr`
        debug_assert!(ptr.as_ptr().cast() >= self.base_ptr.as_ptr());
        // Getting the Index from ptr
        unsafe { ptr.cast().offset_from(self.base_ptr) as u32 }
    }

    /// Drops the object pointed to by `ptr` and releases its slot back to the pool,
    /// without performing bounds checks.
    ///
    /// # Safety
    /// - The pointer must have been obtained via [`Object::into_raw_ptr`] which releated to this pool.
    /// - The pointer must belong to this pool and point to a valid initialized object.
    /// - The caller must ensure no double-retire or aliasing access occurs.
    /// - Using an out-of-pool pointer results in undefined behavior.
    ///
    /// # Debug Mode
    /// - Performs range assertions to ensure pointer validity.
    ///
    /// # Example
    /// ```
    /// use axiom_mem::buff_manager::ObjectPoolManager;
    /// let pool = ObjectPoolManager::new(2);
    /// let obj = pool.pop_free(42).unwrap();
    /// let raw = unsafe { obj.into_raw_ptr() };
    /// unsafe { pool.retire_by_ptr_unchecked(raw); }
    /// ```
    #[cfg(debug_assertions)]
    #[inline(always)]
    pub unsafe fn retire_by_ptr_unchecked(&self, ptr: NonNull<T>) {
        unsafe {
            // Just checking weather the pointer in range or not
            debug_assert!(ptr.as_ptr().cast() <= self.base_ptr.as_ptr().add(self.size as usize));
            // Droping the value
            (*ptr.cast::<MaybeUninit<T>>().as_ptr()).assume_init_drop();
            // getting the drop idx
            let idx = self.get_idx_by_ptr_unchecked(ptr);
            // Marking the drop idx as free
            self.retire(idx);
        }
    }

    /// Drops and retires an object by pointer, with safety assertions.
    ///
    /// This version performs runtime bound checks to ensure `ptr` lies within the
    /// allocated pool memory range.
    ///
    /// # Panics
    /// Panics if `ptr` is outside the valid memory range of the pool.
    ///
    /// # Safety
    /// The pointer must have been obtained via [`Object::into_raw_ptr`] which releated to this pool.
    ///
    /// # Example
    /// ```
    /// use axiom_mem::buff_manager::ObjectPoolManager;
    /// let pool = ObjectPoolManager::new(4);
    /// let obj = pool.pop_free(99).unwrap();
    /// let raw = unsafe { obj.into_raw_ptr() };
    /// unsafe { pool.retire_by_ptr(raw) };
    /// ```
    #[inline(always)]
    pub unsafe fn retire_by_ptr(&self, ptr: NonNull<T>) {
        assert!(self.base_ptr <= ptr.cast(), "`ptr` must be >= `base_ptr`");
        unsafe {
            assert!(
                self.base_ptr.add(self.size as usize) >= ptr.cast(),
                "`ptr` <= `peek_ptr"
            );
            self.retire_by_ptr_unchecked(ptr);
        }
    }

    /// Checks if the slot with the given index is currently free in the pool.
    ///
    /// # Parameters
    /// - `id`: The index of the slot to check.
    ///
    /// # Returns
    /// - `true` if the slot is free and can be allocated.
    /// - `false` if the slot is currently in use.
    ///
    /// # Safety
    /// This function uses an `UnsafeCell` internally to access the free index map,
    /// but it provides a safe interface for checking slot availability.
    pub const fn is_free_idx(&self, id: u32) -> bool {
        assert!(id <= self.size, "`Id` must be <= `self.size`");
        unsafe { (*self.free_idx_map.get()).is_free(id) }
    }

    /// Checks if a given slot index is free in the pool without bounds checks.
    ///
    /// # Safety
    /// - Caller must ensure that `id` is within the valid range of indices.
    /// - Passing an out-of-bounds `id` can lead to undefined behavior.
    ///
    /// # Parameters
    /// - `id`: The slot index to check.
    ///
    /// # Returns
    /// - `true` if the slot is free.
    /// - `false` if the slot is occupied.
    ///
    /// # Notes
    /// - This delegates to the underlying `FreeIdxMap::is_free` method.
    /// - No bounds or validity checks are performed.
    pub const unsafe fn is_free_idx_unchecked(&self, id: u32) -> bool {
        unsafe { (*self.free_idx_map.get()).is_free(id) }
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

    #[test]
    fn test_into_raw_ptr_and_retire() {
        let pool = ObjectPoolManager::new(2);
        let obj = pool.pop_free("fucke_you".to_string()).unwrap();
        let raw = unsafe { obj.into_raw_ptr() };
        unsafe { pool.retire_by_ptr(raw) }; // must reclaim slot safely
        // Reuse slot
        let obj2 = pool.pop_free("fucke_you2".to_string()).unwrap();
        assert_eq!(*obj2, "fucke_you2".to_string());
    }

    #[test]
    fn test_get_idx_by_ptr_unchecked() {
        let pool = ObjectPoolManager::new(3);
        let obj = pool.pop_free("fucke_you".to_string()).unwrap();
        let raw = unsafe { obj.into_raw_ptr() };
        let idx = unsafe { pool.get_idx_by_ptr_unchecked(raw) };
        assert_eq!(idx, 2);
        unsafe { pool.retire_by_ptr(raw) };
    }

    #[test]
    fn test_retire_by_ptr_unchecked() {
        let pool = ObjectPoolManager::new(2);
        let obj = pool.pop_free("fucke_you".to_string()).unwrap();
        let raw = unsafe { obj.into_raw_ptr() };
        unsafe { pool.retire_by_ptr_unchecked(raw) }; // drop and free
        // Check that slot is reusable
        let obj2 = pool.pop_free("fucke_you2".to_string()).unwrap();
        assert_eq!(*obj2, "fucke_you2".to_string());
    }

    #[test]
    fn test_retire_by_ptr_with_assertions() {
        let pool = ObjectPoolManager::new(2);
        let obj = pool.pop_free("fucke_you".to_string()).unwrap();
        let raw = unsafe { obj.into_raw_ptr() };
        unsafe { pool.retire_by_ptr(raw) }; // safe, asserts pointer in bounds
        let obj2 = pool.pop_free("fucke_you2".to_string()).unwrap();
        assert_eq!(*obj2, "fucke_you2".to_string());
    }

    #[test]
    #[should_panic]
    fn test_retire_by_ptr_out_of_bounds() {
        let pool = ObjectPoolManager::<String>::new(2);
        let bogus_ptr = NonNull::dangling(); // invalid pointer
        unsafe { pool.retire_by_ptr(bogus_ptr) }; // must panic
    }

    #[test]
    fn test_stack_like_pop_free_and_idx_verification() {
        let pool = ObjectPoolManager::new(4);

        // Allocate Boxes in stack-like manner: idx should go 3,2,1,0
        let a = pool.pop_free(Box::new(10)).unwrap();
        let b = pool.pop_free(Box::new(20)).unwrap();
        let c = pool.pop_free(Box::new(30)).unwrap();
        let d = pool.pop_free(Box::new(40)).unwrap();

        assert!(pool.pop_free(Box::new(50)).is_err(), "Pool should be full");

        // Verify get_idx_by_ptr_unchecked correctness
        let ptr_a = unsafe { a.into_raw_ptr() };
        let ptr_b = unsafe { b.into_raw_ptr() };
        let ptr_c = unsafe { c.into_raw_ptr() };
        let ptr_d = unsafe { d.into_raw_ptr() };

        let idx_a = unsafe { pool.get_idx_by_ptr_unchecked(ptr_a) };
        let idx_b = unsafe { pool.get_idx_by_ptr_unchecked(ptr_b) };
        let idx_c = unsafe { pool.get_idx_by_ptr_unchecked(ptr_c) };
        let idx_d = unsafe { pool.get_idx_by_ptr_unchecked(ptr_d) };

        // Since stack-like allocation, highest index allocated first
        assert_eq!(idx_a, 3);
        assert_eq!(idx_b, 2);
        assert_eq!(idx_c, 1);
        assert_eq!(idx_d, 0);

        // Check is_free_idx
        assert!(!pool.is_free_idx(idx_a));
        assert!(!pool.is_free_idx(idx_b));
        assert!(!pool.is_free_idx(idx_c));
        assert!(!pool.is_free_idx(idx_d));
    }

    #[test]
    fn test_retire_by_ptr_unchecked_and_bounds() {
        let pool = ObjectPoolManager::new(4);

        let s1 = pool.pop_free(String::from("hello")).unwrap();
        let s2 = pool.pop_free(String::from("world")).unwrap();

        let p1 = unsafe { s1.into_raw_ptr() };
        let p2 = unsafe { s2.into_raw_ptr() };

        // Retire both using unchecked
        unsafe { pool.retire_by_ptr_unchecked(p1) };
        unsafe { pool.retire_by_ptr_unchecked(p2) };

        // Ensure slots marked free
        let idx1 = unsafe { pool.get_idx_by_ptr_unchecked(p1) };
        let idx2 = unsafe { pool.get_idx_by_ptr_unchecked(p2) };
        assert!(pool.is_free_idx(idx1));
        assert!(pool.is_free_idx(idx2));
    }

    #[test]
    fn test_retire_by_ptr_safe_assertions_and_edge() {
        let pool = ObjectPoolManager::new(2);

        let v1 = pool.pop_free(vec![1, 2, 3]).unwrap();
        let v2 = pool.pop_free(vec![4, 5, 6]).unwrap();

        let ptr1 = unsafe { v1.into_raw_ptr() };
        let ptr2 = unsafe { v2.into_raw_ptr() };

        // In-range pointers, should work
        unsafe { pool.retire_by_ptr(ptr1) };
        unsafe { pool.retire_by_ptr(ptr2) };

        assert!(pool.is_free_idx(0));
        assert!(pool.is_free_idx(1));
    }

    #[test]
    #[should_panic]
    fn test_retire_by_ptr_out_of_upper_bound_panics() {
        let pool = ObjectPoolManager::new(2);
        let bogus_ptr =
            unsafe { NonNull::new_unchecked(pool.base_ptr.as_ptr().add(5).cast::<String>()) };
        unsafe { pool.retire_by_ptr(bogus_ptr) };
    }

    #[test]
    #[should_panic]
    fn test_retire_by_ptr_below_lower_bound_panics() {
        let pool = ObjectPoolManager::new(2);
        let bogus_ptr =
            unsafe { NonNull::new_unchecked(pool.base_ptr.as_ptr().sub(1).cast::<String>()) };
        unsafe { pool.retire_by_ptr(bogus_ptr) };
    }

    #[test]
    fn test_multiple_alloc_retire_and_idx_consistency() {
        let pool = ObjectPoolManager::new(3);

        let o1 = pool.pop_free(Box::new(100)).unwrap();
        let o2 = pool.pop_free(Box::new(200)).unwrap();
        let o3 = pool.pop_free(Box::new(300)).unwrap();

        let p1 = unsafe { o1.into_raw_ptr() };
        let p2 = unsafe { o2.into_raw_ptr() };
        let p3 = unsafe { o3.into_raw_ptr() };

        // Retire middle first
        unsafe { pool.retire_by_ptr(p2) };

        let new_obj = pool.pop_free(Box::new(999)).unwrap();
        let new_ptr = unsafe { new_obj.into_raw_ptr() };
        let new_idx = unsafe { pool.get_idx_by_ptr_unchecked(new_ptr) };

        // Should reuse index 1
        assert_eq!(new_idx, 1);
        assert!(!pool.is_free_idx(new_idx));

        // Retire rest
        unsafe { pool.retire_by_ptr(p1) };
        unsafe { pool.retire_by_ptr(p3) };
        unsafe { pool.retire_by_ptr(new_ptr) };

        // All should now be free
        assert!(pool.is_free_idx(0));
        assert!(pool.is_free_idx(1));
        assert!(pool.is_free_idx(2));
    }

    #[test]
    fn test_string_vec_box_with_retire_by_ptr_unchecked() {
        let pool = ObjectPoolManager::new(3);

        let s = pool.pop_free(String::from("rust")).unwrap();
        let b = pool
            .pop_free("Box::new(vec![1, 2, 3])".to_string())
            .unwrap();
        let v = pool.pop_free("vec![10, 20, 30]".to_string()).unwrap();

        let p_s = unsafe { s.into_raw_ptr() };
        let p_b = unsafe { b.into_raw_ptr() };
        let p_v = unsafe { v.into_raw_ptr() };

        unsafe {
            pool.retire_by_ptr_unchecked(p_s);
            pool.retire_by_ptr_unchecked(p_b);
            pool.retire_by_ptr_unchecked(p_v);
        }

        // Ensure they are free
        assert!(pool.is_free_idx(0));
        assert!(pool.is_free_idx(1));
        assert!(pool.is_free_idx(2));

        // Reallocate to make sure no UB
        let new_s = pool.pop_free(String::from("safe")).unwrap();
        assert_eq!(*new_s, "safe");
    }

    #[test]
    fn test_from_raw_parts_reconstruct() {
        let pool = ObjectPoolManager::new(2);
        let obj = pool.pop_free(Box::new(7)).unwrap();

        let raw = unsafe { obj.into_raw_ptr() };
        let id = unsafe { ObjectPoolManager::get_idx_by_ptr_unchecked(&pool, raw) };
        assert_eq!(id, 1);
        let reconstructed = unsafe { Object::from_raw_parts(raw, &pool) };
        assert_eq!(*reconstructed, Box::new(7));

        // Dropping reconstructed frees slot
        drop(reconstructed);
        let obj2 = pool.pop_free(Box::new(99)).unwrap();
        assert_eq!(obj2.id(), 1);
    }
}
