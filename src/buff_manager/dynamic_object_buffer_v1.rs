use std::{alloc::{Layout, alloc, dealloc}, cell::{Cell, UnsafeCell}, marker::PhantomData, mem::{self, MaybeUninit}, ops::{Deref, DerefMut}, ptr::NonNull};

use crate::buff_manager::dynamic_free_idx_map_v1::DynFreeIdxManagerV1;
#[cfg(target_arch = "aarch64")]
const TAG_BITS: usize = 8; // use top-byte (TBI)
#[cfg(not(target_arch = "aarch64"))]
const TAG_BITS: usize = 3; // use low bits (alignment)

#[derive(Clone)]
struct Tracker<T>{
    pub(crate) base_ptr:NonNull<MaybeUninit<T>>,
    pub(crate) free_map_start_idx:u32,
    // MaxLen per block is 2^16 - 1
    pub(crate) len:u32,
}

impl<T> Tracker<T>{
    pub fn new(size:u32)->Self{
        let layout = Self::layout(size);
        let base_ptr = NonNull::new(
            unsafe {
                alloc(layout) as *mut _

            }
        ).expect("Unable to allocate Memory");

        Self{
            base_ptr,
            free_map_start_idx:0,
            len:size
        }
        
    }
    
    // User should change it cousiesly
    pub const unsafe fn set_free_map_start_idx(&mut self,idx:u32){
        self.free_map_start_idx = idx
    }

    const fn layout(size:u32)->Layout{
        match Layout::array::<MaybeUninit<T>>(size as usize){
            Ok(layout )=>layout,
            Err(_)=>panic!("Error while Creating Layout")
        }

    }
    pub fn calculate_idx_by_ptr(&self,ptr:NonNull<MaybeUninit<T>>)->u32{
        self.free_map_start_idx + unsafe { ptr.offset_from(self.base_ptr) as u32}
    }
}

impl<T> Drop for Tracker<T>{
    fn drop(&mut self) {
        let layout = Self::layout(self.len);
        unsafe { dealloc(self.base_ptr.as_ptr() as _, layout) }
    }
}

pub struct Object<'o, T> {
    /// Pointer to the object inside the pool.
    ptr: NonNull<MaybeUninit<T>>,

    /// Reference to the pool that owns this object.
    pool_ref: &'o DynamicObjectPoolManagerV1<T>,

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
    pub fn into_inner(self) -> T {
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
    pub const unsafe fn into_tuple(self) -> (NonNull<T>, u32, &'o DynamicObjectPoolManagerV1<T>) {
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
        pool_ref: &'o DynamicObjectPoolManagerV1<T>,
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
    ///   within the lifetime of the [`DynamicObjectPoolManagerV1`] that owns it.
    /// - The pool will not automatically reclaim or drop the value when this pointer
    ///   is no longer used; the user must call respective [`DynamicObjectPoolManagerV1::retire_by_ptr()`]
    ///   manually when done.
    ///
    /// # Example
    /// ```
    /// use axiom_mem::buff_manager::DynamicObjectPoolManagerV1;
    /// let pool = DynamicObjectPoolManagerV1::new(8);
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
    pub unsafe fn from_raw_parts(ptr: NonNull<T>, pool_ref: &'o DynamicObjectPoolManagerV1<T>) -> Self {
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

pub struct DynamicObjectPoolManagerV1<T>{
    free_idx_map:UnsafeCell<DynFreeIdxManagerV1>,
    sorted_tracker_vec:Vec<Tracker<T>>,
    tracker_vec_empty_idx_list:Vec<Tracker<T>>,
    size:u32,
    free_slots:u32,
    _marker:PhantomData<Cell<()>>
}



impl<T> DynamicObjectPoolManagerV1<T>{
    const DEFAULT_CAPACITY:u16 = 5;
    const DEFAULT_SIZE:u16 = 2048;
    const DEFAULT_GROW_CAPACITY:u16 = 1024;
    pub fn new_with(size:u32)->Self{
        let new_size = size.next_multiple_of(size);
        let free_idx_map = UnsafeCell::new(DynFreeIdxManagerV1::new(new_size));
        let mut sorted_tracker_vec = Vec::with_capacity(Self::DEFAULT_CAPACITY as usize);
        let tracker = Tracker::new(size);
        let mut tracker_vec_empty_idx_list = Vec::with_capacity(Self::DEFAULT_CAPACITY as usize);
        tracker_vec_empty_idx_list.push((&tracker).clone());
        sorted_tracker_vec.push(tracker);
        Self{
            free_idx_map,
            sorted_tracker_vec,
            tracker_vec_empty_idx_list,
            size,
            free_slots:size,
            _marker:PhantomData
        }
        
    }

    #[allow(invalid_reference_casting)]
    unsafe fn get_mut (&self)->&mut Self{
        unsafe { &mut *((self as *const Self as usize) as *mut Self) }
        
    }

    pub unsafe fn retire(&self,id:u32){
        unsafe { (*self.free_idx_map.get()).retire(id) }
    }

    pub unsafe fn get_idx_by_ptr_unchecked(&self,ptr:NonNull<T>)->u32{
        unimplemented!()
    }
}