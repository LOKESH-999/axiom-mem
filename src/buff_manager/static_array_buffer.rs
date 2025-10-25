//! # Buffer Pool Manager Core
//!
//! This module provides low-level memory and index management primitives for
//! a fixed-size buffer pool allocator. It includes:
//!
//! - [`RawBuffers`]: manages a contiguous, non-owning memory region.
//! - [`FreeIdxManager`]: tracks free and used buffer indices via a compact bitmap.
//! - [`BufferPoolManager`]: coordinates allocation and retire of buffer blocks.
//! - [`Buff`]: a lightweight, non-owning handle to a buffer block.
//!
//! ## Design Goals
//!
//! - Minimal per-block overhead using bit-level tracking.
//! - No dynamic allocations after initialization.
//! - Optimized for single-threaded, low-latency usage.
//! - Safe abstractions over unsafe memory and pointer manipulations.

use std::{
    alloc::{Layout, alloc, dealloc},
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::NonNull,
};

use crate::buff_manager::static_free_idx_map::FreeIdxManager;

/// Represents a contiguous memory region divided into fixed-size blocks.
///
/// This struct holds:
/// - A non-null pointer to the start of the memory region.
/// - The size of each block in bytes (`block_size`).
/// - The total number of blocks (`n_blocks`).
///
/// It **does not** manage the allocation or deallocation of memory.
/// The user (or higher-level manager) is responsible for ensuring
/// the validity of the memory pointed to by `buff`.
///
/// # Safety
/// This structure assumes the pointer is valid for reads/writes up to
/// `block_size * n_blocks` bytes. No lifetime tracking or aliasing checks are enforced.
struct RawBuffers<T> {
    /// Non-null pointer to the start of the buffer.
    buff: NonNull<MaybeUninit<T>>,
    /// Size of each block (in units of `T` or bytes, depending on context).
    block_size: u32,
    /// Total number of blocks in the buffer.
    n_blocks: u32,
}
impl<T> RawBuffers<T> {
    pub fn new(block_size: u32, n_blocks: u32) -> Self {
        assert!(block_size > 0, "`block_size` must be > `0`");
        assert!(n_blocks > 0, "`n_blocks` must be > `0`");
        let layout = Self::layout(block_size, n_blocks);
        let ptr = NonNull::new(unsafe { alloc(layout) as *mut MaybeUninit<T> })
            .expect("FAILED WHILE ALLOCATING MEMORY");
        Self {
            buff: ptr,
            block_size,
            n_blocks,
        }
    }
    const fn layout(block_size: u32, n_blocks: u32) -> Layout {
        let size = block_size as u64 * n_blocks as u64;
        let layout = Layout::array::<MaybeUninit<T>>(size as usize);
        match layout {
            Ok(lay) => lay,
            Err(_) => panic!("COULD'NT CREATE LAYOUT"),
        }
    }
    const unsafe fn get(&self, idx: u32) -> NonNull<MaybeUninit<T>> {
        let count = self.block_size as u64 * idx as u64;
        unsafe { self.buff.add(count as usize) }
    }
    const fn n_blocks(&self) -> u32 {
        self.n_blocks
    }
}

impl<T> Drop for RawBuffers<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.buff.as_ptr() as _,
                Self::layout(self.block_size, self.n_blocks),
            )
        }
    }
}
/// A lightweight handle referencing a memory buffer region.
///
/// This struct is typically returned by a pool allocator when a client
/// requests a buffer. It holds:
/// - Pointer to the actual buffer region.
/// - The logical ID (`id`) of the block within the pool.
/// - A length field for user-defined capacity semantics.
/// - A reference back to the managing [`BufferPoolManager`].
///
/// This handle is **non-owning**; it does not perform deallocation
/// when dropped. The lifetime `'o` ensures it cannot outlive the manager.
pub struct Buff<'o, T> {
    /// Pointer to the underlying memory block (`MaybeUninit<T>`).
    buff_ptr: NonNull<MaybeUninit<T>>,
    /// Logical index of the block in the pool.
    id: u32,
    /// Capacity of the block (number of `T` elements).
    len: u32,
    /// Reference back to the owning pool manager.
    buff_pool_manager_ref: &'o BufferPoolManager<T>,
}

/////////////////////////////////////////////////////////////
/// Construction and basic metadata access
/////////////////////////////////////////////////////////////
#[allow(clippy::len_without_is_empty)]
impl<'o, T> Buff<'o, T> {
    /// Creates a new `Buff` handle.
    ///
    /// Usually called internally by the pool manager. Users should obtain `Buff`
    /// instances via [`BufferPoolManager::pop_free()`].
    pub const fn new(
        buff_ptr: NonNull<MaybeUninit<T>>,
        id: u32,
        len: u32,
        buff_pool_manager_ref: &'o BufferPoolManager<T>,
    ) -> Self {
        Self {
            buff_pool_manager_ref,
            buff_ptr,
            id,
            len,
        }
    }

    /// Returns the block index in the pool.
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// Returns the capacity of the block (number of `T` slots).
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Returns a raw pointer to the memory block (`MaybeUninit<T>`).
    pub const fn as_mut_ptr(&self) -> *mut MaybeUninit<T> {
        self.buff_ptr.as_ptr()
    }
}

/////////////////////////////////////////////////////////////
/// Writing values
/////////////////////////////////////////////////////////////
impl<'o, T> Buff<'o, T> {
    /// Writes a value into the buffer at a given index **without bounds checking**.
    ///
    /// # Safety
    /// - `idx` must be within `[0, len]`.
    /// - The slot must not already be initialized to avoid double-drop.
    pub unsafe fn write_unchecked(&self, idx: u32, value: T) {
        unsafe {
            let ptr = self.buff_ptr.as_ptr().add(idx as usize);
            (*ptr).write(value);
        }
    }

    /// Writes a value at a given index with bounds checking.
    ///
    /// Returns `true` if successful, or `false` if `idx >= len`.
    pub fn write(&self, idx: u32, value: T) -> bool {
        if idx >= self.len {
            return false;
        }
        unsafe { self.write_unchecked(idx, value) };
        true
    }
}

/////////////////////////////////////////////////////////////
/// Reading values
/////////////////////////////////////////////////////////////
impl<'o, T> Buff<'o, T> {
    /// Reads a value by copying it (requires `T: Copy`).
    ///
    /// # Safety
    /// The element at `idx` must have been initialized.
    pub unsafe fn read_unchecked(&self, idx: u32) -> T
    where
        T: Copy,
    {
        unsafe { *(*self.buff_ptr.as_ptr().add(idx as usize)).as_ptr() }
    }

    /// Returns a reference to an initialized element.
    ///
    /// # Safety
    /// - The element at `idx` must be initialized.
    /// - `idx` must be within `[0, len)`.
    pub unsafe fn get_ref(&self, idx: u32) -> &T {
        unsafe { &*(*self.buff_ptr.as_ptr().add(idx as usize)).as_ptr() }
    }

    /// Returns a mutable reference to an initialized element.
    ///
    /// # Safety
    /// - The element at `idx` must be initialized.
    /// - `idx` must be within `[0, len]`.
    pub unsafe fn get_mut(&mut self, idx: u32) -> &mut T {
        unsafe { &mut *(*self.buff_ptr.as_ptr().add(idx as usize)).as_mut_ptr() }
    }
}

/////////////////////////////////////////////////////////////
/// Slice access over initialized portions
/////////////////////////////////////////////////////////////
impl<'o, T> Buff<'o, T> {
    /// Returns a slice over the first `init_len` initialized elements.
    ///
    /// # Safety
    /// `init_len` elements must have been initialized.
    pub unsafe fn as_slice(&self, init_len: u32) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buff_ptr.as_ptr() as *const T, init_len as usize) }
    }

    /// Returns a mutable slice over the first `init_len` initialized elements.
    ///
    /// # Safety
    /// `init_len` elements must have been initialized.
    pub unsafe fn as_slice_mut(&mut self, init_len: u32) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.buff_ptr.as_ptr() as *mut T, init_len as usize)
        }
    }
}

/////////////////////////////////////////////////////////////
/// Initialization helpers
/////////////////////////////////////////////////////////////
impl<'o, T> Buff<'o, T> {
    /// Fills all slots with `T::default()`.
    ///
    /// # Safety
    /// Must not overwrite already-initialized slots to avoid double-drop.
    pub unsafe fn fill_default(&self)
    where
        T: Default,
    {
        for i in 0..self.len {
            unsafe { self.write_unchecked(i, T::default()) }
        }
    }
}

/////////////////////////////////////////////////////////////
/// Byte-level access
/////////////////////////////////////////////////////////////
impl<'o, T> Buff<'o, T> {
    /// Returns a byte slice over the underlying memory.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.buff_ptr.as_ptr() as *const u8,
                (self.len as usize) * std::mem::size_of::<T>(),
            )
        }
    }

    /// Returns a mutable byte slice over the underlying memory.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buff_ptr.as_ptr() as *mut u8,
                (self.len as usize) * std::mem::size_of::<T>(),
            )
        }
    }
}

/////////////////////////////////////////////////////////////
/// Lifecycle management
/////////////////////////////////////////////////////////////
impl<T> Drop for Buff<'_, T> {
    /// Releases the block back to the pool manager when dropped.
    ///
    /// # Safety
    /// Assumes all necessary drops for initialized elements have been handled.
    fn drop(&mut self) {
        unsafe { self.buff_pool_manager_ref.retire(self.id) };
    }
}

/// A high-level controller that manages both the raw memory buffers
/// and their allocation bitmap.
///
/// `BufferPoolManager` acts as the central allocator and recycler
/// for fixed-size buffer blocks. It provides an abstraction over:
/// - [`RawBuffers<T>`]: the contiguous memory pool of `T` blocks.
/// - [`FreeIdxManager`]: a bitmap or freelist tracker for managing
///   which blocks are free or occupied.
///
/// This type can serve as the foundation for a cache, memory slab, or
/// general-purpose allocator where allocations and releases are tracked
/// manually and efficiently.
///
/// # Layout Overview
/// ```text
/// ┌────────────────────────────┐
/// │  BufferPoolManager<T>  │
/// │ ┌────────────────────────┐ │
/// │ │   RawBuffers<T>        │ │  -> owns raw block memory
/// │ └────────────────────────┘ │
/// │ ┌────────────────────────┐ │
/// │ │   FreeIdxManager       │ │  -> tracks free/used indices
/// │ └────────────────────────┘ │
/// └────────────────────────────┘
/// ```
///
/// # Safety
/// This struct internally uses [`UnsafeCell`] for interior mutability.
/// It assumes single-threaded or externally synchronized usage.
/// The caller must ensure:
/// - No aliasing mutable access to buffers or index maps occurs.
/// - Released indices are not reused after `retire()` until reallocated.
/// - `Buff<'_, T>` instances obtained from [`pop_free()`] do not outlive
///   the manager or overlap with other active `Buff` instances.
///
/// # Notes
/// - `Send`, `Sync`, `Clone`, and `Copy` are intentionally disabled
///   via [`PhantomData<Cell<()>>`].
/// - Future extensions may add async-aware or lock-free APIs.
pub struct BufferPoolManager<T> {
    /// Low-level buffer managing contiguous blocks of memory.
    buff: UnsafeCell<RawBuffers<T>>,

    /// Bitmap-based free list manager for block allocation tracking.
    free_list_manager: UnsafeCell<FreeIdxManager>,

    /// Marker type to prevent auto-traits like `Send`, `Sync`, `Clone`, and `Copy`.
    _marker: PhantomData<Cell<()>>,
}

impl<T> BufferPoolManager<T> {
    /// Creates a new buffer pool manager.
    ///
    /// # Arguments
    /// * `block_size` – Size of each individual block in bytes or units of `T`.
    /// * `n_blocks` – Total number of blocks in the pool.
    ///
    /// # Returns
    /// A [`BufferPoolManager`] instance initialized with
    /// a contiguous memory pool and an empty free-list map.
    pub fn new(block_size: u32, n_blocks: u32) -> Self {
        let buff = UnsafeCell::new(RawBuffers::new(block_size, n_blocks));
        let free_list_manager = UnsafeCell::new(FreeIdxManager::new(n_blocks));
        Self {
            buff,
            free_list_manager,
            _marker: PhantomData,
        }
    }

    /// Returns total no of blocks/capacity
    pub const fn n_blocks(&self) -> u32 {
        unsafe { &*self.buff.get() }.n_blocks()
    }

    /// Attempts to acquire a free buffer block from the pool.
    ///
    /// # Returns
    /// `Some(Buff<'_, T>)` if a free block is available, or `None`
    /// if the pool is fully allocated.
    ///
    /// # Safety
    /// The returned `Buff` provides unique mutable access to a block.
    /// It must be explicitly released using [`retire()`] when done.
    pub const fn pop_free(&self) -> Option<Buff<'_, T>> {
        let (buff, idx_map) = self.get_mut();
        let free_idx = idx_map.get_free_idx();
        if free_idx != FreeIdxManager::NULL_IDX {
            let buff_ptr = unsafe { buff.get(free_idx) };
            return Some(Buff::new(buff_ptr, free_idx, buff.block_size, self));
        }
        None
    }

    /// Returns mutable access to both the raw buffer and free-list manager.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to prevent aliasing.
    #[allow(clippy::mut_from_ref)]
    const fn get_mut(&self) -> (&mut RawBuffers<T>, &mut FreeIdxManager) {
        unsafe {
            let buff = &mut *self.buff.get();
            let idx_map = &mut *self.free_list_manager.get();
            (buff, idx_map)
        }
    }

    /// Releases a previously allocated buffer index, marking it as free.
    ///
    /// # Safety
    /// - `idx` must have been previously allocated by this manager.
    /// - The buffer at `idx` must not be accessed after retire.
    /// - Double retire or use-after-retire is undefined behavior.
    const unsafe fn retire(&self, idx: u32) {
        let (_, idx_map) = self.get_mut();
        unsafe { idx_map.retire(idx) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocate multiple buffers, write different values in each, and verify isolation
    #[test]
    fn multi_buffer_isolation() {
        let pool = BufferPoolManager::<u32>::new(4, 3); // 3 blocks, 4 u32 per block

        let b1 = pool.pop_free().unwrap();
        let b2 = pool.pop_free().unwrap();
        let b3 = pool.pop_free().unwrap();

        unsafe {
            // Fill b1 with 10, b2 with 20, b3 with 30
            for i in 0..b1.len() {
                b1.write_unchecked(i, 10 + i);
                b2.write_unchecked(i, 100 + i);
                b3.write_unchecked(i, 200 + i);
            }

            // Verify each buffer independently
            for i in 0..b1.len() {
                assert_eq!(b1.read_unchecked(i), 10 + i);
                assert_eq!(b2.read_unchecked(i), 100 + i);
                assert_eq!(b3.read_unchecked(i), 200 + i);
            }

            // Overwrite b2 partially and verify isolation
            b2.write_unchecked(0, 999);
            assert_eq!(b2.read_unchecked(0), 999);
            assert_eq!(b1.read_unchecked(0), 10);
            assert_eq!(b3.read_unchecked(0), 200);
        }
    }

    /// Stress allocation: allocate all, write unique pattern, drop some, and reallocate
    #[test]
    fn allocation_pattern_consistency() {
        let n_blocks = 6;
        let block_size = 5;
        let pool = BufferPoolManager::<u32>::new(block_size, n_blocks);

        let mut handles = Vec::new();

        // Allocate all blocks and fill unique pattern
        for i in 0..n_blocks {
            let b = pool.pop_free().unwrap();
            unsafe {
                for j in 0..b.len() {
                    b.write_unchecked(j, (i as u32) * 100 + j);
                }
            }
            handles.push(b);
        }

        // Check patterns
        for (i, b) in handles.iter().enumerate() {
            unsafe {
                for j in 0..b.len() {
                    assert_eq!(b.read_unchecked(j), (i as u32) * 100 + j);
                }
            }
        }

        // Drop first 3 buffers, reallocate, write new pattern, verify
        handles.drain(0..3);
        let mut new_handles = Vec::new();
        for i in 0..3 {
            let b = pool.pop_free().unwrap();
            unsafe {
                for j in 0..b.len() {
                    b.write_unchecked(j, 1000 + (i as u32) * 10 + j);
                }
            }
            new_handles.push(b);
        }

        // Verify old untouched blocks are intact
        for (i, b) in handles.iter().enumerate() {
            unsafe {
                for j in 0..b.len() {
                    assert_eq!(b.read_unchecked(j), ((i + 3) as u32) * 100 + j);
                }
            }
        }

        // Verify new blocks
        for (i, b) in new_handles.iter().enumerate() {
            unsafe {
                for j in 0..b.len() {
                    assert_eq!(b.read_unchecked(j), 1000 + (i as u32) * 10 + j);
                }
            }
        }
    }

    /// Test byte-level writes do not corrupt neighboring buffers
    #[test]
    fn byte_level_memory_safety() {
        let pool = BufferPoolManager::<u32>::new(2, 2);

        let mut b1 = pool.pop_free().unwrap();
        let b2 = pool.pop_free().unwrap();

        unsafe {
            b1.write_unchecked(0, 0x11223344);
            b2.write_unchecked(0, 0x55667788);

            let b1_bytes = b1.as_bytes_mut();
            b1_bytes[0] = 0xFF;

            let val_b1 = b1.read_unchecked(0);
            let val_b2 = b2.read_unchecked(0);

            assert_ne!(val_b1, val_b2, "b1 modification should not affect b2");
        }
    }

    /// Test drop-based retire of buffers and reuse
    #[test]
    fn drop_release_and_reallocate() {
        let pool = BufferPoolManager::<u32>::new(3, 3);

        let b1 = pool.pop_free().unwrap();
        let b2 = pool.pop_free().unwrap();
        let b3 = pool.pop_free().unwrap();

        assert!(pool.pop_free().is_none());

        drop(b2); // retire middle buffer

        let b_new = pool.pop_free().unwrap();
        assert_eq!(b_new.id(), 1, "Released buffer index should be reused");

        drop(b1);
        drop(b3);
        drop(b_new);

        // All blocks should be free now
        let mut collected_ids = Vec::new();
        let mut buff = vec![];
        for _ in 0..3 {
            let b = pool.pop_free().unwrap();
            collected_ids.push(b.id());
            buff.push(b);
        }
        collected_ids.sort();
        assert_eq!(collected_ids, vec![0, 1, 2]);
        assert!(pool.pop_free().is_none());
    }

    /// Edge case: request more than available blocks
    #[test]
    fn over_allocation_returns_none() {
        let pool = BufferPoolManager::<u32>::new(2, 2);

        let _b1 = pool.pop_free().unwrap();
        let _b2 = pool.pop_free().unwrap();

        assert!(
            pool.pop_free().is_none(),
            "Should return None when pool exhausted"
        );
    }

    #[test]
    fn stress_allocation_and_reuse() {
        let pool = BufferPoolManager::<u32>::new(5, 10);

        let mut handles = Vec::new();
        // allocate all
        for _ in 0..10 {
            let b = pool.pop_free().unwrap();
            unsafe { b.fill_default() };
            handles.push(b);
        }
        // pool exhausted
        assert!(pool.pop_free().is_none());

        // drop half
        for b in handles {
            drop(b);
        }
        let mut _unused = Vec::new();
        for _ in 0..5 {
            let b = pool.pop_free().unwrap();
            _unused.push(b);
        }

        // reallocate 5, verify indices reused
        let mut reused_ids = Vec::new();
        let mut buff = Vec::new();
        for _ in 0..5 {
            let b = pool.pop_free().unwrap();
            reused_ids.push(b.id());
            buff.push(b);
        }
        reused_ids.sort();
        assert_eq!(reused_ids, vec![0, 1, 2, 3, 4]); // first half should be reused
    }

    #[test]
    fn independent_buffer_integrity() {
        let pool = BufferPoolManager::<u32>::new(4, 3);
        let b1 = pool.pop_free().unwrap();
        let b2 = pool.pop_free().unwrap();
        let b3 = pool.pop_free().unwrap();

        unsafe {
            for i in 0..b1.len() {
                b1.write_unchecked(i, 100 + i);
            }
            for i in 0..b2.len() {
                b2.write_unchecked(i, 200 + i);
            }
            for i in 0..b3.len() {
                b3.write_unchecked(i, 300 + i);
            }

            for i in 0..b1.len() {
                assert_eq!(b1.read_unchecked(i), 100 + i);
            }
            for i in 0..b2.len() {
                assert_eq!(b2.read_unchecked(i), 200 + i);
            }
            for i in 0..b3.len() {
                assert_eq!(b3.read_unchecked(i), 300 + i);
            }
        }
    }

    #[test]
    fn slice_and_bytes_safety() {
        let pool = BufferPoolManager::<u32>::new(4, 2);
        let mut b1 = pool.pop_free().unwrap();
        let b2 = pool.pop_free().unwrap();

        unsafe {
            b1.fill_default();
            b2.fill_default();
        }

        let b1_bytes = b1.as_bytes_mut();
        b1_bytes[0] = 0xFF;
        unsafe {
            let val1 = b1.read_unchecked(0);
            let val2 = b2.read_unchecked(0);
            assert_ne!(
                val1, val2,
                "Byte-level modification must not affect other buffers"
            );
        }
    }

    #[test]
    fn checked_write_bounds() {
        let pool = BufferPoolManager::<u32>::new(3, 1);
        let b = pool.pop_free().unwrap();

        // inside bounds
        assert!(b.write(0, 10));
        assert!(b.write(2, 20));

        // outside bounds
        assert!(!b.write(3, 30));
        assert!(!b.write(100, 50));
    }

    #[test]
    fn unchecked_write_overflow_simulation() {
        let pool = BufferPoolManager::<u32>::new(1, 1);
        let b = pool.pop_free().unwrap();
        unsafe {
            b.write_unchecked(0, 42); // valid
            let ptr = b.as_mut_ptr();
            // deliberately write beyond len, unsafe, simulate stress
            (*ptr.add(1)).write(999);
        }
    }

    #[test]
    fn drop_and_reuse_multiple_cycles() {
        let pool = BufferPoolManager::<u32>::new(2, 2);
        {
            let b1 = pool.pop_free().unwrap();
            let b2 = pool.pop_free().unwrap();
            assert!(pool.pop_free().is_none());
            drop(b2);
            let b3 = pool.pop_free().unwrap();
            assert_eq!(b3.id(), 0);
            drop(b1);
            drop(b3);
        }
        // all free
        let mut ids = vec![];
        let mut buff = vec![];
        for _ in 0..2 {
            let b = pool.pop_free().unwrap();
            ids.push(b.id());
            buff.push(b);
        }
        ids.sort();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn full_pool_allocation_and_exhaustion() {
        let pool = BufferPoolManager::<u32>::new(1, 1);
        let b = pool.pop_free().unwrap();
        assert!(pool.pop_free().is_none(), "Pool should be exhausted");
        drop(b);
        assert!(
            pool.pop_free().is_some(),
            "Pool should allow reuse after drop"
        );
    }
}
