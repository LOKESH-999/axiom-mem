//! Fixed-size slot pool with O(1) allocation and stable memory layout.
//!
//! ## Overview
//!
//! This module provides a low-level, high-performance memory pool built on:
//!
//! - [`FreeIdxManager`] → bitmap-based free index tracking
//! - [`FixedPool`]      → raw memory + index mapping
//! - [`Slot`]           → ownership + lifecycle of allocated values
//!
//! The design prioritizes:
//! - predictable latency (no scanning, no division)
//! - cache locality
//! - minimal branching
//!
//! ## Core Model
//!
//! Memory is pre-allocated and divided into fixed slots.
//!
//! Each slot is represented by a single bit:
//!
//! - `occupied (0)` → contains initialized `T`
//! - `free (1)`     → contains uninitialized memory
//!
//! This invariant must always hold.
//!
//! ## Safety Model
//!
//! The API is split into:
//!
//! - Safe methods → enforce invariants
//! - Unsafe methods → assume invariants
//!
//! Unsafe methods must only be used when correctness is already proven.
//!
//! ## Shared Memory (mmap)
//!
//! When using mmap-backed allocations:
//!
//! - Memory may be visible across processes
//! - This module provides **no synchronization**
//! - Other processes must treat memory as:
//!   - read-only
//!   - or externally synchronized
//!
//! The shared mapping is intended only for:
//! - observation
//! - debugging
//! - external coordination
//!
//! It is NOT safe for concurrent mutation.
//!
//! ## Constraints
//!
//! - `T` must not be zero-sized
//! - Proper alignment must be respected
//! - Pointer arithmetic assumes same allocation origin
use std::{
    alloc::{Layout, alloc, dealloc},
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::NonNull,
};

use crate::{
    free_idx_map::FreeIdxManager,
    slot_pool::static_pool::{
        errors::{PoolError, PoolResult},
        objects::object::Slot,
    },
};

/// Defines how the pool should allocate its backing memory.
pub enum AllocMode {
    /// Uses the standard global allocator (default).
    Heap,
    /// Uses a custom allocation and deallocation function.
    Custom {
        alloc_fn: unsafe fn(Layout) -> NonNull<u8>,
        dealloc_fn: unsafe fn(NonNull<u8>, Layout),
    },
    #[cfg(all(feature = "mmap", unix))]
    /// Uses an anonymous memory map (private to this process).
    MmapAnon,
    #[cfg(all(feature = "mmap", unix))]
    /// Uses a named shared memory map via POSIX shared memory (IPC).
    MmapSharedNamed(String),
}

/// A fixed-capacity, index-backed memory pool for `T`.
///
/// This pool provides:
/// - O(1) allocation via bitmap-based index management
/// - Stable memory addresses (no relocation)
/// - Zero-overhead pointer-to-index mapping using `offset_from`
///
/// ## Design Model
///
/// The pool separates responsibilities:
///
/// - [`FreeIdxManager`] → tracks allocation state (free/occupied)
/// - [`FixedPool`]      → manages memory region and index mapping
/// - [`Slot`]           → owns the lifecycle of an allocated object
///
/// ## Initialization Invariant
///
/// A slot is considered:
///
/// - **occupied** → contains an initialized `T`
/// - **free**     → contains uninitialized memory
///
/// This invariant must always hold.
///
/// ## Drop Semantics
///
/// The pool **does NOT drop remaining elements on `Drop`**.
///
/// This is intentional:
/// - The caller (via [`Slot`]) is responsible for dropping values.
/// - Dropping the pool with live elements results in **leaked values**.
///
/// In debug builds, you are expected to ensure all slots are freed.
///
/// ## Safety Model
///
/// Safe APIs:
/// - Validate pointer bounds
/// - Prevent double-free
///
/// Unsafe APIs:
/// - Assume all invariants are upheld by the caller
/// - Can cause undefined behavior if misused
///
/// ## Shared Memory (Mmap)
///
/// When using shared memory modes:
/// - The memory region is shared across processes
/// - This pool **does NOT provide synchronization**
/// - Other processes must treat memory as **read-only or externally synchronized**
///
/// The shared map is intended **only for observation or external coordination**,
/// not concurrent mutation.
///
/// ## Restrictions
///
/// - Zero-sized types (`T`) are not supported
/// - `T` must be properly aligned
pub struct FixedPool<T> {
    free_idx_map: UnsafeCell<FreeIdxManager>,
    base_ptr: NonNull<MaybeUninit<T>>,
    memory: AllocMode,
    capacity: u32,
}

impl<T> FixedPool<T> {
    #[inline(always)]
    /// Returns a mutable reference to the internal [`FreeIdxManager`].
    ///
    /// Internally uses [`UnsafeCell`] to allow mutation through `&self`.
    ///
    /// ## Safety
    ///
    /// Caller (this module) must ensure:
    /// - no concurrent access
    /// - no aliasing mutable references
    ///
    /// Violating this leads to undefined behavior.
    fn get_mut_free_idx_manager(&self) -> &mut FreeIdxManager {
        unsafe { &mut *self.free_idx_map.get() }
    }

    /// Allocates a new slot and initializes it with `data`.
    ///
    /// ## Behavior
    ///
    /// - Retrieves a free index from the internal bitmap
    /// - Writes `data` into the corresponding memory slot
    /// - Returns a [`Slot`] which owns the value
    ///
    /// ## Returns
    ///
    /// - `Ok(Slot)` → allocation successful
    /// - `Err(data)` → pool is full
    ///
    /// ## Invariants
    ///
    /// - The returned slot corresponds to an initialized `T`
    /// - The index is marked as occupied in the bitmap
    ///
    /// ## Safety
    ///
    /// Internally uses unsafe memory writes, but is safe because:
    /// - Index is guaranteed valid
    /// - Memory is properly allocated
    #[inline]
    pub fn alloc(&self, data: T) -> Result<Slot<'_, T>, T> {
        let idx = self.get_mut_free_idx_manager().get_free_idx();
        if idx == FreeIdxManager::NULL_IDX {
            return Err(data);
        }
        let ptr = unsafe {
            let ptr = self.base_ptr.as_ptr().add(idx as usize);
            (*ptr).write(data);
            NonNull::new_unchecked(ptr)
        };
        let slot = Slot {
            // safe by allocation
            ptr,
            pool_ref: &self,
            idx,
        };
        Ok(slot)
    }

    /// Checks whether a slot at `idx` is free.
    ///
    /// ## Errors
    ///
    /// Returns [`PoolError::FreeMap`] if:
    /// - `idx` is out of bounds
    #[inline(always)]
    pub fn is_free(&self, idx: u32) -> PoolResult<bool> {
        self.get_mut_free_idx_manager()
            .is_free(idx)
            .map_err(PoolError::from)
    }

    /// Marks a slot as free without any validation.
    ///
    /// ## Safety
    ///
    /// Caller must guarantee:
    /// - `idx` is within bounds
    /// - The slot is currently occupied
    /// - The value at `idx` has already been dropped
    ///
    /// Violating these conditions may result in:
    /// - Double free
    /// - Memory corruption
    /// - Undefined behavior
    #[inline(always)]
    pub unsafe fn unchecked_retire(&self, idx: u32) {
        unsafe { self.get_mut_free_idx_manager().unchecked_retire(idx) };
    }

    /// Converts a pointer to its corresponding slot index.
    ///
    /// ## Safety
    ///
    /// Caller must guarantee:
    /// - `ptr` was allocated from this pool
    /// - `ptr` lies within the pool memory range
    ///
    /// No bounds checking is performed.
    /// Invalid pointers result in undefined behavior.
    #[inline(always)]
    pub unsafe fn ptr_to_idx_unchecked(&self, ptr: NonNull<T>) -> u32 {
        // Checking `ptr` > `self.base_ptr`
        debug_assert!(ptr.as_ptr().cast() >= self.base_ptr.as_ptr());
        // Getting the Index from ptr
        unsafe { ptr.cast().offset_from(self.base_ptr) as u32 }
    }

    /// Converts a pointer into its corresponding slot index.
    ///
    /// ## Behavior
    ///
    /// - Validates pointer bounds
    /// - Computes index using `offset_from`
    ///
    /// ## Errors
    ///
    /// - [`PointerOutOfBounds`] if pointer is not within pool
    /// - [`PointerMisaligned`] if pointer is not aligned to `T`
    ///
    /// ## Notes
    ///
    /// This operation does not use division.
    /// Index calculation is constant-time.
    #[inline(always)]
    fn ptr_to_idx(&self, ptr: NonNull<T>) -> PoolResult<u32> {
        let base = self.base_ptr.as_ptr();
        let end = unsafe { base.add(self.capacity as usize) };

        let p = ptr.as_ptr().cast();

        // 1. Bounds check (pure pointer comparison)
        if p < base || p >= end {
            return Err(PoolError::PointerOutOfBounds);
        }

        // 2. Compute index using built-in offset (NO division in your code)
        let idx = unsafe { p.offset_from(base) as usize };

        // 3. Optional: debug alignment check (can be removed in release)
        debug_assert_eq!(
            (p as usize - base as usize) % std::mem::size_of::<T>(),
            0,
            "misaligned pointer"
        );

        Ok(idx as u32)
    }

    /// Drops a value and marks its slot as free without validation.
    ///
    /// ## Safety
    ///
    /// Caller must guarantee:
    /// - `ptr` is valid and belongs to this pool
    /// - `ptr` points to an initialized `T`
    /// - The slot is not already freed
    ///
    /// This function:
    /// - Drops the value
    /// - Frees the corresponding index
    ///
    /// Violating invariants may cause:
    /// - Double drop
    /// - Memory corruption
    /// - Undefined behavior
    #[inline(always)]
    pub unsafe fn retire_by_ptr_unchecked(&self, ptr: NonNull<T>) {
        unsafe {
            // Just checking weather the pointer in range or not
            debug_assert!(
                ptr.as_ptr().cast() <= self.base_ptr.as_ptr().add(self.capacity as usize)
            );
            // Drop the value only after verifying it is live.
            std::ptr::drop_in_place(ptr.as_ptr());
            // getting the drop idx
            let idx = self.ptr_to_idx_unchecked(ptr);
            // Marking the drop idx as free
            self.unchecked_retire(idx);
        }
    }

    /// Drops a value and frees its slot.
    ///
    /// ## Behavior
    ///
    /// - Validates pointer belongs to the pool
    /// - Ensures slot is currently occupied
    /// - Drops the value
    /// - Marks slot as free
    ///
    /// ## Errors
    ///
    /// - [`PointerOutOfBounds`] → pointer not in pool
    /// - [`SlotAlreadyFree`] → double free attempt
    ///
    /// ## Guarantees
    ///
    /// - Safe against double free
    /// - Safe pointer validation
    #[inline(always)]
    pub fn retire_by_ptr(&self, ptr: NonNull<T>) -> PoolResult<()> {
        let idx = self.ptr_to_idx(ptr)?;

        // Verify the slot is currently occupied before dropping it.
        if self.get_mut_free_idx_manager().is_free(idx)? {
            return Err(PoolError::SlotAlreadyFree);
        }

        unsafe {
            // Drop the value only after verifying it is live.
            std::ptr::drop_in_place(ptr.as_ptr());

            // Mark it as free using the inner unsafe API.
            self.get_mut_free_idx_manager().unchecked_retire(idx);
        }

        Ok(())
    }
}

impl<T> FixedPool<T> {
    const fn layout(size: u32) -> Layout {
        match Layout::array::<MaybeUninit<T>>(size as usize) {
            Ok(l) => l,
            Err(_) => panic!("Invalid layout"),
        }
    }

    /// Creates a new fixed pool with given capacity.
    ///
    /// ## Behavior
    ///
    /// - Allocates contiguous memory for `capacity` elements
    /// - Initializes internal free index manager
    ///
    /// ## Panics
    ///
    /// - If `T` is zero-sized
    /// - If allocation fails
    pub fn new(size: u32) -> Self {
        Self::new_with(size, AllocMode::Heap)
    }

    /// Creates a new fixed pool with given capacity.
    ///
    /// ## Behavior
    ///
    /// - Allocates contiguous memory for `capacity` elements
    /// - Initializes internal free index manager
    ///
    /// ## Panics
    ///
    /// - If `T` is zero-sized
    /// - If allocation fails
    pub fn new_with(size: u32, memory: AllocMode) -> Self {
        assert!(std::mem::size_of::<T>() > 0, "ZST not supported");
        let layout = Self::layout(size);

        let base_ptr = match &memory {
            AllocMode::Heap => unsafe {
                NonNull::new(alloc(layout) as *mut _).expect("Heap allocation failed")
            },
            AllocMode::Custom { alloc_fn, .. } => unsafe { alloc_fn(layout).cast() },
            #[cfg(all(feature = "mmap", unix))]
            AllocMode::MmapAnon => Self::mmap_anon(layout.size()).cast(),
            #[cfg(all(feature = "mmap", unix))]
            AllocMode::MmapSharedNamed(map_name) => {
                Self::mmap_shared_named(map_name, layout.size()).cast()
            }
        };

        Self {
            free_idx_map: UnsafeCell::new(FreeIdxManager::new(size)),
            base_ptr,
            memory,
            capacity: size,
        }
    }

    #[cfg(all(feature = "mmap", unix))]
    fn mmap_anon(size: usize) -> NonNull<u8> {
        unsafe {
            let addr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if addr == libc::MAP_FAILED {
                panic!("mmap anon failed");
            }
            NonNull::new(addr as *mut u8).unwrap()
        }
    }

    /// Creates or opens a named shared memory region.
    ///
    /// ## Important
    ///
    /// This shared memory is intended for:
    /// - Observation
    /// - External coordination
    ///
    /// It does NOT provide:
    /// - Synchronization
    /// - Atomicity
    /// - Safety across processes
    ///
    /// Concurrent mutation across processes without external synchronization
    /// results in undefined behavior.
    #[cfg(all(feature = "mmap", unix))]
    fn mmap_shared_named(name: &str, size: usize) -> NonNull<u8> {
        unsafe {
            let c_name = std::ffi::CString::new(name).expect("Invalid C string");

            //  Open POSIX shared memory object
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666);

            if fd < 0 {
                panic!("shm_open failed");
            }

            //  Set the size of the shared memory object
            if libc::ftruncate(fd, size as libc::off_t) != 0 {
                libc::close(fd);
                panic!("ftruncate failed");
            }

            //  Map it into memory
            let addr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );

            libc::close(fd); // FD is no longer needed after mapping

            if addr == libc::MAP_FAILED {
                panic!("mmap shared failed");
            }

            NonNull::new(addr as *mut u8).unwrap()
        }
    }
}

impl<T> Drop for FixedPool<T> {
    fn drop(&mut self) {
        let layout = Self::layout(self.capacity);
        let ptr: *mut std::ffi::c_void = self.base_ptr.as_ptr().cast();

        unsafe {
            match &self.memory {
                AllocMode::Heap => {
                    dealloc(self.base_ptr.as_ptr().cast(), layout);
                }
                AllocMode::Custom { dealloc_fn, .. } => {
                    dealloc_fn(self.base_ptr.cast(), layout);
                }
                #[cfg(all(feature = "mmap", unix))]
                AllocMode::MmapAnon | AllocMode::MmapSharedNamed(_) => {
                    // Let the OS clean up the shared file,
                    // or let the user do it manually so the debugger isn't broken.
                    libc::munmap(ptr, layout.size());
                }
            }
        }
    }
}
