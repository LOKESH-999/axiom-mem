use std::{
    alloc::{Layout, alloc, dealloc},
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::NonNull,
};

use crate::free_idx_map::FreeIdxManager;

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

pub struct FixedPool<T> {
    free_idx_map: UnsafeCell<FreeIdxManager>,
    base_ptr: NonNull<MaybeUninit<T>>,
    memory: AllocMode,
    capacity: u32,
}

impl<T> FixedPool<T>{
    pub fn alloc(&self,data:T)->Result<(),T>{
        Err(data)
    }

    fn get_mut_free_idx_manager(&self)->&mut FreeIdxManager{
        unsafe { &mut *self.free_idx_map.get() }
    }

}

impl<T> FixedPool<T> {
    const fn layout(size: u32) -> Layout {
        match Layout::array::<MaybeUninit<T>>(size as usize) {
            Ok(l) => l,
            Err(_) => panic!("Invalid layout"),
        }
    }

    pub fn new(size: u32) -> Self {
        Self::new_with(size, AllocMode::Heap)
    }

    pub fn new_with(size: u32, memory: AllocMode) -> Self {
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
