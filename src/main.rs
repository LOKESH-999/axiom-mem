#![feature(allocator_api)]

//! # Example: Using Buffer and Object Pools
//!
//! This example demonstrates basic usage of the `axiom_mem` crate:
//! - Allocating memory from a fixed-size buffer pool
//! - Allocating typed objects from an object pool
//! - Automatic reclamation of slots via RAII handles
//!
//! Safety: Many operations in this crate are unsafe for performance reasons. 
//! Unsafe functions (like `read_unchecked` / `write_unchecked`) must only be used correctly, 
//! i.e., initialized slots and valid indices.

fn main() {
    // -------------------------------------------------------------------------
    // Buffer pool example (fixed-size blocks of u32, 4 elements per block, 3 blocks)
    // -------------------------------------------------------------------------
    {
        use axiom_mem::buff_manager::BufferPoolManager;

        let pool = BufferPoolManager::<u32>::new(4, 3);

        if let Some(block) = pool.pop_free() {
            // write into the block (unsafe API)
            unsafe { block.write_unchecked(0, 42); }

            // read back the value
            let val = unsafe { block.read_unchecked(0) };
            println!("buffer block[0] = {}", val);

            // dropping `block` automatically returns the slot to the pool
        } else {
            println!("no buffer available");
        }
    }

    // -------------------------------------------------------------------------
    // Object pool example (typed objects)
    // -------------------------------------------------------------------------
    {
        use axiom_mem::buff_manager::ObjectPoolManager;

        let pool = ObjectPoolManager::<String>::new(8);

        if let Ok(obj) = pool.pop_free("hello from object pool".to_string()) {
            println!("object contains: {}", &*obj);
            // dropping `obj` will call the destructor and free the slot
        } else {
            println!("no object slot available");
        }
    }

    println!("example run complete");
}
