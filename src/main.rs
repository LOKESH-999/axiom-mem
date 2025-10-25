#![feature(allocator_api)]
fn main() {
    // Example usage of the buffer & object pools in this crate.
    // This file is intentionally small and demonstrates basic allocation,
    // use, and automatic reclamation on Drop.

    // Buffer pool example (fixed-size blocks of u32, 4 elems per block, 3 blocks)
    {
        use axiom_mem::buff_manager::BufferPoolManager;

        let pool = BufferPoolManager::<u32>::new(4, 3);
        if let Some(block) = pool.pop_free() {
            // write into the block (unsafe API provided by the pool)
            unsafe {
                block.write_unchecked(0, 42);
            }
            // read back the value
            let val = unsafe { block.read_unchecked(0) };
            println!("buffer block[0] = {}", val);
            // dropping `block` returns the slot to the pool
        } else {
            println!("no buffer available");
        }
    }

    // Object pool example (typed objects)
    {
        use axiom_mem::buff_manager::ObjectPoolManager;

        let pool = ObjectPoolManager::<String>::new(8);
        if let Ok(obj) = pool.pop_free("hello from object pool".to_string()) {
            println!("object contains: {}", &*obj);
            // dropping `obj` will call its destructor and free the slot
        } else {
            println!("no object slot available");
        }
    }

    println!("example run complete");
}
