# axiom_mem

A small, low-level Rust packages that implements fixed-capacity, single-threaded memory pools:
- contiguous raw buffers for fixed-size blocks,
- bitmap-based free-index tracking,
- typed object pools that provide RAII handles which free their slot on drop.

The crate is intended as a compact building block for systems that need deterministic, low-overhead allocation of many fixed-size slots without per-item heap allocations.

## Status
Stable for single-threaded usage patterns and experimental for broader use. The crate intentionally uses unsafe primitives to attain low overhead and therefore requires careful use.

## Key goals
- No per-item heap allocations after pool initialization.
- Constant-time allocation and free (amortized O(1)).
- Minimal metadata per slot.
- Deterministic lifetime and reclamation via RAII handles.

## Build & test
Build:
```sh
cargo build
```

Run tests:
```sh
cargo test
```

Run example binary (small demo in `src/main.rs`):
```sh
cargo run 
```

## Project layout (high level)
- src/lib.rs — crate root and public re-exports.
- src/main.rs — small example/demo showing basic usage.
- src/cache_padded.rs — a CachePadded wrapper (cache-line padding utility).
- src/buff_manager/
  - static_array_buffer.rs — contiguous buffer backing, pool manager, block handle (Buff).
  - static_free_idx_map.rs — FreeIdxManager: bitmap-based free / retire index map.
  - static_object_buffer.rs — ObjectPoolManager and typed Object handle.

## Module summaries & API overview

Note: method names below reflect the public API shape used across the codebase (constructor names and common methods). Inspect the source for exact signatures.

- buff_manager::RawBuffers (static_array_buffer.rs)
  - Purpose: single allocation that holds N contiguous slots of T (stored as uninitialized memory).
  - Responsibilities: allocate/deallocate the memory region, provide raw pointer arithmetic and per-slot access helpers.
  - Safety: exposes raw pointers; callers must ensure correct initialization and not create aliasing violations.

- buff_manager::BufferPoolManager<T> (static_array_buffer.rs)
  - Purpose: manage a pool of fixed-size buffers (blocks) that each contain a fixed number of T elements.
  - Typical API:
    - new(elements_per_block: usize, block_count: usize) -> Self
    - pop_free() -> Option<Buff<T>> — allocate a free block, returning a RAII handle.
  - Use: good for grouping small arrays/records into slot-sized blocks.

- buff_manager::Buff<T> (static_array_buffer.rs)
  - Purpose: RAII, non-owning handle to a single block returned by BufferPoolManager.
  - Features:
    - read_unchecked / write_unchecked (unsafe) helpers for indexed access to the block.
    - On Drop the handle returns its slot to the pool (via FreeIdxManager).
  - Safety: indexing and reads/writes are unchecked; user must ensure element initialization correctness and no aliasing.

- buff_manager::FreeIdxManager (static_free_idx_map.rs)
  - Purpose: compact bitmap that tracks free vs used slot indices.
  - Features:
    - allocate a free index in O(1) amortized,
    - retire / free an index,
    - sanity checks to detect double-free or invalid retire.
  - Tests: module contains unit tests that validate allocation/exhaustion/retire behavior.

- buff_manager::ObjectPoolManager<T> (static_object_buffer.rs)
  - Purpose: typed object pool where objects of type T are stored in contiguous memory slots.
  - Typical API:
    - new(capacity: usize) -> Self
    - pop_free(value: T) -> Option<Object<T>> — place value into a free slot and return an Object handle.
  - Objects are kept in MaybeUninit slots and dropped explicitly by Object's Drop impl.

- buff_manager::Object<T> (static_object_buffer.rs)
  - Purpose: RAII handle returned by ObjectPoolManager::pop_free.
  - Features:
    - Deref / DerefMut implementations to access the stored T.
    - Drop implementation that calls T's destructor and frees the slot in the pool.
  - Safety: object access is safe as long as the handle is alive. Do not extract raw pointers and use them after drop.

- cache_padded::CachePadded (cache_padded.rs)
  - Purpose: small wrapper that pads an inner value to a cache-line boundary to avoid false sharing in concurrent contexts.
  - Note: the crate is currently single-threaded; CachePadded is included as a utility for future extensions.

## Usage examples

Buffer pool (blocks of 4 u32, 3 blocks):
```rust
use axiom_mem::buff_manager::BufferPoolManager;

let pool = BufferPoolManager::<u32>::new(4, 3);
if let Some(mut block) = pool.pop_free() {
    // unsafe indexed write/read helpers
    unsafe { block.write_unchecked(0, 42); }
    let val = unsafe { block.read_unchecked(0) };
    assert_eq!(val, 42);
    // dropping block frees the slot
}
```

Object pool (typed values):
```rust
use axiom_mem::buff_manager::ObjectPoolManager;

let pool = ObjectPoolManager::<String>::new(8);
let obj = pool.pop_free("hello".to_string()).expect("slot available");
assert_eq!(&*obj, "hello");
// dropping obj will call String's destructor and free slot
```

## Safety & soundness notes (read carefully)
- The library uses UnsafeCell, MaybeUninit, raw pointers, and manual allocation. These choices are deliberate to keep runtime overhead low.
- The crate is intended for single-threaded usage or with external synchronization provided by the caller. It is not thread-safe unless you add synchronization.
- Misuse patterns that can lead to undefined behavior:
  - Double-retire (freeing an index twice).
  - Use-after-free of a block or object.
  - Violating aliasing rules by making multiple mutable references into the same slot.
  - Leaking raw pointers into long-lived structures and using them after the originating pool was dropped.
- The public RAII handles (Buff, Object) are designed to reduce these mistakes; prefer them rather than manual pointer manipulation.

## Performance characteristics
- Allocation/free operations are constant-time and designed to be low-overhead.
- Memory layout is compact: contiguous slots reduce fragmentation and improve cache locality.
- No per-item heap allocation after pool initialization, reducing allocator pressure.
- Microbenchmarks will depend on usage patterns and element size; test with realistic workloads.

## Testing
- Unit tests are colocated in modules (see `src/buff_manager/*.rs`) and cover allocation/exhaustion/retire scenarios.
- Run full test suite with `cargo test`.

## Extending the crate
Possible next steps or contributions:
- Optional thread-safe wrappers / internal atomics for concurrent allocation.
- Growable pools that can resize the underlying buffer at runtime.
- Alternative index allocators (lock-free freelists, stacked free-lists).
- Benchmarks and CI for performance regression testing.

## Example binary
`src/main.rs` contains a small example that demonstrates:
- creating a buffer pool and writing/reading values,
- creating an object pool and allocating a typed object,
- printing example output.

## Contributing
- Fork, implement, and open PRs.
- Keep changes focused and add tests for behavior changes.
- Follow the repository style and run `cargo fmt` and `cargo clippy` before submitting.
