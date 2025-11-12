//! # Low-Level Memory Utilities
//!
//! `axiom_mem` provides **high-performance, low-level memory management utilities**,
//! allowing management of `T`-typed memory with either **pre-allocated** or
//! **dynamically growing** (auto-resizing) pools.
//!
//! ## Features
//! - `allocator_api` — Enables integration with custom allocators for standard collections.
//!
//! ## Modules
//! - [`buff_manager`] — Core memory pool management, including static and dynamic object pools.
//! - [`cache_padded`] — Provides cache-line padding to reduce false sharing or improve memory layout.
//!
//! ## Safety
//! Many operations involve raw pointers and unchecked memory access.
//! Users must uphold the documented invariants to avoid undefined behavior.
//! Public RAII handles (`Buff`, `Object`) are designed to reduce misuse,
//! but unsafe code boundaries still exist internally for performance reasons.
//!
//! ## Overflow & Arithmetic Behavior
//! Earlier versions required disabling overflow checks for performance.
//! **As of v0.2.0**, overflow handling has been corrected —
//! internal arithmetic now uses **saturating** or **wrapping** operations
//! where appropriate to maintain both **safety** and **branchless performance**.
//!
//! Users no longer need to disable overflow checks in debug or release builds.
pub mod buff_manager;
pub mod cache_padded;
