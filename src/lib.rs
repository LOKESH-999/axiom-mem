//! # Low-Level Memory Utilities
//! This crate provides high-performance memory management utilities,
//! allowing management of `T`-typed memory with either pre-allocated
//! or dynamically growing (auto-resizing) pools.
//!
//! ## Features
//! - `allocator_api` — Enables the use of custom allocators with standard collections.
//!
//! ## Modules
//!
//! - `buff_manager` — Core memory pool management, including static and dynamic object pools.
//! - `cache_padded` — Provides cache-line padding to reduce false sharing or improve memory layout.
//!
//! ## Safety
//!
//! Many operations involve raw pointers and unchecked memory access. Users must adhere
//! to the documented invariants to avoid undefined behavior.
//! ## **Overflow Warning:**  
//! This crate is designed for high-performance memory management using **branchless operations**.  
//! Around **80%–90% of the critical core operations** are branchless.  
//! Most arithmetic operations do **not** perform overflow checks.  
//! In practice, overflow occurs in roughly **60% of usage cases**.  
//! There is **no safe way for users to prevent overflow** without rewriting significant portions of the code.  
//! This crate assumes that overflow checks are effectively **disabled** to achieve maximum speed.
pub mod buff_manager;
pub mod cache_padded;
