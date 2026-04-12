//! ============================================================
//! 🧩 Buffer Manager — Module Overview
//! ============================================================
//!
//! This module groups all buffer-pool implementations used in
//! `axiom-mem`. Buffers are organized based on whether their
//! capacity is **fixed** or **growable**:
//!
//! - [`static_buff`]
//!     Non-growable buffers. Capacity is fixed at initialization.
//!     These pools are optimized for predictable memory usage,
//!     stable latency, and cache-friendly access patterns.
//!
//! - [`dynamic_buff`]
//!     Growable buffers. Capacity can increase at runtime when
//!     allocation demand grows. Useful for bursty or unbounded
//!     workloads where flexibility is required.
//!
//! # Re-exports
//! The most common static buffer managers are re-exported here for
//! convenience:
//!
//! - [`BufferPoolManager`] — Manages fixed-size array buffers  
//! - [`ObjectPoolManager`] — Manages fixed-size object pools
//!
//! These re-exports allow direct usage via:
//!
//! ```rust
//! use axiom_mem::buff_manager::BufferPoolManager;
//! ```
//!
//! ============================================================

pub mod dynamic_buff;
pub mod static_buff;

pub use static_buff::static_array_buffer::BufferPoolManager;
pub use static_buff::static_object_buffer::ObjectPoolManager;
