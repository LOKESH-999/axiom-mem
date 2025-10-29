//! # Memory Pool Modules
//!
//! High-performance memory management primitives for reusable buffer and object pools.
//!
//! This crate offers both **static** and **dynamic** allocation strategies designed
//! for systems where low-latency and predictable memory access are critical.
//!
//! ## Modules
//!
//! - [`dynamic_free_idx_map_v1`] — dynamic bitmap for tracking free slots in resizable pools.
//! - [`static_array_buffer`] — fixed-size buffer pool manager for raw memory arrays.
//! - [`static_free_idx_map`] — compact bitmap-based free index tracking for static pools.
//! - [`static_object_buffer`] — ergonomic object pool with automatic slot recycling.
//!
//! ## Re-exports
//!
//! - [`BufferPoolManager`] — from [`static_array_buffer`]; manages fixed-size raw buffers.
//! - [`ObjectPoolManager`] — from [`static_object_buffer`]; manages typed objects with automatic reuse.
//!
//! ## Safety
//!
//! Many of the low-level operations allow raw pointer arithmetic and unchecked access.
//! These APIs are safe when used according to their documented contracts but
//! may cause undefined behavior if misused.
pub mod dynamic_free_idx_map_v1;
pub mod static_array_buffer;
pub mod static_free_idx_map;
pub mod static_object_buffer;

pub use static_array_buffer::BufferPoolManager;
pub use static_object_buffer::ObjectPoolManager;
