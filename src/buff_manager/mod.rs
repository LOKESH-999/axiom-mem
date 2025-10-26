//! # Memory Pool Modules
//!
//! This crate provides low-level memory pool and object management utilities
//! with both static and dynamic strategies. It is designed for high-performance
//! allocation, reuse, and safe/unsafe raw access patterns.
//!
//! ## Modules
//!
//! - `dynamic_free_idx_map` — Dynamic bitmap for tracking free slots in pools.
//! - `static_array_buffer` — Static buffer pool manager for raw memory arrays.
//! - `static_free_idx_map` — Bitmap-based free index tracking for static pools.
//! - `static_object_buffer` — Static object pool manager with ergonomic handles.
//!
//! ## Re-exports
//!
//! - [`BufferPoolManager`] — from `static_array_buffer`, manages fixed-size memory buffers.
//! - [`ObjectPoolManager`] — from `static_object_buffer`, manages initialized objects in a pool with automatic slot recycling.
//!
//! ## Safety
//!
//! Many of the low-level operations allow raw pointer access and unchecked
//! operations. Users must follow the safety contracts to avoid undefined behavior.
pub mod dynamic_free_idx_map;
pub mod static_array_buffer;
pub mod static_free_idx_map;
pub mod static_object_buffer;

pub use static_array_buffer::BufferPoolManager;
pub use static_object_buffer::ObjectPoolManager;
