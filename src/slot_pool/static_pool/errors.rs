//! Error types for the fixed pool system.
//!
//! ## Overview
//!
//! Defines [`PoolError`] and [`PoolResult`] used across:
//! - pool operations
//! - pointer validation
//! - slot lifecycle management
//!
//! ## Design
//!
//! Errors are:
//! - lightweight (`Copy`, no allocation)
//! - deterministic (no dynamic context)
//! - aligned with core invariants of the pool
//!
//! ## Scope
//!
//! Covers:
//! - pointer correctness (bounds, alignment)
//! - allocation state (double free)
//! - underlying allocator errors ([`FreeIdxManager`])
//!
//! This module does NOT handle:
//! - synchronization errors
//! - cross-process safety (mmap users must enforce this externally)
use core::fmt;
use std::error::Error;

use crate::free_idx_map::FreeMapError;

/// Error type for operations on [`FixedPool`].
///
/// ## Overview
///
/// This enum represents all failure conditions that can occur while:
/// - validating pointers
/// - managing slot lifecycle
/// - interacting with the underlying [`FreeIdxManager`]
///
/// ## Design
///
/// Errors are:
/// - lightweight (`Copy`)
/// - deterministic (no allocation)
/// - tightly coupled to pool invariants
///
/// ## Variants
///
/// - [`FreeMap`] → propagated errors from the index allocator
/// - [`PointerOutOfBounds`] → pointer not within pool memory
/// - [`PointerMisaligned`] → pointer not aligned to `T`
/// - [`SlotAlreadyFree`] → double free attempt
/// - [`ZeroSizedTypeUnsupported`] → `T` must have non-zero size
///
/// ## Usage
///
/// Used with [`PoolResult`] for all fallible pool operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolError {
    /// Wrapper around the inner free-index allocator errors.
    FreeMap(FreeMapError),

    /// The pointer is not inside the pool range.
    PointerOutOfBounds,

    /// The pointer is inside the range but not aligned to `T`.
    PointerMisaligned,

    /// The slot is already free, so dropping + retiring it would be a double free.
    SlotAlreadyFree,

    /// Zero-sized types are not supported by this pool layout.
    ZeroSizedTypeUnsupported,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolError::FreeMap(e) => write!(f, "{e}"),
            PoolError::PointerOutOfBounds => write!(f, "pointer is out of pool bounds"),
            PoolError::PointerMisaligned => write!(f, "pointer is not aligned to T"),
            PoolError::SlotAlreadyFree => write!(f, "slot is already free"),
            PoolError::ZeroSizedTypeUnsupported => {
                write!(f, "zero-sized types are not supported")
            }
        }
    }
}

impl Error for PoolError {}

/// Converts [`FreeMapError`] into [`PoolError::FreeMap`].
impl From<FreeMapError> for PoolError {
    fn from(value: FreeMapError) -> Self {
        PoolError::FreeMap(value)
    }
}

/// Result type for pool operations.
pub type PoolResult<T> = Result<T, PoolError>;
