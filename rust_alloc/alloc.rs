use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyMemoryView};
use pyo3::exceptions::PyRuntimeError;
use std::collections::HashMap;
use std::os::raw::c_void;
use std::ptr;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

#[pyclass]
#[derive(Clone, Hash, Eq, PartialEq)]
struct BufferOptions {
    #[pyo3(get, set)]
    image: Option<String>,  // Simplified from ImageDType
    #[pyo3(get, set)]
    uncached: bool,
    #[pyo3(get, set)]
    cpu_access: bool,
    #[pyo3(get, set)]
    host: bool,
    #[pyo3(get, set)]
    nolru: bool,
    #[pyo3(get, set)]
    external_ptr: Option<usize>,
}

#[pymethods]
impl BufferOptions {
    #[new]
    fn new() -> Self {
        BufferOptions {
            image: None,
            uncached: false,
            cpu_access: false,
            host: false,
            nolru: false,
            external_ptr: None,
        }
    }
}

type CacheKey = (usize, u64); // size and options hash
type CacheValue = Vec<*mut c_void>;

#[pyclass]
struct LRUAllocator {
    cache: Mutex<HashMap<CacheKey, CacheValue>>,
}

#[pymethods]
impl LRUAllocator {
    #[new]
    fn new() -> Self {
        LRUAllocator {
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn alloc(&self, size: usize, options: Option<BufferOptions>) -> PyResult<*mut c_void> {
        let options = options.unwrap_or_else(BufferOptions::new);
        let mut hasher = DefaultHasher::new();
        options.hash(&mut hasher);
        let key = (size, hasher.finish());

        // Try to get from cache first
        if let Some(cached) = self.cache.lock().unwrap().get_mut(&key) {
            if let Some(ptr) = cached.pop() {
                return Ok(ptr);
            }
        }

        // Allocate new buffer if cache is empty
        self.allocate_new(size, &options)
    }

    fn free(&self, ptr: *mut c_void, size: usize, options: Option<BufferOptions>) -> PyResult<()> {
        let options = options.unwrap_or_else(BufferOptions::new);
        
        if !options.nolru {
            let mut hasher = DefaultHasher::new();
            options.hash(&mut hasher);
            let key = (size, hasher.finish());
            
            self.cache.lock().unwrap()
                .entry(key)
                .or_insert_with(Vec::new)
                .push(ptr);
        } else {
            unsafe {
                libc::free(ptr);
            }
        }
        Ok(())
    }

    fn free_cache(&self) -> PyResult<()> {
        let mut cache = self.cache.lock().unwrap();
        for (_, ptrs) in cache.iter() {
            for &ptr in ptrs {
                unsafe {
                    libc::free(ptr);
                }
            }
        }
        cache.clear();
        Ok(())
    }

    fn copyin(&self, dest: *mut c_void, src: &PyMemoryView) -> PyResult<()> {
        let src_bytes = src.get_bytes()?;
        unsafe {
            ptr::copy_nonoverlapping(
                src_bytes.as_ptr() as *const c_void,
                dest,
                src_bytes.len()
            );
        }
        Ok(())
    }

    fn copyout(&self, py: Python, dest: &PyMemoryView, src: *const c_void) -> PyResult<()> {
        let dest_bytes = dest.get_bytes()?;
        unsafe {
            ptr::copy_nonoverlapping(
                src,
                dest_bytes.as_ptr() as *mut c_void,
                dest_bytes.len()
            );
        }
        Ok(())
    }
}

impl LRUAllocator {
    fn allocate_new(&self, size: usize, options: &BufferOptions) -> PyResult<*mut c_void> {
        if let Some(external_ptr) = options.external_ptr {
            Ok(external_ptr as *mut c_void)
        } else {
            let ptr = unsafe { libc::malloc(size) };
            if ptr.is_null() {
                // Try to free cache and retry allocation
                self.free_cache()?;
                let ptr = unsafe { libc::malloc(size) };
                if ptr.is_null() {
                    return Err(PyRuntimeError::new_err("Memory allocation failed"));
                }
                Ok(ptr)
            } else {
                Ok(ptr)
            }
        }
    }
}

#[pymodule]
fn rust_allocator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BufferOptions>()?;
    m.add_class::<LRUAllocator>()?;
    Ok(())
}
