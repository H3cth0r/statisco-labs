use pyo3::prelude::*;

#[pyfunction]
fn get_current_time() -> PyResult<String> {
    Python::with_gil(|py| {
        let time_module = py.import_bound("time")?;
        let format_string = "%Y-%m-%d %H:%M:%S";
        let args = (format_string,);
        
        let result: String = time_module
            .call_method1("strftime", args)?
            .extract()?;
        
        Ok(result)
    })
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyrust_alloc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_current_time, m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

