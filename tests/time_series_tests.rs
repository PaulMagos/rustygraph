// Integration tests for TimeSeries

use rustygraph::time_series::{TimeSeries, TimeSeriesError};

#[test]
fn test_from_raw() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(series.len(), 3);
    assert!(!series.is_empty());
}

#[test]
fn test_new_with_timestamps() {
    let result = TimeSeries::new(
        vec![0.0, 1.0, 2.0],
        vec![Some(1.0), Some(2.0), Some(3.0)]
    );
    assert!(result.is_ok());
    let series = result.unwrap();
    assert_eq!(series.len(), 3);
}

#[test]
fn test_new_length_mismatch() {
    let result = TimeSeries::new(
        vec![0.0, 1.0],
        vec![Some(1.0), Some(2.0), Some(3.0)]
    );
    assert!(result.is_err());
    match result {
        Err(TimeSeriesError::LengthMismatch { timestamps, values }) => {
            assert_eq!(timestamps, 2);
            assert_eq!(values, 3);
        }
        _ => panic!("Expected LengthMismatch error"),
    }
}

#[test]
fn test_new_empty() {
    let result = TimeSeries::<f64>::new(vec![], vec![]);
    assert!(result.is_err());
    match result {
        Err(TimeSeriesError::EmptyData) => {},
        _ => panic!("Expected EmptyData error"),
    }
}

#[test]
fn test_len_and_is_empty() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0]).unwrap();
    assert_eq!(series.len(), 2);
    assert!(!series.is_empty());
}

#[test]
fn test_with_missing_values() {
    let result = TimeSeries::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![Some(1.0), None, Some(3.0), Some(4.0)]
    );
    assert!(result.is_ok());
    let series = result.unwrap();
    assert_eq!(series.len(), 4);
}

#[test]
fn test_all_missing_values() {
    let result = TimeSeries::<f64>::new(
        vec![0.0, 1.0, 2.0],
        vec![None, None, None]
    );
    assert!(result.is_ok());
    let series = result.unwrap();
    assert_eq!(series.len(), 3);
}

#[test]
fn test_auto_generated_timestamps() {
    let series = TimeSeries::from_raw(vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(series.timestamps, vec![0.0, 1.0, 2.0]);
}

