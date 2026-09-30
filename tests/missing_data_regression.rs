//! Regression: NearestNeighbor imputation used `unwrap_or((None?, ..))`, whose
//! eagerly evaluated `None?` made it return `None` for every input.

use rustygraph::{MissingDataStrategy, TimeSeries};

fn impute(values: Vec<Option<f64>>) -> Vec<Option<f64>> {
    let ts = (0..values.len()).map(|i| i as f64).collect();
    TimeSeries::new(ts, values)
        .unwrap()
        .handle_missing(MissingDataStrategy::NearestNeighbor)
        .unwrap()
        .values
}

#[test]
fn nearest_neighbor_fills_from_closest_side() {
    assert_eq!(
        impute(vec![Some(1.0), None, Some(3.0)]),
        vec![Some(1.0), Some(1.0), Some(3.0)]
    ); // tie -> previous
    assert_eq!(impute(vec![Some(1.0), None, None, Some(9.0)])[2], Some(9.0));
    assert_eq!(impute(vec![None, Some(2.0)])[0], Some(2.0));
    assert_eq!(impute(vec![Some(2.0), None])[1], Some(2.0));
}
