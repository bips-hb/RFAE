# Preprocess input data

This function prepares input data.

## Usage

``` r
prep_x(x, to_numeric = NULL, to_factor = NULL, default = 5)
```

## Arguments

- x:

  Input `data.frame`.

- to_numeric:

  List of variables to force as numeric.

- to_factor:

  List of variables to force as factor.

- default:

  Threshold to classify a variable as numeric (more than default unique
  values) or factor (less or equal to unique values).

## Value

Preprocessed `data.frame`.
