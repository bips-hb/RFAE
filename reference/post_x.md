# Post-process data

This function prepares output data.

## Usage

``` r
post_x(x, meta, round = TRUE)
```

## Arguments

- x:

  Input `data.frame`.

- meta:

  Metadata.

- round:

  Round continuous variables to their respective maximum precision in
  the real data set?

## Value

A `data.frame` which follows the structure and ordering of the input
dataset.
