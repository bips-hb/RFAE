library(testthat)
library(data.table)

test_that("prep_x correctly converts types based on thresholds", {
  df <- data.frame(
    char = rep(c("a", "b"), length.out = 10),
    int_low = rep(c(1L, 2L), length.out = 10),
    int_high = 1:10,
    stringsAsFactors = FALSE
  )

  # Silence the specific threshold warnings
  res <- suppressWarnings(prep_x(df, default = 5))

  # Check conversions
  expect_s3_class(res$x$char, "factor")
  expect_s3_class(res$x$int_low, "ordered")  # < 6 unique -> ordered factor
  expect_true(is.numeric(res$x$int_high))    # > 5 unique -> numeric

  # Check return list structure
  expect_named(res, c("x", "to_numeric", "to_factor"))
})

test_that("post_x restores types and classes correctly", {
  # Mock metadata structure your function expects
  meta <- list(
    input_class = "data.table",
    levels = data.table(
      variable = c("cat", "cat"),
      val = c("low", "high"),
      number = 1:2
    ),
    metadata = data.table(
      variable = c("cat", "num"),
      class = c("factor", "numeric"),
      decimals = c(0, 2)
    )
  )

  # Input needs to match columns in metadata
  input <- data.table(num = c(1.1234, 2.888), cat = c(1, 2))
  res <- post_x(input, meta, round = TRUE)
  expect_equal(res$num[1], 1.12)
  expect_s3_class(res$cat, "factor")
  expect_equal(levels(res$cat), c("low", "high"))
})

