test_that("error function calculates metrics correctly", {
  X <- data.frame(
    A = c(1, 2, 3, 4, 5),
    B = c(1, 2, 3, 4, 5),
    C = factor(c("a", "a", "b", "b", "c"))
  )

  Xhat <- data.frame(
    A = c(1, 2, 3, 4, 5),      # Perfect R^2 = 1
    B = c(1, 2, 3, 4, 10),
    C = c("a", "b", "b", "a", "c") # 3/5 matches = 0.6 accuracy
  )

  result <- reconstruction_error(Xhat, X)
  expect_type(result, "list")
  expect_named(result, c("num_error", "cat_error", "num_avg", "cat_avg", "ovr_error"))

  expect_equal(result$num_error$A, 0)

  expect_equal(result$cat_error$C, 0.4)

  # Check overall averages are numeric
  expect_true(is.numeric(result$num_avg))
  expect_true(is.numeric(result$cat_avg))
})

test_that("error function handles single-type datasets", {
  # Test with only numeric data
  X_num <- iris[, 1:4]
  res_num <- reconstruction_error(X_num, X_num)

  expect_equal(res_num$cat_avg, "No variables")
  expect_equal(res_num$num_avg, 0)

  # Test with only categorical data
  X_cat <- data.frame(v1 = c("A", "B"), v2 = c("C", "D"))
  res_cat <- reconstruction_error(X_cat, X_cat)

  expect_equal(res_cat$num_avg, "No variables")
  expect_equal(res_cat$cat_avg, 0)
})

