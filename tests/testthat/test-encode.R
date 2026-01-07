library(testthat)
library(ranger)
set.seed(42)
trn <- sample(1:nrow(iris), 100)
tst <- setdiff(1:nrow(iris), trn)
rf <- ranger(Species ~ ., data = iris[trn, ], num.trees=50)

test_that("encode produces correct output structure", {
  k_val <- 3L
  emap <- encode(rf, iris[trn, ], k = k_val)

  expect_s3_class(emap, "encode")
  expect_equal(ncol(emap$Z), k_val)
  expect_equal(nrow(emap$Z), nrow(iris[trn, ]))
  expect_length(emap$lambda, k_val)
  expect_equal(dim(emap$V), c(nrow(iris[trn, ]), k_val))

  # Check if A is a square sparse matrix
  expect_s4_class(emap$A, "dgCMatrix")
  expect_equal(nrow(emap$A), ncol(emap$A))
})

test_that("encode handles categorical and numeric metadata correctly", {
  emap <- encode(rf, iris[trn, ], k = 2)
  meta <- emap$meta$metadata
  # encode filters out the class label automatically thanks to ranger handling
  expect_false(meta[variable == "Sepal.Length", fctr])
  expect_true(is.null(emap$meta$levels))
  expect_false("Species" %in% emap$meta$levels$variable)
})

test_that("predict.encode works on new data", {
  emap <- encode(rf, iris[trn, ], k = 2)

  # Ensure no error on out-of-sample data
  expect_no_error(
    emb_test <- predict(emap, rf, iris[tst, ])
  )

  expect_equal(nrow(emb_test), nrow(iris[tst, ]))
  expect_equal(ncol(emb_test), 2)
})
