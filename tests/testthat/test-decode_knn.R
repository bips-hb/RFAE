library(testthat)
library(ranger)
library(arf)
set.seed(42)
trn <- sample(1:nrow(iris), 100)
tst <- setdiff(1:nrow(iris), trn)
arf <- adversarial_rf(iris[trn, ], num_trees = 20)

emap <- encode(arf, iris[trn, ], k = 2)
emb_tst <- predict(emap, arf, iris[tst, ])

test_that("decode_knn returns correct structure", {
  # This tests the eForest scheme (train_decoder)
  out <- decode_knn(arf, emap, emb_tst, k = 5, parallel = FALSE)

  expect_type(out, "list")
  expect_named(out, c("x_hat", "x_tilde"))
  expect_s3_class(out$x_hat, "data.frame")
  expect_equal(nrow(out$x_hat), nrow(iris[tst, ]))
  # no class label
  expect_equal(ncol(out$x_hat), ncol(iris[trn, ]))
})

test_that("decode_knn handles k=1 (nearest neighbor only)", {
  out <- decode_knn(arf, emap, emb_tst, k = 1, parallel = FALSE)
  expect_equal(nrow(out$x_hat), nrow(iris[tst, ]))
  # With k=1, the output should contain valid factor levels from the training set
  expect_true(all(out$x_hat$Species %in% levels(iris$Species)))
})


