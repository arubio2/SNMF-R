# Function to handle potential negative values very close to zero, setting them to zero
# Note: This function is not currently called in NMFKLMixing, but could be used for numerical stability.
setNegativeZero <- function (x) {
  x[x < 0 & x > -1e-10] <- 0 # Set values between -1e-10 and 0 to 0
  return(x)
}

# Function to control and validate the dimensions of input matrices for NMF
controlDimensionNMF <- function (W, H, V, k) {
  if (ncol(W) != k) {
    stop("W must have 'k' columns. 'k' represents the number of latent components.")
  }
  if (nrow(H) != k) {
    stop("H must have 'k' rows. 'k' represents the number of latent components.")
  }
  if (ncol(V) != ncol(H)) {
    stop("V (original matrix) and H (coefficient matrix) must have the same number of columns.")
  }
  if (nrow(V) != nrow(W)) {
    stop("V (original matrix) and W (basis matrix) must have the same number of rows.")
  }
}

safe_pmax <- function(X, eps) {
  objectClass <- class(X)[[1]]

  if (objectClass == "gpu.matrix.torch" || objectClass == "gpu.matrix.tensorflow") {
    # max(X, eps) with arithmetic only: on GPUmatrix the elementwise comparison
    # (X < eps) is ~10x slower than abs() (e.g. 84 ms vs 2 ms for 2000 x 4000).
    return((X + eps + abs(X - eps)) / 2)
  }

  pmax(X, eps)
}