#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC);
  
  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  // ==========================================================================================
  // FIX 1: Define a correct TiledMMA
  // ------------------------------------------------------------------------------------------
  // The original TiledMMA definition was incorrect. It failed to properly map the
  // computation to the 32 threads of a warp, causing the data loading to fail,
  // which left registers full of zeros and produced a zero-filled output.
  //
  // The definition below creates a 64x64x16 warp-level MMA (Matrix Multiply-Accumulate)
  // operation. It does this by tiling the base 16x8x16 hardware MMA instruction
  // (mma_atom) 4 times in the M dimension and 8 times in the N dimension.
  // This forms a 4x8 grid of MMA atoms, which can be perfectly handled by the
  // 32 threads (4x8) of a warp.
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_atom = MMA_Atom<mma_op>;

  // Map 4x8=32 threads to a 4x8 grid
  using ThrLayout = Layout<Shape<_4,_8>>;
  // Tile the 16x8x16 mma_atom into a 4x8 grid
  using AtomLayoutMN = Layout<Shape<_4,_8>>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, AtomLayoutMN{}, ThrLayout{}));

  // The Tile sizes must match or be a multiple of our defined MMA size.
  // Our MMA computes a C tile of shape (4*16, 8*8) = (64, 64)
  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 32; // The slicing size for K can be set independently

  // ==========================================================================================


  //using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  //using mma_traits = MMA_Traits<mma_op>;
  //using mma_atom = MMA_Atom<mma_traits>;

  //using MMA = decltype(make_tiled_mma(mma_atom{}, 
  //                    make_layout(Shape<_2, _2, _1>{}), 
  //                    make_layout(Shape<_1, _2, _1>{})));
  //constexpr int kTileM = 128; 
  //constexpr int kTileN = 128; 
  //constexpr int kTileK = 32; 

  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  for (int i = 0; i < 100; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // --- Result Comparison ---
  T *Cptr_host = (T*)malloc(sizeof(T) * m * n);
  T *Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  // ==========================================================================================
  // FIX 2: Correct the comparison logic
  // ------------------------------------------------------------------------------------------
  // The original comparison was incorrect because it assumed identical memory layouts.
  // In reality, our kernel's result C(m,n) is the transpose of the cuBLAS result.
  // Therefore, we need to use 2D indexing to logically compare the element C(i, j)
  // from both matrices.
  float threshold = 0.1;
  int error_count = 0;
  for (int i = 0; i < m; ++i) { // iterate over rows
    for (int j = 0; j < n; ++j) { // iterate over columns
      // Cptr_host is a row-major (m, n) matrix. C(i, j) is at index i * n + j
      float v1 = static_cast<float>(Cptr_host[i * n + j]);

      // Cptr_cublas_host stores the result of B^T*A and has shape (n,m) but
      // was written with a leading dimension (ldc) of n. So the element
      // corresponding to C(i,j) is also at index i * n + j.
      float v2 = static_cast<float>(Cptr_cublas_host[i * n + j]);

      if (fabs(v2 - v1) > threshold) {
        if(error_count < 10) { // Only print the first 10 errors
            printf("Mismatch at (%d, %d): kernel_val (v1) = %f, cublas_val (v2) = %f, diff = %f\n",
                   i, j, v1, v2, fabs(v2-v1));
        }
        error_count++;
      }
    }
  }

  if (error_count == 0) {
      printf("\nSUCCESS: Results match with cuBLAS!\n");
  } else {
      printf("\nFAILURE: Found %d mismatches.\n", error_count);
  }
  // ==========================================================================================

  // Free memory
  free(Aptr_host);
  free(Bptr_host);
  free(Cptr_host);
  free(Cptr_cublas_host);
  cudaFree(Aptr);
  cudaFree(Bptr);
  cudaFree(Cptr);
  cudaFree(Cptr_cublas);
  cublasDestroy(handle);

  return 0;
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
