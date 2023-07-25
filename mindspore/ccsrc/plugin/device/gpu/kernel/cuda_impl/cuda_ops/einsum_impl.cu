/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime.h>
#include "einsum_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
template <typename T>
__global__ void Diagonal(const size_t out_size, const T *input, const size_t *inp_shape, const size_t shape_size,
                         const size_t left_dim, const size_t right_dim, T *output) {
  size_t out_stride;
  size_t in_stride;
  size_t out_pos;
  size_t in_pos;
  size_t dim_val = inp_shape[right_dim];
  size_t pos_arr[EINSUM_MAX_DIMENSION];

  // for example 4-D: pos = posArray[0] * input_shape[1] * input_shape[2] * input_shape[3] +
  //                        posArray[1] * input_shape[2] * input_shape[3] +
  //                        posArray[2] * input_shape[3] +
  //                        posArray[3]
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_size; pos += blockDim.x * gridDim.x) {
    out_pos = pos;
    out_stride = out_size / inp_shape[0];
    in_stride = out_stride * dim_val;
    pos_arr[0] = out_pos / out_stride;
    in_pos = pos_arr[0] * in_stride;
    size_t idx_dim = 1;
    for (idx_dim = 1; idx_dim < right_dim; idx_dim++) {
      out_pos -= pos_arr[idx_dim - 1] * out_stride;
      out_stride = out_stride / inp_shape[idx_dim];
      pos_arr[idx_dim] = out_pos / out_stride;
      in_stride = out_stride * dim_val;
      in_pos += pos_arr[idx_dim] * in_stride;
    }
    out_pos -= pos_arr[right_dim - 1] * out_stride;
    out_stride = out_stride;
    pos_arr[right_dim] = 0;
    in_pos += pos_arr[left_dim] * out_stride;
    for (idx_dim = right_dim + 1; idx_dim < shape_size; idx_dim++) {
      out_pos -= pos_arr[idx_dim - 1] * out_stride;
      out_stride = out_stride / inp_shape[idx_dim];
      pos_arr[idx_dim] = out_pos / out_stride;
      in_pos += pos_arr[idx_dim] * out_stride;
    }
    output[pos] = input[in_pos];
  }
}
template <typename T>
cudaError_t CalDiagonal(const size_t size, const T *input, const size_t *input_shape, const size_t shape_size,
                        const size_t left_dim, const size_t right_dim, T *output, cudaStream_t cuda_stream) {
  Diagonal<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, input_shape, shape_size, left_dim, right_dim,
                                                              output);
  return GetCudaStatus();
}
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<double>(const size_t size, const double *input,
                                                         const size_t *input_shape, const size_t shape_size,
                                                         const size_t left_dim, const size_t right_dim, double *output,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<float>(const size_t size, const float *input,
                                                        const size_t *input_shape, const size_t shape_size,
                                                        const size_t left_dim, const size_t right_dim, float *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonal<half>(const size_t size, const half *input, const size_t *input_shape,
                                                       const size_t shape_size, const size_t left_dim,
                                                       const size_t right_dim, half *output, cudaStream_t cuda_stream);
template <typename T>
__global__ void DiagonalGrad(const size_t d_size, const T *dout, const size_t *inp_shape, const size_t shape_size,
                             const size_t left_dim, const size_t right_dim, T *d_inp) {
  size_t out_stride;
  size_t in_stride;
  size_t out_pos;
  size_t in_pos;
  size_t dim_val = inp_shape[right_dim];
  size_t pos_arr[EINSUM_MAX_DIMENSION];

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d_size; pos += blockDim.x * gridDim.x) {
    out_pos = pos;
    out_stride = d_size / inp_shape[0];
    in_stride = out_stride * dim_val;
    pos_arr[0] = out_pos / out_stride;
    in_pos = pos_arr[0] * in_stride;
    size_t idx_dim = 1;
    for (idx_dim = 1; idx_dim < right_dim; idx_dim++) {
      out_pos -= pos_arr[idx_dim - 1] * out_stride;
      out_stride = out_stride / inp_shape[idx_dim];
      pos_arr[idx_dim] = out_pos / out_stride;
      in_stride = out_stride * dim_val;
      in_pos += pos_arr[idx_dim] * in_stride;
    }
    out_pos -= pos_arr[right_dim - 1] * out_stride;
    out_stride = out_stride;
    pos_arr[right_dim] = 0;
    in_pos += pos_arr[left_dim] * out_stride;
    for (idx_dim = right_dim + 1; idx_dim < shape_size; idx_dim++) {
      out_pos -= pos_arr[idx_dim - 1] * out_stride;
      out_stride = out_stride / inp_shape[idx_dim];
      pos_arr[idx_dim] = out_pos / out_stride;
      in_pos += pos_arr[idx_dim] * out_stride;
    }
    d_inp[in_pos] = dout[pos];
  }
}
template <typename T>
cudaError_t CalDiagonalGrad(const size_t d_size, const T *dout, const size_t *input_shape, const size_t shape_size,
                            const size_t left_dim, const size_t right_dim, T *d_inp, cudaStream_t cuda_stream) {
  DiagonalGrad<<<GET_BLOCKS(d_size), GET_THREADS, 0, cuda_stream>>>(d_size, dout, input_shape, shape_size, left_dim,
                                                                    right_dim, d_inp);
  return GetCudaStatus();
}
template CUDA_LIB_EXPORT cudaError_t CalDiagonalGrad<double>(const size_t size, const double *dout,
                                                             const size_t *input_shape, const size_t shape_size,
                                                             const size_t left_dim, const size_t right_dim,
                                                             double *d_inp, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonalGrad<float>(const size_t size, const float *dout,
                                                            const size_t *input_shape, const size_t shape_size,
                                                            const size_t left_dim, const size_t right_dim, float *d_inp,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDiagonalGrad<half>(const size_t size, const half *dout,
                                                           const size_t *input_shape, const size_t shape_size,
                                                           const size_t left_dim, const size_t right_dim, half *d_inp,
                                                           cudaStream_t cuda_stream);
template <typename T>
__global__ void ReduceSum(const size_t out_size, const T *input, T *output, const size_t *out_shape,
                          const size_t shape_size, const size_t reduce_dim, const size_t dim_val) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_size; pos += blockDim.x * gridDim.x) {
    size_t cur_pos = pos;
    size_t stride;
    size_t out_pos_arr[EINSUM_MAX_DIMENSION];
    stride = out_size / out_shape[0];
    out_pos_arr[0] = cur_pos / stride;
    for (int cur_dim = 1; cur_dim < shape_size; ++cur_dim) {
      cur_pos -= out_pos_arr[cur_dim - 1] * stride;
      stride /= out_shape[cur_dim];
      out_pos_arr[cur_dim] = cur_pos / stride;
    }
    double sum_val = 0;
    for (int idx = 0; idx < dim_val; ++idx) {
      cur_pos = 0;
      stride = 1;
      for (int cur_dim = shape_size - 1; cur_dim > reduce_dim; --cur_dim) {
        cur_pos += out_pos_arr[cur_dim] * stride;
        stride *= out_shape[cur_dim];
      }
      cur_pos += idx * stride;
      stride *= dim_val;
      for (int cur_dim = reduce_dim - 1; cur_dim >= 0; --cur_dim) {
        cur_pos += out_pos_arr[cur_dim] * stride;
        stride *= out_shape[cur_dim];
      }
      sum_val += static_cast<double>(input[cur_pos]);
    }
    output[pos] = static_cast<T>(sum_val);
  }
}

template <>
__global__ void ReduceSum(const size_t out_size, const half *input, half *output, const size_t *out_shape,
                          const size_t shape_size, const size_t reduce_dim, const size_t dim_val) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_size; pos += blockDim.x * gridDim.x) {
    size_t cur_pos = pos;
    size_t stride;
    size_t out_pos_arr[EINSUM_MAX_DIMENSION];
    stride = out_size / out_shape[0];
    out_pos_arr[0] = cur_pos / stride;
    for (int cur_dim = 1; cur_dim < shape_size; ++cur_dim) {
      cur_pos -= out_pos_arr[cur_dim - 1] * stride;
      stride /= out_shape[cur_dim];
      out_pos_arr[cur_dim] = cur_pos / stride;
    }
    float sum_val = 0;
    for (int idx = 0; idx < dim_val; ++idx) {
      cur_pos = 0;
      stride = 1;
      for (int cur_dim = shape_size - 1; cur_dim > reduce_dim; --cur_dim) {
        cur_pos += out_pos_arr[cur_dim] * stride;
        stride *= out_shape[cur_dim];
      }
      cur_pos += idx * stride;
      stride *= dim_val;
      for (int cur_dim = reduce_dim - 1; cur_dim >= 0; --cur_dim) {
        cur_pos += out_pos_arr[cur_dim] * stride;
        stride *= out_shape[cur_dim];
      }
      sum_val += __half2float(input[cur_pos]);
    }
    output[pos] = __float2half(sum_val);
  }
}

template <typename T>
cudaError_t CalReduceSum(const size_t out_size, const T *input, T *output, const size_t *out_shape,
                         const size_t shape_size, const size_t reduce_dim, const size_t dim_val,
                         cudaStream_t cuda_stream) {
  ReduceSum<<<GET_BLOCKS(out_size), GET_THREADS, 0, cuda_stream>>>(out_size, input, output, out_shape, shape_size,
                                                                   reduce_dim, dim_val);
  return GetCudaStatus();
}

template <typename T>
struct MulFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs * rhs); }
};

template <>
struct MulFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return __float2half(__half2float(lhs) * __half2float(rhs));
  }
};
template <typename T>
struct DotMulFunc {
  __device__ __host__ __forceinline__ double operator()(const T &lhs, const T &rhs) {
    return static_cast<double>(lhs * rhs);
  }
};

template <>
struct DotMulFunc<half> {
  __device__ __host__ __forceinline__ double operator()(const half &lhs, const half &rhs) {
    return static_cast<double>(__half2float(lhs) * __half2float(rhs));
  }
};

template <typename T>
struct CastFunc {
  __device__ __host__ __forceinline__ T operator()(const double data) { return static_cast<T>(data); }
};
template <>
struct CastFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const double data) {
    return __float2half(static_cast<float>(data));
  }
};

template <typename T>
__global__ void Dot(const size_t size, T *input_a, const T *input_b, double *output, T *out_t) {
  DynamicSharedMem<double> share_mem;
  double *s_data = share_mem.addr();
  double sum_val = 0;

  for (size_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos < size; pos += blockDim.x * gridDim.x) {
    sum_val += DotMulFunc<T>()(input_a[pos], input_b[pos]);
  }
  s_data[threadIdx.x] = sum_val;
  __syncthreads();

  int idx = blockDim.x / 2;
  while (idx != 0) {
    if (threadIdx.x < idx) {
      s_data[threadIdx.x] += s_data[threadIdx.x + idx];
    }
    __syncthreads();
    idx /= 2;
  }
  if (threadIdx.x == 0) {
    MsAtomicAdd(output, s_data[0]);
  }
  out_t[0] = CastFunc<T>()(output[0]);
}

template <typename T>
cudaError_t CalDot(const size_t size, T *input_a, const T *input_b, T *output, cudaStream_t cuda_stream) {
  int threads_num = GET_THREADS;
  int share_mem_size = GET_THREADS * sizeof(double);
  double *cur_out;
  cudaMalloc(&cur_out, sizeof(double));
  cudaMemset(cur_out, 0, sizeof(double) * 1);
  Dot<<<GET_BLOCKS(size), GET_THREADS, share_mem_size, cuda_stream>>>(size, input_a, input_b, cur_out, output);
  cudaFree(cur_out);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalDot<double>(const size_t size, double *input_a, const double *input_b,
                                                    double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDot<float>(const size_t size, float *input_a, const float *input_b,
                                                   float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDot<half>(const size_t size, half *input_a, const half *input_b, half *output,
                                                  cudaStream_t cuda_stream);

template <typename T>
__global__ void DotGrad(const size_t size, const T dout, T *mid_res, T *input_b, T *input_a) {
  for (size_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos < size; pos += blockDim.x * gridDim.x) {
    input_a[pos] = MulFunc<T>()(input_b[pos], dout);
    input_b[pos] = MulFunc<T>()(mid_res[pos], dout);
  }
}
template <typename T>
cudaError_t CalDotGrad(const size_t size, const T dout, T *mid_res, T *input_b, T *input_a, cudaStream_t cuda_stream) {
  DotGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dout, mid_res, input_b, input_a);
  return GetCudaStatus();
}
template CUDA_LIB_EXPORT cudaError_t CalDotGrad<double>(const size_t size, const double dout, double *mid_res,
                                                        double *input_b, double *input_a, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDotGrad<float>(const size_t size, const float dout, float *mid_res,
                                                       float *input_b, float *input_a, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDotGrad<half>(const size_t size, const half dout, half *mid_res, half *input_b,
                                                      half *input_a, cudaStream_t cuda_stream);
// Element-wise ArithMetic
template <typename T>
__global__ void ElewiseArithMulKernel(const size_t nums, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = MulFunc<T>()(x0[pos], x1[pos]);
  }
}
// Broadcast comparison
__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }
// Broadcast Arithmetic
template <typename T>
__global__ void BroadcastArithMulKernel(const size_t shape_len, const size_t *lft_shape, const size_t lft_num,
                                        const size_t *rht_shape, const size_t rht_num, const size_t *out_shape,
                                        const size_t out_num, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_num; pos += blockDim.x * gridDim.x) {
    size_t out_pos = pos;
    size_t lft_stride = lft_num / lft_shape[0];
    size_t rht_stride = rht_num / rht_shape[0];
    size_t out_stride = out_num / out_shape[0];
    size_t out_idx = out_pos / out_stride;
    size_t lft_pos = Index(out_idx, lft_shape[0]) * lft_stride;
    size_t rht_pos = Index(out_idx, rht_shape[0]) * rht_stride;
    for (size_t idx_dim = 1; idx_dim < shape_len; idx_dim++) {
      out_pos -= out_idx * out_stride;
      lft_stride /= lft_shape[idx_dim];
      rht_stride /= rht_shape[idx_dim];
      out_stride /= out_shape[idx_dim];
      out_idx = out_pos / out_stride;
      lft_pos += Index(out_idx, lft_shape[idx_dim]) * lft_stride;
      rht_pos += Index(out_idx, rht_shape[idx_dim]) * rht_stride;
    }
    y[pos] = MulFunc<T>()(x0[lft_pos], x1[rht_pos]);
  }
}
template <typename T>
cudaError_t CalMul(const bool broadcast_flag, const size_t shape_len, const size_t *lft_shape, const size_t lft_num,
                   const size_t *rht_shape, const size_t rht_num, const size_t *out_shape, const size_t out_num,
                   const T *x0, const T *x1, T *y, cudaStream_t stream) {
  if (broadcast_flag) {
    BroadcastArithMulKernel<<<GET_BLOCKS(out_num), GET_THREADS, 0, stream>>>(shape_len, lft_shape, lft_num, rht_shape,
                                                                             rht_num, out_shape, out_num, x0, x1, y);
  } else {
    ElewiseArithMulKernel<<<GET_BLOCKS(out_num), GET_THREADS, 0, stream>>>(out_num, x0, x1, y);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalMul<double>(const bool broadcast_flag, const size_t shape_len,
                                                    const size_t *lft_shape, const size_t lft_num,
                                                    const size_t *rht_shape, const size_t rht_num,
                                                    const size_t *out_shape, const size_t out_num, const double *x0,
                                                    const double *x1, double *y, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalMul<float>(const bool broadcast_flag, const size_t shape_len,
                                                   const size_t *lft_shape, const size_t lft_num,
                                                   const size_t *rht_shape, const size_t rht_num,
                                                   const size_t *out_shape, const size_t out_num, const float *x0,
                                                   const float *x1, float *y, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalMul<half>(const bool broadcast_flag, const size_t shape_len,
                                                  const size_t *lft_shape, const size_t lft_num,
                                                  const size_t *rht_shape, const size_t rht_num,
                                                  const size_t *out_shape, const size_t out_num, const half *x0,
                                                  const half *x1, half *y, cudaStream_t stream);
