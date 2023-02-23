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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/affine_grid_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

#ifndef CHECK_CUDNN_AFFINE_GRID
#define CHECK_CUDNN_AFFINE_GRID(call, message)                                                                     \
  {                                                                                                                \
    cudnnStatus_t status = (call);                                                                                 \
    if (status != CUDNN_STATUS_SUCCESS) {                                                                          \
      fprintf(stderr, "Got CUDNN error %d %s at %s:%d\n. For AffineGrid, %s", status, cudnnGetErrorString(status), \
              __FILE__, __LINE__, message);                                                                        \
      return status;                                                                                               \
    }                                                                                                              \
  }
#endif  // CHECK_CUDNN_AFFINE_GRID

template <typename T>
struct Point2 {
  Point2() = default;
  T x;
  T y;
  __device__ __forceinline__ Point2<T> operator+(const Point2<T> &p) { return {this->x + p.x, this->y + p.y}; }
};

template <typename T>
struct Point3 {
  Point3() = default;
  T x;
  T y;
  T z;
  __device__ __forceinline__ Point3<T> operator+(const Point3<T> &p) {
    return {this->x + p.x, this->y + p.y, this->z + p.z};
  }
};

template <typename T>
struct Theta2 {
  Theta2() = default;
  T r00;
  T r01;
  T t0;
  T r10;
  T r11;
  T t1;
};

template <typename T>
struct Theta3 {
  Theta3() = default;
  T r00;
  T r01;
  T r02;
  T t0;
  T r10;
  T r11;
  T r12;
  T t1;
  T r20;
  T r21;
  T r22;
  T t2;
};

template <typename T>
__device__ T linspace(const int32_t &step, const int32_t &n_steps, const bool &align_corners) {
  if (n_steps <= 1) {
    return 0;  // if n_steps is less than 1, just return 0.
  }
  if (align_corners) {
    return static_cast<T>(2 * step - n_steps + 1) / static_cast<T>(n_steps - 1);
  } else {
    return static_cast<T>(2 * step - n_steps + 1) / static_cast<T>(n_steps);
  }
}

template <>  // template specialization for half type.
__device__ half linspace(const int32_t &step, const int32_t &n_steps, const bool &align_corners) {
  if (n_steps <= 1) {
    return {0};  // if n_steps is less than 1, just return 0.
  }
  if (align_corners) {
    return __int2half_rz(2 * step - n_steps + 1) / __int2half_rz(n_steps - 1);
  } else {
    return __int2half_rz(2 * step - n_steps + 1) / __int2half_rz(n_steps);
  }
}

template <typename T>
__global__ void ScaleOneRowOfTheta2Kernel(const T *theta_ptr, T *scaled_theta_ptr, const T alpha_x, const T alpha_y,
                                          const size_t len) {
  Point3<T> row{};
  auto *theta_ptr_casted = (Point3<T> *)theta_ptr;
  auto *scaled_theta_ptr_casted = (Point3<T> *)scaled_theta_ptr;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
    row = theta_ptr_casted[i];
    scaled_theta_ptr_casted[i] = {row.x * alpha_x, row.y * alpha_y, row.z};
  }
}

template <typename T>
cudnnDataType_t Type2CudnnType() {
  if (std::is_same<T, half>::value) {
    return CUDNN_DATA_HALF;
  }
  if (std::is_same<T, float>::value) {
    return CUDNN_DATA_FLOAT;
  }
  if (std::is_same<T, double>::value) {
    return CUDNN_DATA_DOUBLE;
  }
  return CUDNN_DATA_FLOAT;
}

template <typename T>
cudnnStatus_t CalculateAffineGrid4D(const T *theta_ptr, T *workspace_ptr, T *grid_ptr, const int32_t &N,
                                    const int32_t &C, const int32_t &H, const int32_t &W, const bool &align_corners,
                                    const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudnnHandle_t cudnn_handle{};
  cudnnSpatialTransformerDescriptor_t st_desc{};
  CHECK_CUDNN_AFFINE_GRID(cudnnCreate(&cudnn_handle), "Create handle failed.");
  CHECK_CUDNN_AFFINE_GRID(cudnnCreateSpatialTransformerDescriptor(&st_desc), "Create descriptor failed.");

  CHECK_CUDNN_AFFINE_GRID(cudnnSetStream(cudnn_handle, cuda_stream), "Set stream failed.");
  int image_size[4] = {N, C, H, W};
  CHECK_CUDNN_AFFINE_GRID(
    cudnnSetSpatialTransformerNdDescriptor(st_desc, CUDNN_SAMPLER_BILINEAR, Type2CudnnType<T>(), 4, image_size),
    "cudnnSetSpatialTransformerNdDescriptor failed.");
  if (align_corners) {
    CHECK_CUDNN_AFFINE_GRID(cudnnSpatialTfGridGeneratorForward(cudnn_handle, st_desc, theta_ptr, grid_ptr),
                            "cudnnSpatialTfGridGeneratorForward failed.");
  } else {
    T alpha_x = static_cast<float>(W - 1) / static_cast<float>(W);
    T alpha_y = static_cast<float>(H - 1) / static_cast<float>(H);
    T *scaled_theta_ptr = workspace_ptr;
    size_t num_rows_of_thetas = N * 2;
    ScaleOneRowOfTheta2Kernel<<<CUDA_BLOCKS(device_id, num_rows_of_thetas), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      theta_ptr, scaled_theta_ptr, alpha_x, alpha_y, num_rows_of_thetas);
    CHECK_CUDNN_AFFINE_GRID(cudnnSpatialTfGridGeneratorForward(cudnn_handle, st_desc, scaled_theta_ptr, grid_ptr),
                            "cudnnSpatialTfGridGeneratorForward failed.");
  }

  CHECK_CUDNN_AFFINE_GRID(cudnnDestroySpatialTransformerDescriptor(st_desc), "Destroy descriptor failed.");
  CHECK_CUDNN_AFFINE_GRID(cudnnDestroy(cudnn_handle), "Destroy handle failed.");
  return CUDNN_STATUS_SUCCESS;
}

template <typename T>
__global__ void CalculateSparseBaseGrid5DKernel(T *base_grid_ptr, const size_t len_base_grid, const int32_t D,
                                                const int32_t H, const int32_t W, const bool align_corners) {
  size_t step, n_steps;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len_base_grid; i += blockDim.x * gridDim.x) {
    if (i < W) {
      step = i;
      n_steps = W;
    } else if (i < W + H) {
      step = i - W;
      n_steps = H;
    } else {
      step = i - W - H;
      n_steps = D;
    }
    base_grid_ptr[i] = linspace<T>(step, n_steps, align_corners);
  }
}

template <typename T>
__global__ void CalculateSparseWrappedGrid5DKernel(const T *theta_ptr, const T *base_grid_ptr, T *wrapped_grid_ptr,
                                                   const size_t len_wrapped_grid, const int32_t N, const int32_t D,
                                                   const int32_t H, const int32_t W) {
  size_t n, ii;
  Theta3<T> theta{};
  T point{}, x_wrapped{}, y_wrapped{}, z_wrapped{};
  for (size_t oi = blockIdx.x * blockDim.x + threadIdx.x; oi < len_wrapped_grid; oi += blockDim.x * gridDim.x) {
    if (oi < N * W) {  // wrap x
      n = oi / W;
      ii = oi % W;
      theta = ((Theta3<T> *)theta_ptr)[n];
      point = base_grid_ptr[ii];
      x_wrapped = point * theta.r00 + theta.t0;
      y_wrapped = point * theta.r10 + theta.t1;
      z_wrapped = point * theta.r20 + theta.t2;
    } else if (oi < N * W + N * H) {  // wrap y
      n = (oi - N * W) / H;
      ii = ((oi - N * W) % H) + W;
      theta = ((Theta3<T> *)theta_ptr)[n];
      point = base_grid_ptr[ii];
      x_wrapped = point * theta.r01;
      y_wrapped = point * theta.r11;
      z_wrapped = point * theta.r21;
    } else {  // wrap z
      n = (oi - N * W - N * H) / D;
      ii = ((oi - N * W - N * H) % D) + W + H;
      theta = ((Theta3<T> *)theta_ptr)[n];
      point = base_grid_ptr[ii];
      x_wrapped = point * theta.r02;
      y_wrapped = point * theta.r12;
      z_wrapped = point * theta.r22;
    }
    ((Point3<T> *)wrapped_grid_ptr)[oi] = {x_wrapped, y_wrapped, z_wrapped};
  }
}

template <typename T>
__global__ void CalculateAffineGrid5DKernel(const T *xs_ptr, const T *ys_ptr, const T *zs_ptr, T *grid_ptr,
                                            const size_t grid_elements, const int32_t D, const int32_t H,
                                            const int32_t W) {
  size_t mul_H_W = H * W, mul_D_H_W = D * H * W;
  size_t n, d, h, w, idx_x, idx_y, idx_z;
  Point3<T> x{}, y{}, z{};
  auto *xs_ptr_casted = (Point3<T> *)xs_ptr;
  auto *ys_ptr_casted = (Point3<T> *)ys_ptr;
  auto *zs_ptr_casted = (Point3<T> *)zs_ptr;
  auto *grid_ptr_casted = (Point3<T> *)grid_ptr;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < grid_elements; i += blockDim.x * gridDim.x) {
    n = i / mul_D_H_W;
    d = i % mul_D_H_W / mul_H_W;
    h = i % mul_D_H_W % mul_H_W / W;
    w = i % mul_D_H_W % mul_H_W % W;
    idx_x = n * W + w;
    idx_y = n * H + h;
    idx_z = n * D + d;
    x = xs_ptr_casted[idx_x];
    y = ys_ptr_casted[idx_y];
    z = zs_ptr_casted[idx_z];
    grid_ptr_casted[i] = {x.x + y.x + z.x, x.y + y.y + z.y, x.z + y.z + z.z};
  }
}

template <typename T>
cudaError_t CalculateAffineGrid5D(const T *theta_ptr, T *workspace_ptr, T *grid_ptr, const int32_t &N, const int32_t &C,
                                  const int32_t &D, const int32_t &H, const int32_t &W, const bool &align_corners,
                                  const uint32_t &device_id, cudaStream_t cuda_stream) {
  // Theta: (N×3×4), Grid: (N×D×H×W×3)
  // step 1: linspace to get x(W) & y(H) & z(D)
  // step 2: wrap with theta to get x_wrapped(N, W, 3) & y_wrapped(N, H, 3) & z_wrapped(N, D, 3)
  // step 3: add to get grid(N, D, H, W, 3)

  T *base_grid_ptr = workspace_ptr;
  size_t len_base_grid = W + H + D;
  CalculateSparseBaseGrid5DKernel<<<CUDA_BLOCKS(device_id, len_base_grid), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    base_grid_ptr, len_base_grid, D, H, W, align_corners);

  T *wrapped_grid_ptr = workspace_ptr + len_base_grid;
  size_t len_wrapped_grid = N * W + N * H + N * D;
  CalculateSparseWrappedGrid5DKernel<<<CUDA_BLOCKS(device_id, len_wrapped_grid), CUDA_THREADS(device_id), 0,
                                       cuda_stream>>>(theta_ptr, base_grid_ptr, wrapped_grid_ptr, len_wrapped_grid, N,
                                                      D, H, W);

  T *xs_ptr = wrapped_grid_ptr;
  T *ys_ptr = wrapped_grid_ptr + (N * W) * 3;
  T *zs_ptr = wrapped_grid_ptr + (N * W + N * H) * 3;
  size_t grid_elements = N * D * H * W;
  CalculateAffineGrid5DKernel<<<CUDA_BLOCKS(device_id, grid_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    xs_ptr, ys_ptr, zs_ptr, grid_ptr, grid_elements, D, H, W);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudnnStatus_t CalculateAffineGrid4D<half>(const half *theta_ptr, half *workspace_ptr,
                                                                   half *grid_ptr, const int32_t &N, const int32_t &C,
                                                                   const int32_t &H, const int32_t &W,
                                                                   const bool &align_corners, const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudnnStatus_t CalculateAffineGrid4D<float>(
  const float *theta_ptr, float *workspace_ptr, float *grid_ptr, const int32_t &N, const int32_t &C, const int32_t &H,
  const int32_t &W, const bool &align_corners, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudnnStatus_t CalculateAffineGrid4D<double>(
  const double *theta_ptr, double *workspace_ptr, double *grid_ptr, const int32_t &N, const int32_t &C,
  const int32_t &H, const int32_t &W, const bool &align_corners, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalculateAffineGrid5D<half>(const half *theta_ptr, half *workspace_ptr,
                                                                 half *grid_ptr, const int32_t &N, const int32_t &C,
                                                                 const int32_t &D, const int32_t &H, const int32_t &W,
                                                                 const bool &align_corners, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalculateAffineGrid5D<float>(const float *theta_ptr, float *workspace_ptr,
                                                                  float *grid_ptr, const int32_t &N, const int32_t &C,
                                                                  const int32_t &D, const int32_t &H, const int32_t &W,
                                                                  const bool &align_corners, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalculateAffineGrid5D<double>(const double *theta_ptr, double *workspace_ptr,
                                                                   double *grid_ptr, const int32_t &N, const int32_t &C,
                                                                   const int32_t &D, const int32_t &H, const int32_t &W,
                                                                   const bool &align_corners, const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);
