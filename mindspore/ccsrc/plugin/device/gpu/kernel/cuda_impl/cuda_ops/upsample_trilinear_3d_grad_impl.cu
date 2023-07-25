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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_trilinear_3d_grad_impl.cuh"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample.cuh"

__device__ __forceinline__ int idx_dhw(const int height, const int width, const int z, const int y, const int x) {
  return (z * height + y) * width + x;
}

template <typename T, typename S>
__device__ __forceinline__ S special_cast(T value) {
  return static_cast<S>(value);
}

template <>
__device__ __forceinline__ float special_cast(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ half special_cast(float value) {
  return __float2half(value);
}

template <typename T, typename S>
__global__ void UpsampleTrilinear3DGradKernel(const size_t elem_num, const T *grad, const int batchsize,
                                              const int channels, const int grad_d, const int grad_h, const int grad_w,
                                              const int grad_dhw, const int dinput_d, const int dinput_h,
                                              const int dinput_w, const int dinput_dhw, const S d_scale,
                                              const S h_scale, const S w_scale, const bool align_corner, T *dinput) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < grad_dhw; pos += blockDim.x * gridDim.x) {
    const int t2 = pos / (grad_h * grad_w);
    const int h2 = pos / grad_w % grad_h;
    const int w2 = pos % grad_w;

    const S t1r = area_pixel_compute_source_index<S>(d_scale, t2, align_corner, false);
    const int t1 = floorf(t1r);
    const int t1p = (t1 < (dinput_d - 1)) ? 1 : 0;
    const S t1lambda = t1r - t1;
    const S t0lambda = static_cast<S>(1) - t1lambda;

    const S h1r = area_pixel_compute_source_index<S>(h_scale, h2, align_corner, false);
    const int h1 = floorf(h1r);
    const int h1p = (h1 < (dinput_h - 1)) ? 1 : 0;
    const S h1lambda = h1r - h1;
    const S h0lambda = static_cast<S>(1) - h1lambda;

    const S w1r = area_pixel_compute_source_index<S>(w_scale, w2, align_corner, false);
    const int w1 = floorf(w1r);
    const int w1p = (w1 < (dinput_w - 1)) ? 1 : 0;
    const S w1lambda = w1r - w1;
    const S w0lambda = static_cast<S>(1) - w1lambda;

    size_t dinput_offset = 0;
    size_t dout_offset = 0;
    for (int n = 0; n < batchsize; ++n) {
      for (int c = 0; c < channels; ++c) {
        const S d2val = special_cast<T, S>(grad[dout_offset + (t2 * grad_h + h2) * grad_w + w2]);

        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1, w1), elem_num,
                      special_cast<S, T>(t0lambda * h0lambda * w0lambda * d2val));
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1, w1 + w1p), elem_num,
                      special_cast<S, T>(t0lambda * h0lambda * w1lambda * d2val));
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1 + h1p, w1), elem_num,
                      special_cast<S, T>(t0lambda * h1lambda * w0lambda * d2val));
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1, h1 + h1p, w1 + w1p), elem_num,
                      special_cast<S, T>(t0lambda * h1lambda * w1lambda * d2val));
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1, w1), elem_num,
                      special_cast<S, T>(t1lambda * h0lambda * w0lambda * d2val));
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1, w1 + w1p), elem_num,
                      special_cast<S, T>(t1lambda * h0lambda * w1lambda * d2val));
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1 + h1p, w1), elem_num,
                      special_cast<S, T>(t1lambda * h1lambda * w0lambda * d2val));
        FastAtomicAdd(dinput, dinput_offset + idx_dhw(dinput_h, dinput_w, t1 + t1p, h1 + h1p, w1 + w1p), elem_num,
                      special_cast<S, T>(t1lambda * h1lambda * w1lambda * d2val));

        dout_offset += grad_dhw;
        dinput_offset += dinput_dhw;
      }
    }
  }
  return;
}

template <typename T, typename S>
cudaError_t CalUpsampleTrilinear3DGrad(const T *grad, const int n, const int c, const int grad_d, const int grad_h,
                                       const int grad_w, const int dinput_d, const int dinput_h, const int dinput_w,
                                       const S d_scale, const S h_scale, const S w_scale, const bool align_corner,
                                       T *dinput, const uint32_t device_id, cudaStream_t cuda_stream) {
  const int dinput_dhw = dinput_d * dinput_h * dinput_w;
  const int grad_dhw = grad_d * grad_h * grad_w;
  const int dinput_size = dinput_dhw * n * c;
  if (dinput_d == grad_d && dinput_h == grad_h && dinput_w == grad_w) {
    CudaMemcpyDeviceToDevice<T, T>
      <<<CUDA_BLOCKS(device_id, dinput_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(dinput_size, grad, dinput);
  } else {
    (void)cudaMemset(dinput, 0, sizeof(T) * dinput_size);
    const int blockSize = std::min(CUDA_THREADS(device_id), 256);
    const int gridSize = (grad_dhw + blockSize - 1) / blockSize;
    UpsampleTrilinear3DGradKernel<T, S><<<gridSize, blockSize, 0, cuda_stream>>>(
      dinput_size, grad, n, c, grad_d, grad_h, grad_w, grad_dhw, dinput_d, dinput_h, dinput_w, dinput_dhw, d_scale,
      h_scale, w_scale, align_corner, dinput);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUpsampleTrilinear3DGrad<half, float>(
  const half *grad, const int n, const int c, const int grad_d, const int grad_h, const int grad_w, const int dinput_d,
  const int dinput_h, const int dinput_w, const float d_scale, const float h_scale, const float w_scale,
  const bool align_corner, half *dinput, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleTrilinear3DGrad<float, float>(
  const float *grad, const int n, const int c, const int grad_d, const int grad_h, const int grad_w, const int dinput_d,
  const int dinput_h, const int dinput_w, const float d_scale, const float h_scale, const float w_scale,
  const bool align_corner, float *dinput, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleTrilinear3DGrad<double, double>(
  const double *grad, const int n, const int c, const int grad_d, const int grad_h, const int grad_w,
  const int dinput_d, const int dinput_h, const int dinput_w, const double d_scale, const double h_scale,
  const double w_scale, const bool align_corner, double *dinput, const uint32_t device_id, cudaStream_t cuda_stream);
