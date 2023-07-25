/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_bilinear_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;                       // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename T>
__global__ void ResizeBilinear(const T *input, const int n, const int c, const int input_h, const int input_w,
                               const int output_h, const int output_w, const int nchw, const int chw, const int hw,
                               const float h_scale, const float w_scale, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / output_w % output_h;
    const int posw = pos % output_w;
    const float posw_scaled = w_scale * posw;
    const float posh_scaled = h_scale * posh;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0);            // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), input_w - 1);  // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0);            // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), input_h - 1);  // NOLINT
    const float w_alpha = posw_scaled - w_low;
    const float w_beta = 1.0f - w_alpha;
    const float h_alpha = posh_scaled - h_low;
    const float h_beta = 1.0f - h_alpha;
    const int input_start = input_h * input_w * (posn * c + posc);
    const T p1 = input[input_start + (h_low * input_w) + w_low];
    const T p2 = input[input_start + (h_low * input_w) + w_high];
    const T p3 = input[input_start + (h_high * input_w) + w_low];
    const T p4 = input[input_start + (h_high * input_w) + w_high];
    output[pos] = (p1 * static_cast<T>(h_beta * w_beta)) + (p2 * static_cast<T>(h_beta * w_alpha)) +
                  (p3 * static_cast<T>(h_alpha * w_beta)) + (p4 * static_cast<T>(h_alpha * w_alpha));
  }
  return;
}

template <typename T>
__global__ void ResizeBilinear_HPC(const T *input, const int n, const int c, const int input_h, const int input_w,
                                   const int output_h, const int output_w, const int nchw, const int chw, const int hw,
                                   const float h_scale, const float w_scale, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / output_w % output_h;
    const int posw = pos % output_w;
    const float posw_scaled = (static_cast<float>(posw) + 0.5f) * w_scale - 0.5f;
    const float posh_scaled = (static_cast<float>(posh) + 0.5f) * h_scale - 0.5f;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0);            // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), input_w - 1);  // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0);            // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), input_h - 1);  // NOLINT
    const float w_alpha = posw_scaled - floorf(posw_scaled);
    const float w_beta = 1.0f - w_alpha;
    const float h_alpha = posh_scaled - floorf(posh_scaled);
    const float h_beta = 1.0f - h_alpha;
    const int input_start = input_h * input_w * (posn * c + posc);
    const T p1 = input[input_start + (h_low * input_w) + w_low];
    const T p2 = input[input_start + (h_low * input_w) + w_high];
    const T p3 = input[input_start + (h_high * input_w) + w_low];
    const T p4 = input[input_start + (h_high * input_w) + w_high];
    output[pos] = (p1 * static_cast<T>(h_beta * w_beta)) + (p2 * static_cast<T>(h_beta * w_alpha)) +
                  (p3 * static_cast<T>(h_alpha * w_beta)) + (p4 * static_cast<T>(h_alpha * w_alpha));
  }
  return;
}

// fp16 path
__global__ void ResizeBilinearGradHalf(const half *input, const int n, const int c, const int input_h,
                                       const int input_w, const int output_h, const int output_w, const int nchw,
                                       const int chw, const int hw, const float h_scale, const float w_scale,
                                       half *output, float *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / input_w % input_h;
    const int posw = pos % input_w;
    const float posw_scaled = w_scale * posw;
    const float posh_scaled = h_scale * posh;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0);             // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), output_w - 1);  // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0);             // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), output_h - 1);  // NOLINT
    const float w_alpha = posw_scaled - w_low;
    const float w_beta = 1.0f - w_alpha;
    const float h_alpha = posh_scaled - h_low;
    const float h_beta = 1.0f - h_alpha;
    const float grad = static_cast<float>(input[pos]);
    const float dp1 = h_beta * w_beta * grad;
    const float dp2 = h_beta * w_alpha * grad;
    const float dp3 = h_alpha * w_beta * grad;
    const float dp4 = h_alpha * w_alpha * grad;
    const int output_start = output_h * output_w * (posn * c + posc);
    atomicAdd(&interim[output_start + (h_low * output_w) + w_low], dp1);
    atomicAdd(&interim[output_start + (h_low * output_w) + w_high], dp2);
    atomicAdd(&interim[output_start + (h_high * output_w) + w_low], dp3);
    atomicAdd(&interim[output_start + (h_high * output_w) + w_high], dp4);
  }
  return;
}

template <typename T>
__global__ void ResizeBilinearGrad(const T *input, const int n, const int c, const int input_h, const int input_w,
                                   const int output_h, const int output_w, const int nchw, const int chw, const int hw,
                                   const float h_scale, const float w_scale, T *output, T *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / input_w % input_h;
    const int posw = pos % input_w;
    const T posw_scaled = w_scale * posw;
    const T posh_scaled = h_scale * posh;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0);             // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), output_w - 1);  // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0);             // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), output_h - 1);  // NOLINT
    const T w_alpha = posw_scaled - w_low;
    const T w_beta = 1.0f - w_alpha;
    const T h_alpha = posh_scaled - h_low;
    const T h_beta = 1.0f - h_alpha;
    const T grad = input[pos];
    T dp1 = h_beta * w_beta * grad;
    T dp2 = h_beta * w_alpha * grad;
    T dp3 = h_alpha * w_beta * grad;
    T dp4 = h_alpha * w_alpha * grad;
    const int output_start = output_h * output_w * (posn * c + posc);
    atomicAdd(&output[output_start + (h_low * output_w) + w_low], dp1);
    atomicAdd(&output[output_start + (h_low * output_w) + w_high], dp2);
    atomicAdd(&output[output_start + (h_high * output_w) + w_low], dp3);
    atomicAdd(&output[output_start + (h_high * output_w) + w_high], dp4);
  }
  return;
}

// fp16 path
__global__ void ResizeBilinearGradHalf_HPC(const half *input, const int n, const int c, const int input_h,
                                           const int input_w, const int output_h, const int output_w, const int nchw,
                                           const int chw, const int hw, const float h_scale, const float w_scale,
                                           half *output, float *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / input_w % input_h;
    const int posw = pos % input_w;
    const float posw_scaled = (static_cast<float>(posw) + 0.5f) * w_scale - 0.5f;
    const float posh_scaled = (static_cast<float>(posh) + 0.5f) * h_scale - 0.5f;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0);             // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), output_w - 1);  // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0);             // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), output_h - 1);  // NOLINT
    const float w_alpha = posw_scaled - floorf(posw_scaled);
    const float w_beta = 1.0f - w_alpha;
    const float h_alpha = posh_scaled - floorf(posh_scaled);
    const float h_beta = 1.0f - h_alpha;
    const float grad = static_cast<float>(input[pos]);
    const float dp1 = h_beta * w_beta * grad;
    const float dp2 = h_beta * w_alpha * grad;
    const float dp3 = h_alpha * w_beta * grad;
    const float dp4 = h_alpha * w_alpha * grad;
    const int output_start = output_h * output_w * (posn * c + posc);
    atomicAdd(&interim[output_start + (h_low * output_w) + w_low], dp1);
    atomicAdd(&interim[output_start + (h_low * output_w) + w_high], dp2);
    atomicAdd(&interim[output_start + (h_high * output_w) + w_low], dp3);
    atomicAdd(&interim[output_start + (h_high * output_w) + w_high], dp4);
  }
  return;
}

template <typename T>
__global__ void ResizeBilinearGrad_HPC(const T *input, const int n, const int c, const int input_h, const int input_w,
                                       const int output_h, const int output_w, const int nchw, const int chw,
                                       const int hw, const float h_scale, const float w_scale, T *output, T *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    const int posn = pos / chw;
    const int posc = pos / hw % c;
    const int posh = pos / input_w % input_h;
    const int posw = pos % input_w;
    const T posw_scaled = (static_cast<T>(posw) + 0.5f) * w_scale - 0.5f;
    const T posh_scaled = (static_cast<T>(posh) + 0.5f) * h_scale - 0.5f;
    const int w_low = max(static_cast<int>(floorf(posw_scaled)), 0);             // NOLINT
    const int w_high = min(static_cast<int>(ceilf(posw_scaled)), output_w - 1);  // NOLINT
    const int h_low = max(static_cast<int>(floorf(posh_scaled)), 0);             // NOLINT
    const int h_high = min(static_cast<int>(ceilf(posh_scaled)), output_h - 1);  // NOLINT
    const T w_alpha = posw_scaled - floorf(posw_scaled);
    const T w_beta = 1.0f - w_alpha;
    const T h_alpha = posh_scaled - floorf(posh_scaled);
    const T h_beta = 1.0f - h_alpha;
    const T grad = input[pos];
    T dp1 = h_beta * w_beta * grad;
    T dp2 = h_beta * w_alpha * grad;
    T dp3 = h_alpha * w_beta * grad;
    T dp4 = h_alpha * w_alpha * grad;
    const int output_start = output_h * output_w * (posn * c + posc);
    atomicAdd(&output[output_start + (h_low * output_w) + w_low], dp1);
    atomicAdd(&output[output_start + (h_low * output_w) + w_high], dp2);
    atomicAdd(&output[output_start + (h_high * output_w) + w_low], dp3);
    atomicAdd(&output[output_start + (h_high * output_w) + w_high], dp4);
  }
  return;
}

__global__ void ResizeBilinearGradPost(const int nchw, half *output, float *interim) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nchw; pos += blockDim.x * gridDim.x) {
    output[pos] = __float2half(interim[pos]);
  }
  return;
}

template <typename T>
cudaError_t CalResizeBilinear(const T *input, const int n, const int c, const int input_h, const int input_w,
                              const int output_h, const int output_w, const float h_scale, const float w_scale,
                              const bool half_pixel_centers, T *output, const uint32_t &device_id,
                              cudaStream_t cuda_stream) {
  const int nchw = n * c * output_h * output_w;
  const int chw = c * output_h * output_w;
  const int hw = output_h * output_w;
  if (half_pixel_centers) {
    ResizeBilinear_HPC<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output);
  } else {
    ResizeBilinear<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output);
  }
  return GetCudaStatus();
}

cudaError_t CalResizeBilinearGradHalf(const half *input, const int n, const int c, const int input_h, const int input_w,
                                      const int output_h, const int output_w, const float h_scale, const float w_scale,
                                      const bool half_pixel_centers, half *output, float *interim,
                                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int hw = input_h * input_w;
  const int chw = c * hw;
  const int nchw = n * chw;
  const int output_num = n * c * output_h * output_w;
  if (half_pixel_centers) {
    ResizeBilinearGradHalf_HPC<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output, interim);
  } else {
    ResizeBilinearGradHalf<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output, interim);
  }
  ResizeBilinearGradPost<<<CUDA_BLOCKS(device_id, output_num), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    output_num, output, interim);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalResizeBilinearGrad(const T *input, const int n, const int c, const int input_h, const int input_w,
                                  const int output_h, const int output_w, const float h_scale, const float w_scale,
                                  const bool half_pixel_centers, T *output, T *interim, const uint32_t &device_id,
                                  cudaStream_t cuda_stream) {
  const int hw = input_h * input_w;
  const int chw = c * hw;
  const int nchw = n * chw;
  if (half_pixel_centers) {
    ResizeBilinearGrad_HPC<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output, interim);
  } else {
    ResizeBilinearGrad<<<CUDA_BLOCKS(device_id, nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, n, c, input_h, input_w, output_h, output_w, nchw, chw, hw, h_scale, w_scale, output, interim);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalResizeBilinear<float>(const float *input, const int n, const int c,
                                                              const int input_h, const int input_w, const int output_h,
                                                              const int output_w, const float h_scale,
                                                              const float w_scale, const bool half_pixel_centers,
                                                              float *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeBilinear<half>(const half *input, const int n, const int c,
                                                             const int input_h, const int input_w, const int output_h,
                                                             const int output_w, const float h_scale,
                                                             const float w_scale, const bool half_pixel_centers,
                                                             half *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeBilinear<double>(const double *input, const int n, const int c,
                                                               const int input_h, const int input_w, const int output_h,
                                                               const int output_w, const float h_scale,
                                                               const float w_scale, const bool half_pixel_centers,
                                                               double *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeBilinearGrad<float>(
  const float *input, const int n, const int c, const int input_h, const int input_w, const int output_h,
  const int output_w, const float h_scale, const float w_scale, const bool half_pixel_centers, float *output,
  float *interim, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalResizeBilinearGrad<double>(
  const double *input, const int n, const int c, const int input_h, const int input_w, const int output_h,
  const int output_w, const float h_scale, const float w_scale, const bool half_pixel_centers, double *output,
  double *interim, const uint32_t &device_id, cudaStream_t cuda_stream);
