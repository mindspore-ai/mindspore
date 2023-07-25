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
 * WITposh WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_trilinear_3d_impl.cuh"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void UpsampleTrilinear3DKernel(const int num_kernels, const T *input, T *output, const int batch_size,
                                          const int channel, const int in_d, const int in_h, const int in_w,
                                          const int out_d, const int out_h, const int out_w, const S d_scale,
                                          const S h_scale, const S w_scale, const bool align_corners, const int in_dhw,
                                          const int out_hw, const int out_dhw) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_kernels; pos += blockDim.x * gridDim.x) {
    const int w2 = (pos % out_hw) % out_w;
    const int h2 = (pos % out_hw) / out_w;
    const int d2 = pos / out_hw;
    // calculate scaled values for input index
    const S t1r = area_pixel_compute_source_index<S>(d_scale, d2, align_corners, false);
    const int t1 = t1r;
    const int t1p = (t1 < in_d - 1) ? t1 + 1 : t1;
    const S lambda_d1 = t1r - t1;
    const S lambda_d0 = static_cast<S>(1) - lambda_d1;
    //
    const S h1r = area_pixel_compute_source_index<S>(h_scale, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < in_h - 1) ? h1 + 1 : h1;
    const S lambda_h1 = h1r - h1;
    const S lambda_h0 = static_cast<S>(1) - lambda_h1;
    //
    const S w1r = area_pixel_compute_source_index<S>(w_scale, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < in_w - 1) ? w1 + 1 : w1;
    const S lambda_w1 = w1r - w1;
    const S lambda_w0 = static_cast<S>(1) - lambda_w1;
    //
    auto in_data = input;
    auto out_data = output;
    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < channel; ++c) {
        const S val = lambda_d0 * (lambda_h0 * (lambda_w0 * static_cast<S>(in_data[(t1 * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<S>(in_data[(t1 * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<S>(in_data[(t1 * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<S>(in_data[(t1 * in_h + h1p) * in_w + w1p]))) +
                      lambda_d1 * (lambda_h0 * (lambda_w0 * static_cast<S>(in_data[(t1p * in_h + h1) * in_w + w1]) +
                                                lambda_w1 * static_cast<S>(in_data[(t1p * in_h + h1) * in_w + w1p])) +
                                   lambda_h1 * (lambda_w0 * static_cast<S>(in_data[(t1p * in_h + h1p) * in_w + w1]) +
                                                lambda_w1 * static_cast<S>(in_data[(t1p * in_h + h1p) * in_w + w1p])));
        out_data[(d2 * out_h + h2) * out_w + w2] = static_cast<T>(val);
        in_data += in_dhw;
        out_data += out_dhw;
      }
    }
  }
  return;
}

template <typename T, typename S>
cudaError_t CalUpsampleTrilinear3D(const T *input, const int n, const int c, const int in_d, const int in_h,
                                   const int in_w, const int out_d, const int out_h, const int out_w, const S d_scale,
                                   const S h_scale, const S w_scale, const bool align_corners, T *output,
                                   const uint32_t device_id, cudaStream_t cuda_stream) {
  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    const int num_kernels = out_d * out_h * out_w * n * c;
    CudaMemcpyDeviceToDevice<T, T>
      <<<CUDA_BLOCKS(device_id, num_kernels), CUDA_THREADS(device_id), 0, cuda_stream>>>(num_kernels, input, output);
  } else {
    const int in_dhw = in_d * in_h * in_w;
    const int out_hw = out_h * out_w;
    const int out_dhw = out_d * out_hw;
    const int num_kernels = out_dhw;
    const int blockSize = std::min(CUDA_THREADS(device_id), 512);
    const int gridSize = (num_kernels + blockSize - 1) / blockSize;
    UpsampleTrilinear3DKernel<T, S>
      <<<gridSize, blockSize, 0, cuda_stream>>>(num_kernels, input, output, n, c, in_d, in_h, in_w, out_d, out_h, out_w,
                                                d_scale, h_scale, w_scale, align_corners, in_dhw, out_hw, out_dhw);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUpsampleTrilinear3D<half, float>(
  const half *input, const int n, const int c, const int in_d, const int in_h, const int in_w, const int out_d,
  const int out_h, const int out_w, const float d_scale, const float h_scale, const float w_scale,
  const bool align_corners, half *output, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleTrilinear3D<float, float>(
  const float *input, const int n, const int c, const int in_d, const int in_h, const int in_w, const int out_d,
  const int out_h, const int out_w, const float d_scale, const float h_scale, const float w_scale,
  const bool align_corners, float *output, const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleTrilinear3D<double, double>(
  const double *input, const int n, const int c, const int in_d, const int in_h, const int in_w, const int out_d,
  const int out_h, const int out_w, const double d_scale, const double h_scale, const double w_scale,
  const bool align_corners, double *output, const uint32_t device_id, cudaStream_t cuda_stream);
