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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/resize_linear_1d.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

namespace {
struct AlignCornersFunc {
  __device__ void operator()(const float &new_x, const int &old_length, const int &new_length, float *old_x) const {
    *old_x = new_length != 1 ? new_x * (old_length - 1) / (new_length - 1) : 0;
  }
};

struct AsymmetricFunc {
  __device__ void operator()(const float &new_x, const int &old_length, const int &new_length, float *old_x) const {
    *old_x = new_length != 0 ? new_x * old_length / new_length : 0;
  }
};

struct HalfPixelFunc {
  __device__ void operator()(const float &new_x, const int &old_length, const int &new_length, float *old_x) const {
    *old_x = new_length != 0 ? (new_x + 0.5) * old_length / new_length - 0.5 : 0;
  }
};
}  // namespace

template <typename T, typename TransformationT>
__global__ void ResizeLinear1DKernel(const int64_t output_size, const int64_t in_width, const int64_t out_width,
                                     const T *input, float *output, const TransformationT func) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= output_size) {
    return;
  }

  int64_t out_i = index % out_width;
  int64_t out_index = index / out_width;
  float in_i;
  func(out_i, in_width, out_width, &in_i);
  const float in_floor = std::floor(in_i);
  const float in_ceil = std::ceil(in_i);
  size_t in_lower = static_cast<size_t>(in_floor > 0 ? in_floor : 0);
  size_t in_upper = static_cast<size_t>(in_ceil < static_cast<float>(in_width - 1) ? in_ceil : in_width - 1);
  float lerp = in_i - in_floor;

  const float left(static_cast<float>(*(input + out_index * in_width + in_lower)));
  const float right(static_cast<float>(*(input + out_index * in_width + in_upper)));
  *(output + out_index * out_width + out_i) = (left + (right - left) * lerp);

  return;
}

template <typename T>
void ResizeLinear1D(const enum ResizeLinearCoordinateTransformationMode mode, const int64_t output_size,
                    const int64_t in_width, const int64_t out_width, const T *input, float *output,
                    const uint32_t device_id, cudaStream_t stream) {
  switch (mode) {
    case ALIGN_CORNERS:
      return ResizeLinear1DKernel<T><<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        output_size, in_width, out_width, input, output, AlignCornersFunc());
    case HALF_PIXEL:
      return ResizeLinear1DKernel<T><<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        output_size, in_width, out_width, input, output, HalfPixelFunc());
    case ASYMMETRIC:
      return ResizeLinear1DKernel<T><<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        output_size, in_width, out_width, input, output, AsymmetricFunc());
    default:
      break;
  }
}

template <typename T, typename TransformationT>
__global__ void ResizeLinear1DGradKernel(const int64_t output_size, const int64_t in_width, const int64_t out_width,
                                         const float *grad_output, T *grad_input, const TransformationT func) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= output_size) {
    return;
  }

  int64_t out_i = index % out_width;
  int64_t out_index = index / out_width;
  float in_i;
  func(out_i, in_width, out_width, &in_i);
  const float in_floor = std::floor(in_i);
  const float in_ceil = std::ceil(in_i);
  size_t in_lower = static_cast<size_t>(in_floor > 0 ? in_floor : 0);
  size_t in_upper = static_cast<size_t>(in_ceil < static_cast<float>(in_width - 1) ? in_ceil : in_width - 1);
  float lerp = in_i - in_floor;

  (void)MsAtomicAdd<T>(grad_input + out_index * in_width + in_lower,
                       static_cast<T>((*(grad_output + out_index * out_width + out_i)) * (1 - lerp)));

  (void)MsAtomicAdd<T>(grad_input + out_index * in_width + in_upper,
                       static_cast<T>((*(grad_output + out_index * out_width + out_i)) * lerp));
  return;
}

template <typename T>
void ResizeLinear1DGrad(const enum ResizeLinearCoordinateTransformationMode mode, const int64_t output_size,
                        const int64_t in_width, const int64_t out_width, const float *grad_output, T *grad_input,
                        const uint32_t device_id, cudaStream_t stream) {
  switch (mode) {
    case ALIGN_CORNERS:
      return ResizeLinear1DGradKernel<T><<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        output_size, in_width, out_width, grad_output, grad_input, AlignCornersFunc());
    case HALF_PIXEL:
      return ResizeLinear1DGradKernel<T><<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        output_size, in_width, out_width, grad_output, grad_input, HalfPixelFunc());
    case ASYMMETRIC:
      return ResizeLinear1DGradKernel<T><<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        output_size, in_width, out_width, grad_output, grad_input, AsymmetricFunc());
    default:
      break;
  }
}

#define RESIZE_LINEAR_1D_FUNC(T)                                                                                 \
  template CUDA_LIB_EXPORT void ResizeLinear1D(                                                                  \
    const enum ResizeLinearCoordinateTransformationMode mode, const int64_t output_size, const int64_t in_width, \
    const int64_t out_width, const T *input, float *output, const uint32_t device_id, cudaStream_t stream);

RESIZE_LINEAR_1D_FUNC(int8_t)
RESIZE_LINEAR_1D_FUNC(uint8_t)
RESIZE_LINEAR_1D_FUNC(int16_t)
RESIZE_LINEAR_1D_FUNC(uint16_t)
RESIZE_LINEAR_1D_FUNC(int)
RESIZE_LINEAR_1D_FUNC(int64_t)
RESIZE_LINEAR_1D_FUNC(half)
RESIZE_LINEAR_1D_FUNC(float)
RESIZE_LINEAR_1D_FUNC(double)

template CUDA_LIB_EXPORT void ResizeLinear1DGrad(const enum ResizeLinearCoordinateTransformationMode mode,
                                                 const int64_t output_size, const int64_t in_width,
                                                 const int64_t out_width, const float *grad_output, float *grad_input,
                                                 const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void ResizeLinear1DGrad(const enum ResizeLinearCoordinateTransformationMode mode,
                                                 const int64_t output_size, const int64_t in_width,
                                                 const int64_t out_width, const float *grad_output, double *grad_input,
                                                 const uint32_t device_id, cudaStream_t stream);
