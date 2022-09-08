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

#include <stdint.h>
#include <assert.h>
#include <algorithm>
#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scale_and_translate_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"


__device__ float ComputeLanczosKernel(float input, float radius) {
  const float PI = 3.14159265359;
  input = abs(input);
  if (input > radius) {
    return 0.0f;
  }
  // Need to special case the limit case of sin(input) / input when input is zero.
  if (input <= 1e-3) {
    return 1.0f;
  }
  return radius * sin(PI * input) * sin(PI * input / radius) / (PI * PI * input * input);
}

__device__ float ComputeGaussianKernel(float input, float radius) {
  const float kRadiusMultiplier = 3.0f;
  /**
   * https://en.wikipedia.org/wiki/Gaussian_function
   * We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
   * for Common Resampling Tasks" for kernels with a support of 3 pixels:
   * www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
   * This implies a radius of 1.5,
   */
  const float sigma = radius / kRadiusMultiplier;
  input = abs(input);
  if (input >= radius) {
    return 0.0;
  }
  return exp(-input * input / (2.0f * sigma * sigma));
}

__device__ float ComputeBoxKernel(float input) {
  float result;
  input = abs(input);
  if (input < 0.5f) {
    result = 1.0;
  } else if (input == 0.5f) {
    result = 0.5f;
  } else {
    result = 0.0;
  }
  return result;
}

__device__ float ComputeTriangleKernel(float input) {
  // https://en.wikipedia.org/wiki/Triangle_function
  float result;
  input = abs(input);
  if (input < 1) {
    result = 1.0 - input;
  } else {
    result = 0.0;
  }
  return result;
}

__device__ float ComputeKeysCubicKernel(float input) {
  /**
   * http://ieeexplore.ieee.org/document/1163711/
   * R. G. Keys. Cubic convolution interpolation for digital image
   * processing. IEEE Transactions on Acoustics, Speech, and Signal
   * Processing, 29(6):1153–1160, 1981.
   */
  input = abs(input);
  float result;
  if (input >= 2) {
    result = 0;
  } else if (input >= 1) {
    result = -0.5f * input + 2.5f;
    result = result * input - 4.0f;
    result = result * input + 2.0;
  } else {
    result = (1.5f * input - 2.5f) * input;
    result = result * input + 1.0;
  }
  return result;
}

__device__ float ComputeMitchellCubicKernel(float input) {
  /**
   * https://doi.org/10.1145/378456.378514
   * D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
   * graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
   * 22(4):221–228, 1988.
   */
  input = abs(input);
  if (input >= 2.0f) {
    return 0.0f;
  } else if (input >= 1.0f) {
    return (((-7.0f / 18.0f) * input + 2.0f) * input - 10.0f / 3.0f) * input + 16.0f / 9.0f;
  } else {
    return (((7.0f / 6.0f) * input - 2.0f) * input) * input + 8.0f / 9.0f;
  }
}

template <typename T>
__global__ void SetZero(T *target, size_t size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    target[i] = 0;
  }
}

__global__ void ComputeSpanSize(const float radius, const float *scale, bool antialias, const int64_t *input_shape,
                                int32_t *spans_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < 2; pos += blockDim.x * gridDim.x) {
    CUDA_KERNEL_ASSERT(scale[pos] > 0);
    const float kernel_scale = antialias ? max(1.0 / scale[pos], 1.0f) : 1.0f;
    spans_size[pos] =
      min(2 * static_cast<int>(ceil(radius * kernel_scale)) + 1, static_cast<int>(input_shape[1 + pos]));
  }
}

template <typename T>
__device__ inline const T &Clamp(const T &low, const T &high, const T &value) {
  if (high < value) return high;
  if (value < low) return low;
  return value;
}

__global__ void ComputeLanczosSpan(const float radius, const int64_t *input_shape, const int32_t *size,
                                   const float *scale, const float *translate, const bool antialias,
                                   const int32_t *spans_size, int32_t *starts, float *weights,
                                   const float numeric_limits_min) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size[0] + size[1]; pos += blockDim.x * gridDim.x) {
    const size_t mode = min(size_t(1), pos / size[0]);
    const int32_t input_size = input_shape[1 + mode];
    const float inv_scale = 1.0 / scale[mode];
    const float kernel_scale = antialias ? max(inv_scale, 1.0f) : 1.0f;
    const int32_t span_size = spans_size[mode];
    const size_t out_index = (mode) ? size[0] * spans_size[0] + (pos - size[0]) * span_size : pos * span_size;
    const float sample_f = (pos - mode * size[0] + 0.5f) * inv_scale - inv_scale * translate[mode];
    if (sample_f < 0 || sample_f > input_size) {
      starts[pos] = 0;
    } else {
      int32_t span_start = ceil(sample_f - radius * kernel_scale - 0.5f);
      int32_t span_end = floor(sample_f + radius * kernel_scale - 0.5f);
      span_start = Clamp(static_cast<int32_t>(0), input_size - 1, span_start);
      span_end = Clamp(static_cast<int32_t>(0), input_size - 1, span_end) + 1;
      float total_weight_sum = 0.0;
      for (int i = 0; i < span_end - span_start; ++i) {
        float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
        float weight = ComputeLanczosKernel(abs(kernel_pos / kernel_scale), radius);
        total_weight_sum += weight;
      }
      if (abs(total_weight_sum) >= 1000.0f * numeric_limits_min) {
        float one_over_total_weight_sum = 1.0f / total_weight_sum;
        for (int i = 0; i < span_end - span_start; ++i) {
          float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
          float weight = ComputeLanczosKernel(abs(kernel_pos / kernel_scale), radius);
          weights[out_index + i] = weight * one_over_total_weight_sum;
        }
      }
      starts[pos] = span_start;
    }
  }
}

__global__ void ComputeGaussianSpan(const float radius, const int64_t *input_shape, const int32_t *size,
                                    const float *scale, const float *translate, const bool antialias,
                                    const int32_t *spans_size, int32_t *starts, float *weights,
                                    const float numeric_limits_min) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size[0] + size[1]; pos += blockDim.x * gridDim.x) {
    const size_t mode = min(size_t(1), pos / size[0]);
    const int64_t input_size = input_shape[1 + mode];
    const float inv_scale = 1.0 / scale[mode];
    const float kernel_scale = antialias ? max(inv_scale, 1.0f) : 1.0f;
    const int32_t span_size = spans_size[mode];
    const size_t out_index = (mode) ? size[0] * spans_size[0] + (pos - size[0]) * span_size : pos * span_size;
    const float sample_f = (pos - mode * size[0] + 0.5f) * inv_scale - inv_scale * translate[mode];
    if (sample_f < 0 || sample_f > input_size) {
      starts[pos] = 0;
    } else {
      int64_t span_start = ceil(sample_f - radius * kernel_scale - 0.5f);
      int64_t span_end = floor(sample_f + radius * kernel_scale - 0.5f);
      span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
      span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
      float total_weight_sum = 0.0;
      for (int i = 0; i < span_end - span_start; ++i) {
        float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
        float weight = ComputeGaussianKernel(abs(kernel_pos / kernel_scale), radius);
        total_weight_sum += weight;
      }
      if (abs(total_weight_sum) >= 1000.0f * numeric_limits_min) {
        float one_over_total_weight_sum = 1.0f / total_weight_sum;
        for (int i = 0; i < span_end - span_start; ++i) {
          float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
          float weight = ComputeGaussianKernel(abs(kernel_pos / kernel_scale), radius);
          weights[out_index + i] = weight * one_over_total_weight_sum;
        }
      }
      starts[pos] = span_start;
    }
  }
}

__global__ void ComputeBoxSpan(const float radius, const int64_t *input_shape, const int32_t *size, const float *scale,
                               const float *translate, const bool antialias, const int32_t *spans_size, int32_t *starts,
                               float *weights, const float numeric_limits_min) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size[0] + size[1]; pos += blockDim.x * gridDim.x) {
    const size_t mode = min(size_t(1), pos / size[0]);
    const int64_t input_size = input_shape[1 + mode];
    const float inv_scale = 1.0 / scale[mode];
    const float kernel_scale = antialias ? max(inv_scale, 1.0f) : 1.0f;
    const int32_t span_size = spans_size[mode];
    const size_t out_index = (mode) ? size[0] * spans_size[0] + (pos - size[0]) * span_size : pos * span_size;
    const float sample_f = (pos - mode * size[0] + 0.5f) * inv_scale - inv_scale * translate[mode];
    if (sample_f < 0 || sample_f > input_size) {
      starts[pos] = 0;
    } else {
      int64_t span_start = ceil(sample_f - radius * kernel_scale - 0.5f);
      int64_t span_end = floor(sample_f + radius * kernel_scale - 0.5f);
      span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
      span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
      float total_weight_sum = 0.0;
      for (int i = 0; i < span_end - span_start; ++i) {
        float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
        float weight = ComputeBoxKernel(abs(kernel_pos / kernel_scale));
        total_weight_sum += weight;
      }
      if (abs(total_weight_sum) >= 1000.0f * numeric_limits_min) {
        float one_over_total_weight_sum = 1.0f / total_weight_sum;
        for (int i = 0; i < span_end - span_start; ++i) {
          float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
          float weight = ComputeBoxKernel(abs(kernel_pos / kernel_scale));
          weights[out_index + i] = weight * one_over_total_weight_sum;
        }
      }
      starts[pos] = span_start;
    }
  }
}

__global__ void ComputeKeysCubicSpan(const float radius, const int64_t *input_shape, const int32_t *size,
                                     const float *scale, const float *translate, const bool antialias,
                                     const int32_t *spans_size, int32_t *starts, float *weights,
                                     const float numeric_limits_min) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size[0] + size[1]; pos += blockDim.x * gridDim.x) {
    const size_t mode = min(size_t(1), pos / size[0]);
    const int64_t input_size = input_shape[1 + mode];
    const float inv_scale = 1.0 / scale[mode];
    const float kernel_scale = antialias ? max(inv_scale, 1.0f) : 1.0f;
    const int32_t span_size = spans_size[mode];
    const size_t out_index = (mode) ? size[0] * spans_size[0] + (pos - size[0]) * span_size : pos * span_size;
    const float sample_f = (pos - mode * size[0] + 0.5f) * inv_scale - inv_scale * translate[mode];
    if (sample_f < 0 || sample_f > input_size) {
      starts[pos] = 0;
    } else {
      int64_t span_start = ceil(sample_f - radius * kernel_scale - 0.5f);
      int64_t span_end = floor(sample_f + radius * kernel_scale - 0.5f);
      span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
      span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
      float total_weight_sum = 0.0;
      for (int i = 0; i < span_end - span_start; ++i) {
        float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
        float weight = ComputeKeysCubicKernel(abs(kernel_pos / kernel_scale));
        total_weight_sum += weight;
      }
      if (abs(total_weight_sum) >= 1000.0f * numeric_limits_min) {
        float one_over_total_weight_sum = 1.0f / total_weight_sum;
        for (int i = 0; i < span_end - span_start; ++i) {
          float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
          float weight = ComputeKeysCubicKernel(abs(kernel_pos / kernel_scale));
          weights[out_index + i] = weight * one_over_total_weight_sum;
        }
      }
      starts[pos] = span_start;
    }
  }
}

__global__ void ComputeMitchellCubicSpan(const float radius, const int64_t *input_shape, const int32_t *size,
                                         const float *scale, const float *translate, const bool antialias,
                                         const int32_t *spans_size, int32_t *starts, float *weights,
                                         const float numeric_limits_min) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size[0] + size[1]; pos += blockDim.x * gridDim.x) {
    const size_t mode = min(size_t(1), pos / size[0]);
    const int64_t input_size = input_shape[1 + mode];
    const float inv_scale = 1.0 / scale[mode];
    const float kernel_scale = antialias ? max(inv_scale, 1.0f) : 1.0f;
    const int32_t span_size = spans_size[mode];
    const size_t out_index = (mode) ? size[0] * spans_size[0] + (pos - size[0]) * span_size : pos * span_size;
    const float sample_f = (pos - mode * size[0] + 0.5f) * inv_scale - inv_scale * translate[mode];
    if (sample_f < 0 || sample_f > input_size) {
      starts[pos] = 0;
    } else {
      int64_t span_start = ceil(sample_f - radius * kernel_scale - 0.5f);
      int64_t span_end = floor(sample_f + radius * kernel_scale - 0.5f);
      span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
      span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
      float total_weight_sum = 0.0;
      for (int i = 0; i < span_end - span_start; ++i) {
        float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
        float weight = ComputeMitchellCubicKernel(abs(kernel_pos / kernel_scale));
        total_weight_sum += weight;
      }
      if (abs(total_weight_sum) >= 1000.0f * numeric_limits_min) {
        float one_over_total_weight_sum = 1.0f / total_weight_sum;
        for (int i = 0; i < span_end - span_start; ++i) {
          float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
          float weight = ComputeMitchellCubicKernel(abs(kernel_pos / kernel_scale));
          weights[out_index + i] = weight * one_over_total_weight_sum;
        }
      }
      starts[pos] = span_start;
    }
  }
}

__global__ void ComputeTriangleSpan(const float radius, const int64_t *input_shape, const int32_t *size,
                                    const float *scale, const float *translate, const bool antialias,
                                    const int32_t *spans_size, int32_t *starts, float *weights,
                                    const float numeric_limits_min) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size[0] + size[1]; pos += blockDim.x * gridDim.x) {
    const size_t mode = min(size_t(1), pos / size[0]);
    const int64_t input_size = input_shape[1 + mode];
    const float inv_scale = 1.0 / scale[mode];
    const float kernel_scale = antialias ? max(inv_scale, 1.0f) : 1.0f;
    const int32_t span_size = spans_size[mode];
    const size_t out_index = (mode) ? size[0] * spans_size[0] + (pos - size[0]) * span_size : pos * span_size;
    const float sample_f = (pos - mode * size[0] + 0.5f) * inv_scale - inv_scale * translate[mode];
    if (sample_f < 0 || sample_f > input_size) {
      starts[pos] = 0;
    } else {
      int64_t span_start = ceil(sample_f - radius * kernel_scale - 0.5f);
      int64_t span_end = floor(sample_f + radius * kernel_scale - 0.5f);
      span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
      span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
      float total_weight_sum = 0.0;
      for (int i = 0; i < span_end - span_start; ++i) {
        float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
        float weight = ComputeTriangleKernel(abs(kernel_pos / kernel_scale));
        total_weight_sum += weight;
      }
      if (abs(total_weight_sum) >= 1000.0f * numeric_limits_min) {
        float one_over_total_weight_sum = 1.0f / total_weight_sum;
        for (int i = 0; i < span_end - span_start; ++i) {
          float kernel_pos = static_cast<float>(span_start + i) + 0.5f - sample_f;
          float weight = ComputeTriangleKernel(abs(kernel_pos / kernel_scale));
          weights[out_index + i] = weight * one_over_total_weight_sum;
        }
      }
      starts[pos] = span_start;
    }
  }
}

template <typename T>
__global__ void GatherRows(const int32_t size, int32_t *spans_size, int *starts, float *weights, const T *images,
                           const int64_t batch, const int64_t input_height, const int64_t input_width,
                           const int64_t output_height, const int64_t output_width, const int64_t channels,
                           const int64_t input_pix_per_batch, const int64_t intermediate_pix_per_batch,
                           float *outputs) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int32_t span_size = spans_size[0];
    int pos_temp = pos;
    int64_t y = pos_temp % output_height;
    int64_t batch_id = pos_temp / output_height;
    const T *image = images + input_pix_per_batch * batch_id;
    float *output = outputs + intermediate_pix_per_batch * batch_id;
    const int64_t in_row_size = input_width * channels;
    const int64_t out_row_size = output_width * channels;
    if (batch_id < 0 || batch_id >= batch) {
      continue;
    }
    if (y < 0 || y >= output_height) {
      continue;
    }
    const float *weights_start = weights + y * span_size;
    const int this_span_size = min(starts[y] + span_size, static_cast<int>(input_height)) - starts[y];
    const float *const weights_end = weights_start + this_span_size;
    float *out_row_data = output + out_row_size * y;
    int in_row = starts[y];
    const T *in_row_data = image + in_row_size * in_row;
    const T *temp_in_vec;
    float temp_weight;
    float *temp_out_vec;
    for (const float *weight_it = weights_start; weight_it != weights_end; ++weight_it) {
      temp_in_vec = in_row_data;
      temp_weight = *weight_it;
      temp_out_vec = out_row_data;
      float *temp_out_vec_end = temp_out_vec + in_row_size;
      for (; temp_out_vec != temp_out_vec_end; ++temp_out_vec, ++temp_in_vec) {
        MsAtomicAdd(temp_out_vec, temp_weight * static_cast<float>(*temp_in_vec));
      }
      in_row_data += in_row_size;
    }
  }
  return;
}

template <typename T>
__global__ void GatherColumns(const int32_t size, int32_t *spans_size, int32_t *starts, float *weight, const T *images,
                              const int64_t batch, const int64_t input_height, const int64_t input_width,
                              const int64_t output_height, const int64_t output_width, const int64_t channels,
                              const int64_t intermediate_pix_per_batch, const int64_t output_pix_per_batch,
                              float *outputs) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int32_t span_size = spans_size[1];
    float *weights = weight + spans_size[0] * output_height;
    int pos_temp = pos;
    int64_t y = pos_temp % output_height;
    pos_temp = pos_temp / output_height;
    int64_t x = pos_temp % output_width;
    int64_t batch_id = pos_temp / output_width;
    const T *image = images + intermediate_pix_per_batch * batch_id;
    float *output = outputs + output_pix_per_batch * batch_id;
    const int64_t in_row_size = input_width * channels;
    const int64_t out_row_size = output_width * channels;
    const T *input_row_start = image + in_row_size * y;
    float *out_pix = output + out_row_size * y;
    out_pix += channels * x;
    if (batch_id < 0 || batch_id >= batch) {
      continue;
    }
    if (y < 0 || y >= output_height) {
      continue;
    }
    if (x < 0 || x >= output_width) {
      continue;
    }
    const float *weights_start = weights + x * span_size;
    const int this_span_size = min(starts[x] + span_size, static_cast<int>(input_width)) - starts[x];
    const float *weights_end = weights_start + this_span_size;
    const T *in_pix = input_row_start + starts[x] * channels;
    for (const float *weight_ptr = weights_start; weight_ptr != weights_end; ++weight_ptr) {
      float w = *weight_ptr;
      for (int c = 0; c < channels; ++c) {
        MsAtomicAdd(out_pix + c, w * static_cast<float>(in_pix[c]));
      }
      in_pix += channels;
    }
  }
  return;
}

__global__ void ComputeGradSpan(const int64_t *input_shape, const int32_t *size, const int32_t *spans_size,
                                const int32_t *forward_starts, const float *forward_weights, int32_t *grad_starts,
                                float *grad_weights, int32_t *weight_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_shape[1] + input_shape[2];
       pos += blockDim.x * gridDim.x) {
    const size_t mode = min(size_t(1), pos/input_shape[1]);
    const int64_t output_size = size[mode];
    const int32_t span_size = spans_size[mode];
    const size_t target_input_index = pos - mode * input_shape[1];
    size_t weight_output_index =
      (!mode) ? pos * output_size : size[0] * input_shape[1] + (pos - input_shape[1]) * size[1];
    grad_starts[pos] = 0;
    weight_size[pos] = 0;
    bool flag = true;
    for (int output_index = 0; output_index < output_size; ++output_index) {
      int input_index = forward_starts[mode * size[0] + output_index];
      for (int j = 0; j < span_size; ++j, ++input_index) {
        if (input_index == target_input_index) {
          const float weight = forward_weights[mode * spans_size[0] * size[0] + output_index * span_size + j];
          if (weight != 0.0f) {
            if (flag) {
              grad_starts[pos] = output_index;
              flag = false;
            }
            grad_weights[weight_output_index + output_index - grad_starts[pos]] += weight;
            weight_size[pos] += 1;
          }
          break;
        }
      }
    }
  }
}

template <typename T>
__global__ void GatherRowsGrad(const size_t thread_num, const int32_t *starts, const float *weights, const T *images,
                               const int64_t *input_shape, const int32_t *size, const int64_t input_pix_per_batch,
                               const int64_t intermediate_pix_per_batch, float *outputs, const int32_t *weight_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < thread_num; pos += blockDim.x * gridDim.x) {
    int64_t y = pos % input_shape[1];
    int32_t span_size = size[0];
    int64_t batch_id = pos / input_shape[1];
    const T *image = images + input_pix_per_batch * batch_id;
    float *output = outputs + intermediate_pix_per_batch * batch_id;
    const int64_t in_row_size = size[1] * input_shape[3];
    const int64_t out_row_size = in_row_size;
    if (batch_id < 0 || batch_id >= input_shape[0]) {
      continue;
    }
    if (y < 0 || y >= input_shape[1]) {
      continue;
    }
    const float *weights_start = weights + y * span_size;
    const int real_weight_size = min(starts[y] + weight_size[y], static_cast<int>(size[0])) - starts[y];
    const float *const weights_end = weights_start + real_weight_size;
    float *out_row_data = output + out_row_size * y;
    int in_row = starts[y];
    const T *in_row_data = image + in_row_size * in_row;
    const T *temp_in_vec;
    float temp_weight;
    float *temp_out_vec;
    for (const float *weight_it = weights_start; weight_it != weights_end; ++weight_it) {
      temp_in_vec = in_row_data;
      temp_weight = *weight_it;
      temp_out_vec = out_row_data;
      float *temp_out_vec_end = temp_out_vec + in_row_size;
      for (; temp_out_vec != temp_out_vec_end; ++temp_out_vec, ++temp_in_vec) {
        MsAtomicAdd(temp_out_vec, temp_weight * static_cast<float>(*temp_in_vec));
      }
      in_row_data += in_row_size;
    }
  }
  return;
}

template <typename T>
__global__ void GatherColumnsGrad(const size_t thread_num, const int32_t *grad_starts, const float *grad_weights,
                                  const T *images, const int64_t *input_shape, const int32_t *size,
                                  const int64_t intermediate_pix_per_batch, const int64_t output_pix_per_batch,
                                  float *outputs, const int32_t *weight_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < thread_num; pos += blockDim.x * gridDim.x) {
    const int32_t *starts = grad_starts + input_shape[1];
    const float *weights = grad_weights + input_shape[1] * size[0];
    int64_t y = pos % input_shape[1];
    size_t tem_pos = pos / input_shape[1];
    int64_t x = tem_pos % input_shape[2];
    int32_t span_size = size[1];
    int64_t batch_id = tem_pos / input_shape[2];
    const T *image = images + intermediate_pix_per_batch * batch_id;
    float *output = outputs + output_pix_per_batch * batch_id;
    const int64_t in_row_size = size[1] * input_shape[3];
    const int64_t out_row_size = input_shape[2] * input_shape[3];
    const T *input_row_start = image + in_row_size * y;
    float *out_pix = output + out_row_size * y;
    out_pix += input_shape[3] * x;
    if (batch_id < 0 || batch_id >= input_shape[0]) {
      continue;
    }
    if (y < 0 || y >= input_shape[1]) {
      continue;
    }
    if (x < 0 || x >= input_shape[2]) {
      continue;
    }
    const float *weights_start = weights + x * span_size;
    const int real_weight_size =
      min(starts[x] + weight_size[input_shape[1] + x], static_cast<int>(size[1])) - starts[x];
    const float *weights_end = weights_start + real_weight_size;
    const T *in_pix = input_row_start + starts[x] * input_shape[3];
    for (const float *weight_ptr = weights_start; weight_ptr != weights_end; ++weight_ptr) {
      float w = *weight_ptr;
      for (int c = 0; c < input_shape[3]; ++c) {
        MsAtomicAdd(out_pix + c, w * static_cast<float>(in_pix[c]));
      }
      in_pix += input_shape[3];
    }
  }
  return;
}

template <typename T>
void CalScaleAndTranslate(const size_t *thread_num, const T *image, const float *scale, const float *translation,
                          int64_t batch, int64_t image_height, int64_t image_width, int64_t channels,
                          int64_t output_height, int64_t output_width, std::string kernel_type, bool antialias,
                          const float radius, const int64_t *input_shape, const int32_t *size, int32_t *spans_size,
                          int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  // Compute span;
  float numeric_limits_min = std::numeric_limits<float>::min();
  SetZero<<<CUDA_BLOCKS(device_id, thread_num[0]), CUDA_THREADS(device_id), 0, cuda_stream>>>(forward_weights,
                                                                                              thread_num[0]);
  ComputeSpanSize<<<CUDA_BLOCKS(device_id, 2), CUDA_THREADS(device_id), 0, cuda_stream>>>(radius, scale, antialias,
                                                                                          input_shape, spans_size);
  if (kernel_type == "lanczos1" || kernel_type == "lanczos3" || kernel_type == "lanczos5") {
    ComputeLanczosSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      radius, input_shape, size, scale, translation, antialias, spans_size, forward_starts, forward_weights,
      numeric_limits_min);
  } else if (kernel_type == "gaussian") {
    ComputeGaussianSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      radius, input_shape, size, scale, translation, antialias, spans_size, forward_starts, forward_weights,
      numeric_limits_min);
  } else if (kernel_type == "box") {
    ComputeBoxSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      radius, input_shape, size, scale, translation, antialias, spans_size, forward_starts, forward_weights,
      numeric_limits_min);
  } else if (kernel_type == "triangle") {
    ComputeTriangleSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      radius, input_shape, size, scale, translation, antialias, spans_size, forward_starts, forward_weights,
      numeric_limits_min);
  } else if (kernel_type == "keyscubic") {
    ComputeKeysCubicSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      radius, input_shape, size, scale, translation, antialias, spans_size, forward_starts, forward_weights,
      numeric_limits_min);
  } else if (kernel_type == "mitchellcubic") {
    ComputeMitchellCubicSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      radius, input_shape, size, scale, translation, antialias, spans_size, forward_starts, forward_weights,
      numeric_limits_min);
  }
  // Set zero
  size_t size_gather_rows = thread_num[2];
  size_t size_gather_columns = thread_num[3];
  int64_t input_pix_per_batch = image_height * image_width * channels;
  int64_t intermediate_pix_per_batch = image_width * output_height * channels;
  int64_t output_pix_per_batch = output_height * output_width * channels;
  SetZero<<<CUDA_BLOCKS(device_id, intermediate_pix_per_batch * batch), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    intermediate, intermediate_pix_per_batch * batch);
  SetZero<<<CUDA_BLOCKS(device_id, output_pix_per_batch * batch), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    output, output_pix_per_batch * batch);
  // GatherRows
  int32_t *col_starts = forward_starts + output_height;
  GatherRows<<<CUDA_BLOCKS(device_id, size_gather_rows), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size_gather_rows, spans_size, forward_starts, forward_weights, image, batch, image_height, image_width,
    output_height, image_width, channels, input_pix_per_batch, intermediate_pix_per_batch, intermediate);
  // GatherColumns
  GatherColumns<<<CUDA_BLOCKS(device_id, size_gather_columns), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size_gather_columns, spans_size, col_starts, forward_weights, intermediate, batch, output_height, image_width,
    output_height, output_width, channels, intermediate_pix_per_batch, output_pix_per_batch, output);
  return;
}

template <typename T>
void CallScaleAndTranslateGrad(const std::string kernel_type, const T *grads, const T *original_image,
                               const float radius, const int64_t *input_shape, const int32_t *size, const float *scale,
                               const float *translate, const bool antialias, int32_t *spans_size,
                               int32_t *forward_starts, int32_t *grad_starts, float *forward_weights,
                               float *grad_weights, const size_t *thread_num, float *intermediate,
                               const int64_t input_pix_per_batch, const int64_t intermediate_pix_per_batch,
                               const int64_t output_pix_per_batch, float *output, int32_t *weight_size,
                               const uint32_t &device_id, cudaStream_t cuda_stream) {
  float numeric_limits_min = std::numeric_limits<float>::min();
  SetZero<<<CUDA_BLOCKS(device_id, thread_num[0]), CUDA_THREADS(device_id), 0, cuda_stream>>>
    (forward_weights, thread_num[0]);
  ComputeSpanSize<<<CUDA_BLOCKS(device_id, 2), CUDA_THREADS(device_id), 0, cuda_stream>>>(radius, scale, antialias,
    input_shape, spans_size);
  if (kernel_type == "lanczos1" || kernel_type == "lanczos3" || kernel_type == "lanczos5") {
    ComputeLanczosSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>
      (radius, input_shape, size, scale, translate, antialias, spans_size, forward_starts, forward_weights,
       numeric_limits_min);
  } else if (kernel_type == "gaussian") {
    ComputeGaussianSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>
      (radius, input_shape, size, scale, translate, antialias, spans_size, forward_starts, forward_weights,
       numeric_limits_min);
  } else if (kernel_type == "box") {
    ComputeBoxSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>
      (radius, input_shape, size, scale, translate, antialias, spans_size, forward_starts, forward_weights,
       numeric_limits_min);
  } else if (kernel_type == "triangle") {
    ComputeTriangleSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>
      (radius, input_shape, size, scale, translate, antialias, spans_size, forward_starts, forward_weights,
       numeric_limits_min);
  } else if (kernel_type == "keyscubic") {
    ComputeKeysCubicSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>
      (radius, input_shape, size, scale, translate, antialias, spans_size, forward_starts, forward_weights,
       numeric_limits_min);
  } else if (kernel_type == "mitchellcubic") {
    ComputeMitchellCubicSpan<<<CUDA_BLOCKS(device_id, thread_num[1]), CUDA_THREADS(device_id), 0, cuda_stream>>>
      (radius, input_shape, size, scale, translate, antialias, spans_size, forward_starts, forward_weights,
       numeric_limits_min);
  }

  ComputeGradSpan<<<CUDA_BLOCKS(device_id, thread_num[2]), CUDA_THREADS(device_id), 0, cuda_stream>>>(input_shape,
    size, spans_size, forward_starts,  forward_weights,  grad_starts, grad_weights, weight_size);
  SetZero<<<CUDA_BLOCKS(device_id, thread_num[3]), CUDA_THREADS(device_id), 0, cuda_stream>>>
    (intermediate, thread_num[3]);
  SetZero<<<CUDA_BLOCKS(device_id, thread_num[4]), CUDA_THREADS(device_id), 0, cuda_stream>>>
    (output, thread_num[4]);
  GatherRowsGrad<<<CUDA_BLOCKS(device_id, thread_num[5]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    thread_num[5], grad_starts, grad_weights, grads, input_shape, size, input_pix_per_batch, intermediate_pix_per_batch,
    intermediate, weight_size);
  GatherColumnsGrad<<<CUDA_BLOCKS(device_id, thread_num[6]), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    thread_num[6], grad_starts, grad_weights, intermediate, input_shape, size, intermediate_pix_per_batch,
    output_pix_per_batch, output, weight_size);
}

template CUDA_LIB_EXPORT void CalScaleAndTranslate<int8_t>(
  const size_t *thread_num, const int8_t *image, const float *scale, const float *translation, int64_t batch,
  int64_t image_height, int64_t image_width, int64_t channels, int64_t output_height, int64_t output_width,
  std::string kernel_type, bool antialias, const float radius, const int64_t *input_shape, const int32_t *size,
  int32_t *spans_size, int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalScaleAndTranslate<int16_t>(
  const size_t *thread_num, const int16_t *image, const float *scale, const float *translation, int64_t batch,
  int64_t image_height, int64_t image_width, int64_t channels, int64_t output_height, int64_t output_width,
  std::string kernel_type, bool antialias, const float radius, const int64_t *input_shape, const int32_t *size,
  int32_t *spans_size, int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalScaleAndTranslate<int32_t>(
  const size_t *thread_num, const int32_t *image, const float *scale, const float *translation, int64_t batch,
  int64_t image_height, int64_t image_width, int64_t channels, int64_t output_height, int64_t output_width,
  std::string kernel_type, bool antialias, const float radius, const int64_t *input_shape, const int32_t *size,
  int32_t *spans_size, int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalScaleAndTranslate<int64_t>(
  const size_t *thread_num, const int64_t *image, const float *scale, const float *translation, int64_t batch,
  int64_t image_height, int64_t image_width, int64_t channels, int64_t output_height, int64_t output_width,
  std::string kernel_type, bool antialias, const float radius, const int64_t *input_shape, const int32_t *size,
  int32_t *spans_size, int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalScaleAndTranslate<half>(
  const size_t *thread_num, const half *image, const float *scale, const float *translation, int64_t batch,
  int64_t image_height, int64_t image_width, int64_t channels, int64_t output_height, int64_t output_width,
  std::string kernel_type, bool antialias, const float radius, const int64_t *input_shape, const int32_t *size,
  int32_t *spans_size, int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalScaleAndTranslate<float>(
  const size_t *thread_num, const float *image, const float *scale, const float *translation, int64_t batch,
  int64_t image_height, int64_t image_width, int64_t channels, int64_t output_height, int64_t output_width,
  std::string kernel_type, bool antialias, const float radius, const int64_t *input_shape, const int32_t *size,
  int32_t *spans_size, int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalScaleAndTranslate<double>(
  const size_t *thread_num, const double *image, const float *scale, const float *translation, int64_t batch,
  int64_t image_height, int64_t image_width, int64_t channels, int64_t output_height, int64_t output_width,
  std::string kernel_type, bool antialias, const float radius, const int64_t *input_shape, const int32_t *size,
  int32_t *spans_size, int32_t *forward_starts, float *forward_weights, float *intermediate, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CallScaleAndTranslateGrad<float>(const std::string kernel_type, const float *grads,
  const float *original_image, const float radius, const int64_t *input_shape, const int32_t *size, const float *scale,
  const float *translate, const bool antialias, int32_t *spans_size, int32_t *forward_starts, int32_t *grad_starts,
  float *forward_weights, float *grad_weights, const size_t *thread_num, float *intermediate,
  const int64_t input_pix_per_batch, const int64_t intermediate_pix_per_batch, const int64_t output_pix_per_batch,
  float *output, int32_t *weight_size, const uint32_t &device_id, cudaStream_t cuda_stream);
