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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

__inline__ __device__ size_t GetTargetIndex(const size_t &output_index, const float &dim_scale,
                                            const size_t &input_size, const size_t &output_size) {
  constexpr size_t kNumberTwo = 2;
  if (output_size == input_size) {
    // scale_factor = 1
    return output_index;
  } else if (output_size == kNumberTwo * input_size) {
    // scale_factor = 2, shift input index
    return output_index >> 1;
  } else {
    return min(static_cast<size_t>(floorf(output_index * dim_scale)), input_size - 1); // NOLINT
  }
}

template <typename T>
__global__ void UpsampleNearest3d(const T *input, const size_t n, const size_t c, const size_t in_d, const size_t in_h,
  const size_t in_w, const size_t in_cdhw, const size_t in_dhw, const size_t in_hw, const size_t out_d,
  const size_t out_h, const size_t out_w, const size_t out_ncdhw, const size_t out_cdhw, const size_t out_dhw,
  const size_t out_hw, const float d_scale, const float h_scale, const float w_scale, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_ncdhw; pos += blockDim.x * gridDim.x) {
    const size_t posn = pos / out_cdhw;
    const size_t posc = pos / out_dhw % c;
    const size_t posd = pos / out_hw % out_d;
    const size_t posh = pos / out_w % out_h;
    const size_t posw = pos % out_w;
    const size_t input_posd = GetTargetIndex(posd, d_scale, in_d, out_d);
    const size_t input_posh = GetTargetIndex(posh, h_scale, in_h, out_h);
    const size_t input_posw = GetTargetIndex(posw, w_scale, in_w, out_w);
    const size_t input_pos = posn * in_cdhw + posc * in_dhw + input_posd * in_hw + input_posh * in_w + input_posw;
    output[pos] = input[input_pos];
  }
  return;
}

template <typename T>
void CalUpsampleNearest3d(const T *input, const size_t n, const size_t c, const size_t in_d, const size_t in_h,
  const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w, const float d_scale,
  const float h_scale, const float w_scale, T *output, cudaStream_t cuda_stream) {
  const size_t out_hw = out_h * out_w;
  const size_t out_dhw = out_d * out_hw;
  const size_t out_cdhw = c * out_dhw;
  const size_t out_ncdhw = n * out_cdhw;
  const size_t in_hw = in_h * in_w;
  const size_t in_dhw = in_d * in_hw;
  const size_t in_cdhw = c * in_dhw;
  UpsampleNearest3d<<<GET_BLOCKS(out_ncdhw), GET_THREADS, 0, cuda_stream>>>(input, n, c, in_d, in_h, in_w, in_cdhw,
    in_dhw, in_hw, out_d, out_h, out_w, out_ncdhw, out_cdhw, out_dhw, out_hw, d_scale, h_scale, w_scale, output);
  return;
}

template CUDA_LIB_EXPORT void CalUpsampleNearest3d<double>(const double *input, const size_t n, const size_t c,
  const size_t in_d, const size_t in_h, const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
  const float d_scale, const float h_scale, const float w_scale, double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleNearest3d<float>(const float *input, const size_t n, const size_t c,
  const size_t in_d, const size_t in_h, const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
  const float d_scale, const float h_scale, const float w_scale, float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleNearest3d<half>(const half *input, const size_t n, const size_t c,
  const size_t in_d, const size_t in_h, const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
  const float d_scale, const float h_scale, const float w_scale, half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleNearest3d<int8_t>(const int8_t *input, const size_t n, const size_t c,
  const size_t in_d, const size_t in_h, const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
  const float d_scale, const float h_scale, const float w_scale, int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleNearest3d<int16_t>(const int16_t *input, const size_t n, const size_t c,
  const size_t in_d, const size_t in_h, const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
  const float d_scale, const float h_scale, const float w_scale, int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleNearest3d<int>(const int *input, const size_t n, const size_t c,
  const size_t in_d, const size_t in_h, const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
  const float d_scale, const float h_scale, const float w_scale, int *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleNearest3d<int64_t>(const int64_t *input, const size_t n, const size_t c,
  const size_t in_d, const size_t in_h, const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
  const float d_scale, const float h_scale, const float w_scale, int64_t *output, cudaStream_t cuda_stream);
