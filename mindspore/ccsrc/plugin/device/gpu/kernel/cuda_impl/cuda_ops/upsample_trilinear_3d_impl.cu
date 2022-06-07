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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

__inline__ __device__ float GetInput(const float *input, size_t index) { return input[index]; }
__inline__ __device__ float GetInput(const half *input, size_t index) {
  // required to maintain precision for fp16 input
  return static_cast<float>(input[index]);
}
__inline__ __device__ float GetTargetIndex(size_t output_idx, float dim_scale, bool align_corner) {
  float return_val = 0;
  if (align_corner) {
    return_val = output_idx * dim_scale;
  } else {
    return_val = dim_scale * (output_idx + 0.5f) - 0.5f;
  }
  return max(return_val, 0.0f);
}

template <typename T>  // float32 / float16
__global__ void UpsampleTrilinear3D(const T *input, const size_t n, const size_t c, const size_t in_d,
                                    const size_t in_h, const size_t in_w, const size_t in_cdhw, const size_t in_dhw,
                                    const size_t in_hw, const size_t out_d, const size_t out_h, const size_t out_w,
                                    const size_t out_ncdhw, const size_t out_cdhw, const size_t out_dhw,
                                    const size_t out_hw, const float d_scale, const float h_scale, const float w_scale,
                                    const bool align_corner, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_ncdhw; pos += blockDim.x * gridDim.x) {
    const size_t posn = pos / out_cdhw;
    const size_t posc = pos / out_dhw % c;
    const size_t posd = pos / out_hw % out_d;
    const size_t posh = pos / out_w % out_h;
    const size_t posw = pos % out_w;
    // calculate scaled values for input index
    const float scaled_in_d = GetTargetIndex(posd, d_scale, align_corner);
    const float scaled_in_h = GetTargetIndex(posh, h_scale, align_corner);
    const float scaled_in_w = GetTargetIndex(posw, w_scale, align_corner);
    // bounding indices and lambda values
    const size_t in_d_floor = floorf(scaled_in_d);  // NOLINT
    const size_t in_h_floor = floorf(scaled_in_h);  // NOLINT
    const size_t in_w_floor = floorf(scaled_in_w);  // NOLINT
    const size_t in_d_ceil = (in_d_floor < in_d - 1) ? in_d_floor + 1 : in_d_floor;
    const size_t in_h_ceil = (in_h_floor < in_h - 1) ? in_h_floor + 1 : in_h_floor;
    const size_t in_w_ceil = (in_w_floor < in_w - 1) ? in_w_floor + 1 : in_w_floor;
    const float lambda_w0 = (scaled_in_w - in_w_floor);
    const float lambda_w1 = (1.0f - lambda_w0);
    const float lambda_h0 = (scaled_in_h - in_h_floor);
    const float lambda_h1 = (1.0f - lambda_h0);
    const float lambda_d0 = (scaled_in_d - in_d_floor);
    const float lambda_d1 = (1.0f - lambda_d0);
    // get required indices
    const size_t p1 = posn * in_cdhw + posc * in_dhw + in_d_floor * in_hw + in_h_floor * in_w + in_w_floor;
    const size_t p2 = posn * in_cdhw + posc * in_dhw + in_d_floor * in_hw + in_h_floor * in_w + in_w_ceil;
    const size_t p3 = posn * in_cdhw + posc * in_dhw + in_d_floor * in_hw + in_h_ceil * in_w + in_w_floor;
    const size_t p4 = posn * in_cdhw + posc * in_dhw + in_d_floor * in_hw + in_h_ceil * in_w + in_w_ceil;
    const size_t p5 = posn * in_cdhw + posc * in_dhw + in_d_ceil * in_hw + in_h_floor * in_w + in_w_floor;
    const size_t p6 = posn * in_cdhw + posc * in_dhw + in_d_ceil * in_hw + in_h_floor * in_w + in_w_ceil;
    const size_t p7 = posn * in_cdhw + posc * in_dhw + in_d_ceil * in_hw + in_h_ceil * in_w + in_w_floor;
    const size_t p8 = posn * in_cdhw + posc * in_dhw + in_d_ceil * in_hw + in_h_ceil * in_w + in_w_ceil;

    const float val = lambda_d1 * ((lambda_h1 * (lambda_w1 * GetInput(input, p1) + lambda_w0 * GetInput(input, p2))) +
                                   (lambda_h0 * (lambda_w1 * GetInput(input, p3) + lambda_w0 * GetInput(input, p4)))) +
                      lambda_d0 * ((lambda_h1 * (lambda_w1 * GetInput(input, p5) + lambda_w0 * GetInput(input, p6))) +
                                   (lambda_h0 * (lambda_w1 * GetInput(input, p7) + lambda_w0 * GetInput(input, p8))));
    output[pos] = static_cast<T>(val);
  }
  return;
}

template <typename T>
void CalUpsampleTrilinear3D(const T *input, const size_t n, const size_t c, const size_t in_d, const size_t in_h,
                            const size_t in_w, const size_t out_d, const size_t out_h, const size_t out_w,
                            const float d_scale, const float h_scale, const float w_scale, const bool align_corner,
                            T *output, cudaStream_t cuda_stream) {
  const size_t out_hw = out_h * out_w;
  const size_t out_dhw = out_d * out_hw;
  const size_t out_cdhw = c * out_dhw;
  const size_t out_ncdhw = n * out_cdhw;
  const size_t in_hw = in_h * in_w;
  const size_t in_dhw = in_d * in_hw;
  const size_t in_cdhw = c * in_dhw;
  UpsampleTrilinear3D<<<GET_BLOCKS(out_ncdhw), GET_THREADS, 0, cuda_stream>>>(
    input, n, c, in_d, in_h, in_w, in_cdhw, in_dhw, in_hw, out_d, out_h, out_w, out_ncdhw, out_cdhw, out_dhw, out_hw,
    d_scale, h_scale, w_scale, align_corner, output);
  return;
}

template CUDA_LIB_EXPORT void CalUpsampleTrilinear3D<half>(const half *input, const size_t n, const size_t c,
                                                           const size_t in_d, const size_t in_h, const size_t in_w,
                                                           const size_t out_d, const size_t out_h, const size_t out_w,
                                                           const float d_scale, const float h_scale,
                                                           const float w_scale, const bool align_corner, half *output,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleTrilinear3D<float>(const float *input, const size_t n, const size_t c,
                                                            const size_t in_d, const size_t in_h, const size_t in_w,
                                                            const size_t out_d, const size_t out_h, const size_t out_w,
                                                            const float d_scale, const float h_scale,
                                                            const float w_scale, const bool align_corner, float *output,
                                                            cudaStream_t cuda_stream);
