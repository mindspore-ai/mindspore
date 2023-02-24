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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

__inline__ __device__ float GetInput(const float *input, size_t index) { return input[index]; }
__inline__ __device__ float GetInput(const half *input, size_t index) { return static_cast<float>(input[index]); }

__inline__ __device__ float AreaPixelComputeSourceIndex(float scale, int dst_index, bool align_corners, bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    float src_idx = scale * (dst_index + 0.5) - 0.5;
    return (!cubic && src_idx < 0.0) ? 0.0 : src_idx;
  }
}

template <typename T>
__global__ void UpsampleTrilinear3DGradInitKernel(const size_t size_init, T *dx) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size_init; index += blockDim.x * gridDim.x) {
    dx[index] = static_cast<T>(.0);
  }
}

template <typename T>
__global__ void UpsampleTrilinear3DGrad(const T *grad, const size_t n, const size_t c, const size_t grad_d,
                                        const size_t grad_h, const size_t grad_w, const size_t grad_ncdhw,
                                        const size_t grad_cdhw, const size_t grad_dhw, const size_t grad_hw,
                                        const size_t dinput_d, const size_t dinput_h, const size_t dinput_w,
                                        const size_t dinput_cdhw, const size_t dinput_dhw, const size_t dinput_hw,
                                        const float d_scale, const float h_scale, const float w_scale,
                                        const bool align_corner, T *dinput) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < grad_ncdhw; pos += blockDim.x * gridDim.x) {
    const size_t posn = pos / grad_cdhw;
    const size_t posc = pos / grad_dhw % c;
    const size_t posd = pos / grad_hw % grad_d;
    const size_t posh = pos / grad_w % grad_h;
    const size_t posw = pos % grad_w;

    const float t1r = AreaPixelComputeSourceIndex(d_scale, posd, align_corner, false);
    const size_t t1 = floorf(t1r);
    const size_t t1p = (t1 < (dinput_d - 1)) ? 1 : 0;
    const float t1lambda = t1r - t1;
    const float t0lambda = 1.0f - t1lambda;

    const float h1r = AreaPixelComputeSourceIndex(h_scale, posh, align_corner, false);
    const size_t h1 = floorf(h1r);
    const size_t h1p = (h1 < (dinput_h - 1)) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.0f - h1lambda;

    const float w1r = AreaPixelComputeSourceIndex(w_scale, posw, align_corner, false);
    const size_t w1 = floorf(w1r);
    const size_t w1p = (w1 < (dinput_w - 1)) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.0f - w1lambda;

    // get required indices
    const size_t p1 = posn * dinput_cdhw + posc * dinput_dhw + t1 * dinput_hw + h1 * dinput_w + w1;
    const size_t p2 = posn * dinput_cdhw + posc * dinput_dhw + t1 * dinput_hw + h1 * dinput_w + (w1 + w1p);
    const size_t p3 = posn * dinput_cdhw + posc * dinput_dhw + t1 * dinput_hw + (h1 + h1p) * dinput_w + w1;
    const size_t p4 = posn * dinput_cdhw + posc * dinput_dhw + t1 * dinput_hw + (h1 + h1p) * dinput_w + (w1 + w1p);
    const size_t p5 = posn * dinput_cdhw + posc * dinput_dhw + (t1 + t1p) * dinput_hw + h1 * dinput_w + w1;
    const size_t p6 = posn * dinput_cdhw + posc * dinput_dhw + (t1 + t1p) * dinput_hw + h1 * dinput_w + (w1 + w1p);
    const size_t p7 = posn * dinput_cdhw + posc * dinput_dhw + (t1 + t1p) * dinput_hw + (h1 + h1p) * dinput_w + w1;
    const size_t p8 =
      posn * dinput_cdhw + posc * dinput_dhw + (t1 + t1p) * dinput_hw + (h1 + h1p) * dinput_w + (w1 + w1p);

    const float d2val = GetInput(grad, pos);

    MsAtomicAdd(dinput + p1, static_cast<T>(t0lambda * h0lambda * w0lambda * d2val));
    MsAtomicAdd(dinput + p2, static_cast<T>(t0lambda * h0lambda * w1lambda * d2val));
    MsAtomicAdd(dinput + p3, static_cast<T>(t0lambda * h1lambda * w0lambda * d2val));
    MsAtomicAdd(dinput + p4, static_cast<T>(t0lambda * h1lambda * w1lambda * d2val));
    MsAtomicAdd(dinput + p5, static_cast<T>(t1lambda * h0lambda * w0lambda * d2val));
    MsAtomicAdd(dinput + p6, static_cast<T>(t1lambda * h0lambda * w1lambda * d2val));
    MsAtomicAdd(dinput + p7, static_cast<T>(t1lambda * h1lambda * w0lambda * d2val));
    MsAtomicAdd(dinput + p8, static_cast<T>(t1lambda * h1lambda * w1lambda * d2val));
  }
  return;
}

template <typename T>
void CalUpsampleTrilinear3DGrad(const T *grad, const size_t n, const size_t c, const size_t grad_d, const size_t grad_h,
                                const size_t grad_w, const size_t dinput_d, const size_t dinput_h,
                                const size_t dinput_w, const float d_scale, const float h_scale, const float w_scale,
                                const bool align_corner, T *dinput, cudaStream_t cuda_stream) {
  const size_t dinput_hw = dinput_h * dinput_w;
  const size_t dinput_dhw = dinput_d * dinput_hw;
  const size_t dinput_cdhw = c * dinput_dhw;
  const size_t dinput_ncdhw = dinput_cdhw * n;

  const size_t grad_hw = grad_h * grad_w;
  const size_t grad_dhw = grad_d * grad_hw;
  const size_t grad_cdhw = c * grad_dhw;
  const size_t grad_ncdhw = n * grad_cdhw;

  UpsampleTrilinear3DGradInitKernel<<<GET_BLOCKS(dinput_ncdhw), GET_THREADS_MAXSIZE(dinput_ncdhw), 0, cuda_stream>>>(
    dinput_ncdhw, dinput);

  UpsampleTrilinear3DGrad<<<GET_BLOCKS(grad_ncdhw), GET_THREADS, 0, cuda_stream>>>(
    grad, n, c, grad_d, grad_h, grad_w, grad_ncdhw, grad_cdhw, grad_dhw, grad_hw, dinput_d, dinput_h, dinput_w,
    dinput_cdhw, dinput_dhw, dinput_hw, d_scale, h_scale, w_scale, align_corner, dinput);
  return;
}

template CUDA_LIB_EXPORT void CalUpsampleTrilinear3DGrad<half>(
  const half *grad, const size_t n, const size_t c, const size_t grad_d, const size_t grad_h, const size_t grad_w,
  const size_t dinput_d, const size_t dinput_h, const size_t dinput_w, const float d_scale, const float h_scale,
  const float w_scale, const bool align_corner, half *dinput, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalUpsampleTrilinear3DGrad<float>(
  const float *grad, const size_t n, const size_t c, const size_t grad_d, const size_t grad_h, const size_t grad_w,
  const size_t dinput_d, const size_t dinput_h, const size_t dinput_w, const float d_scale, const float h_scale,
  const float w_scale, const bool align_corner, float *dinput, cudaStream_t cuda_stream);
