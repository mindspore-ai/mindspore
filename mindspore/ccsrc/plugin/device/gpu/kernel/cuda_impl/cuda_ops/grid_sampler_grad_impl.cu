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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/grid_sampler_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template<typename T>
static __forceinline__ __device__
void safe_add_2d(T *data, int h, int w,
                 int sH, int sW, int H, int W,
                 T delta,
                 const size_t NC_offset) {
  if (within_bounds_2d(h, w, H, W)) {
    MsAtomicAdd(data + NC_offset + h * sH + w * sW, delta);
  }
}

template<typename T>
static __forceinline__ __device__
void add_value_bounded(
    T* data, T x, T y, int W, int H, int sW, int sH,
    T delta,
    GridSamplerPaddingMode padding_mode,
    bool align_corners,
    const size_t NC_offset) {
  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int ix = static_cast<int>(x);
  int iy = static_cast<int>(y);

  safe_add_2d(data, iy, ix, sH, sW, H, W, delta, NC_offset);
}

template<typename T>
static __forceinline__ __device__
void safe_add_3d(T *data, int d, int h, int w,
                 int sD, int sH, int sW, int D, int H, int W,
                 T delta,
                 const size_t NC_offset) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    MsAtomicAdd(data + NC_offset + d * sD + h * sH + w * sW, delta);
  }
}

template <typename T>
__global__ void GridSamplerGradInitKernel(const size_t size_init, T *dx) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size_init; index += blockDim.x * gridDim.x) {
    dx[index] = static_cast<T>(.0);
  }
}

template <typename T>
__inline__ __device__ T GetInput(const T *input, size_t index) { return input[index]; }
__inline__ __device__ float GetInput(const half *input, size_t index) { return __half2float(input[index]); }

template <typename T>
__global__ void GridSampler2DGradKernel(const size_t size, T *grad_addr, T *input_addr,
                                        T *grid_addr, T *dinput_addr, T *dgrid_addr,
                                        const size_t C, const size_t inp_H, const size_t inp_W,
                                        const size_t out_H, const size_t out_W,
                                        const size_t inp_sN, const size_t inp_sC,
                                        const size_t inp_sH, const size_t inp_sW,
                                        const size_t grid_sN, const size_t grid_sH,
                                        const size_t grid_sW, const size_t grid_sCoor,
                                        const size_t dinp_sN, const size_t dinp_sC,
                                        const size_t dinp_sH, const size_t dinp_sW,
                                        const size_t grad_sN, const size_t grad_sC,
                                        const size_t grad_sH, const size_t grad_sW,
                                        const size_t dgrid_sW, GridSamplerInterpolationMode interpolation_mode,
                                        GridSamplerPaddingMode padding_mode, bool align_corners) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x) {
    const size_t w = index % out_W;
    const size_t h = (index / out_W) % out_H;
    const size_t n = index / (out_H * out_W);
    const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    auto x = GetInput(grid_addr, grid_offset);
    auto y = GetInput(grid_addr, grid_offset + grid_sCoor);

    // ItmType is the intermediate type for computing.
    // If input type T is fp16, ItmType represents the upcasting type fp32 of T. Otherwise, im_type is the same as T.
    using ItmType = decltype(x);

    // multipliers for gradients on ix, and iy
    ItmType dix_mul, diy_mul;
    ItmType ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &dix_mul);
    ItmType iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &diy_mul);

    if (interpolation_mode == GridSamplerInterpolationMode::BILINEAR) {
      // get NE, NW, SE, SW pixel values from (x, y)
      int64_t ix_nw = static_cast<int64_t>(::floor(ix));
      int64_t iy_nw = static_cast<int64_t>(::floor(iy));
      int64_t ix_ne = ix_nw + 1;
      int64_t iy_ne = iy_nw;
      int64_t ix_sw = ix_nw;
      int64_t iy_sw = iy_nw + 1;
      int64_t ix_se = ix_nw + 1;
      int64_t iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      ItmType nw = (ix_se - ix)    * (iy_se - iy);
      ItmType ne = (ix    - ix_sw) * (iy_sw - iy);
      ItmType sw = (ix_ne - ix)    * (iy    - iy_ne);
      ItmType se = (ix    - ix_nw) * (iy    - iy_nw);

      ItmType dix = 0, diy = 0;
      size_t grad_idx_NCHW = n * grad_sN + h * grad_sH + w * grad_sW;
      size_t NC_offset = n * dinp_sN;
      T *inp_ptr_NC = input_addr + n * inp_sN;
      for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += dinp_sC, grad_idx_NCHW += grad_sC) {
        auto grad = GetInput(grad_addr, grad_idx_NCHW);

        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        safe_add_2d(dinput_addr, iy_nw, ix_nw, dinp_sH, dinp_sW, inp_H, inp_W, static_cast<T>(nw * grad), NC_offset);
        safe_add_2d(dinput_addr, iy_ne, ix_ne, dinp_sH, dinp_sW, inp_H, inp_W, static_cast<T>(ne * grad), NC_offset);
        safe_add_2d(dinput_addr, iy_sw, ix_sw, dinp_sH, dinp_sW, inp_H, inp_W, static_cast<T>(sw * grad), NC_offset);
        safe_add_2d(dinput_addr, iy_se, ix_se, dinp_sH, dinp_sW, inp_H, inp_W, static_cast<T>(se * grad), NC_offset);

        // calculate grad_grid
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          auto nw_val = GetInput(inp_ptr_NC, iy_nw * inp_sH + ix_nw * inp_sW);
          dix -= nw_val * (iy_se - iy) * grad;
          diy -= nw_val * (ix_se - ix) * grad;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          auto ne_val = GetInput(inp_ptr_NC, iy_ne * inp_sH + ix_ne * inp_sW);
          dix += ne_val * (iy_sw - iy) * grad;
          diy -= ne_val * (ix - ix_sw) * grad;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          auto sw_val = GetInput(inp_ptr_NC, iy_sw * inp_sH + ix_sw * inp_sW);
          dix -= sw_val * (iy - iy_ne) * grad;
          diy += sw_val * (ix_ne - ix) * grad;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          auto se_val = GetInput(inp_ptr_NC, iy_se * inp_sH + ix_se * inp_sW);
          dix += se_val * (iy - iy_nw) * grad;
          diy += se_val * (ix - ix_nw) * grad;
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with dgrid_sW to directly compute dgrid_ptr_NHW
      //   2. directly assign to dgrid_ptr_NHW[0], dgrid_ptr_NHW[1]
      T *dgrid_ptr_NHW = dgrid_addr + index * dgrid_sW;
      dgrid_ptr_NHW[0] = static_cast<T>(dix_mul * dix);
      dgrid_ptr_NHW[1] = static_cast<T>(diy_mul * diy);
    } else if (interpolation_mode == GridSamplerInterpolationMode::NEAREST) {
      int64_t ix_nearest = static_cast<int64_t>(::round(ix));
      int64_t iy_nearest = static_cast<int64_t>(::round(iy));

      // assign nearest neighbor pixel value to output pixel
      T *grad_ptr_NCHW = grad_addr + n * grad_sN + h * grad_sH + w * grad_sW;
      size_t NC_offset = n * dinp_sN;
      for (size_t c = 0; c < C; ++c, NC_offset += dinp_sC, grad_ptr_NCHW += grad_sC) {
        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        safe_add_2d(dinput_addr, iy_nearest, ix_nearest, dinp_sH, dinp_sW, inp_H, inp_W, *grad_ptr_NCHW, NC_offset);
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with dgrid_sW to directly compute dgrid_ptr_NHW
      //   2. directly assign to dgrid_ptr_NHW[0], dgrid_ptr_NHW[1]
      T *dgrid_ptr_NHW = dgrid_addr + index * dgrid_sW;
      dgrid_ptr_NHW[0] = static_cast<T>(0);
      dgrid_ptr_NHW[1] = static_cast<T>(0);
    } else if (interpolation_mode == GridSamplerInterpolationMode::BICUBIC) {
      ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &dix_mul);
      iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &diy_mul);

      ItmType ix_nw = ::floor(ix);
      ItmType iy_nw = ::floor(iy);

      const ItmType tx = ix - ix_nw;
      const ItmType ty = iy - iy_nw;

      ItmType x_coeffs[4];
      ItmType y_coeffs[4];
      ItmType x_coeffs_grad[4];
      ItmType y_coeffs_grad[4];

      get_cubic_upsampling_coefficients<ItmType>(x_coeffs, tx);
      get_cubic_upsampling_coefficients<ItmType>(y_coeffs, ty);
      get_cubic_coefficients_grad<ItmType>(x_coeffs_grad, tx);
      get_cubic_coefficients_grad<ItmType>(y_coeffs_grad, ty);

      ItmType dix = 0;
      ItmType diy = 0;

      size_t grad_idx_NCHW = n * grad_sN + h * grad_sH + w * grad_sW;
      size_t NC_offset = n * dinp_sN;
      T *inp_ptr_NC = input_addr + n * inp_sN;

      for (size_t c = 0; c < C; ++c, grad_idx_NCHW += grad_sC, NC_offset += dinp_sC, inp_ptr_NC+= inp_sC) {
        auto grad = GetInput(grad_addr, grad_idx_NCHW);

        #pragma unroll 4
        for (size_t i = 0; i < 4; ++i) {
          #pragma unroll 4
          for (size_t j = 0; j < 4; ++j) {
            // set input gradient. See Note [Passing pointer and offset to fastAtomicAdd].
            add_value_bounded<T>(dinput_addr, ix_nw - 1 + i, iy_nw - 1 + j, inp_W, inp_H, dinp_sW, dinp_sH,
              grad * x_coeffs[i] * y_coeffs[j],
              padding_mode,
              align_corners,
              NC_offset);

            // set grid gradient
            ItmType val = get_value_bounded<T>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
              inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners);

            dix -= val * x_coeffs_grad[i] * y_coeffs[j] * grad;
            diy -= val * y_coeffs_grad[j] * x_coeffs[i] * grad;
          }
        }
      }

      T *dgrid_ptr_NHW = dgrid_addr + index * dgrid_sW;
      dgrid_ptr_NHW[0] = static_cast<T>(dix_mul * dix);
      dgrid_ptr_NHW[1] = static_cast<T>(diy_mul * diy);
    }
  }
}

template <typename T>
void GridSampler2DGrad(const size_t size, const size_t dinput_size,
                       const size_t dgrid_size, T *grad_addr, T *input_addr,
                       T *grid_addr, T *dinput_addr, T *dgrid_addr,
                       const std::vector<size_t> &grad_shape,
                       const std::vector<size_t> &input_shape,
                       const std::vector<size_t> &grid_shape,
                       const std::vector<size_t> &dinput_shape,
                       const std::vector<size_t> &dgrid_shape,
                       const std::vector<size_t> &grad_stride,
                       const std::vector<size_t> &input_stride,
                       const std::vector<size_t> &grid_stride,
                       const std::vector<size_t> &dinput_stride,
                       const std::vector<size_t> &dgrid_stride,
                       const GridSamplerInterpolationMode interpolation_mode,
                       const GridSamplerPaddingMode padding_mode,
                       const bool align_corners,
                       cudaStream_t cuda_stream) {
  GridSamplerGradInitKernel<<<GET_BLOCKS(dinput_size), GET_THREADS_MAXSIZE(dinput_size), 0, cuda_stream>>>(
    dinput_size, dinput_addr);
  GridSamplerGradInitKernel<<<GET_BLOCKS(dgrid_size), GET_THREADS_MAXSIZE(dgrid_size), 0, cuda_stream>>>(
    dgrid_size, dgrid_addr);
  size_t thread_per_block = 256;
  size_t block_per_grid = (size + thread_per_block - 1) / thread_per_block;
  GridSampler2DGradKernel<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    size, grad_addr, input_addr, grid_addr, dinput_addr, dgrid_addr,
    input_shape[1], input_shape[2], input_shape[3],
    grid_shape[1], grid_shape[2],
    input_stride[0], input_stride[1], input_stride[2], input_stride[3],
    grid_stride[0], grid_stride[1], grid_stride[2], grid_stride[3],
    dinput_stride[0], dinput_stride[1], dinput_stride[2], dinput_stride[3],
    grad_stride[0], grad_stride[1], grad_stride[2], grad_stride[3],
    dgrid_stride[2], interpolation_mode, padding_mode, align_corners);
}

template CUDA_LIB_EXPORT void GridSampler2DGrad<half>(const size_t size, const size_t dinput_size,
                                                       const size_t dgrid_size, half *grad_addr, half *input_addr,
                                                       half *grid_addr, half *dinput_addr, half *dgrid_addr,
                                                       const std::vector<size_t> &grad_shape,
                                                       const std::vector<size_t> &input_shape,
                                                       const std::vector<size_t> &grid_shape,
                                                       const std::vector<size_t> &dinput_shape,
                                                       const std::vector<size_t> &dgrid_shape,
                                                       const std::vector<size_t> &grad_stride,
                                                       const std::vector<size_t> &input_stride,
                                                       const std::vector<size_t> &grid_stride,
                                                       const std::vector<size_t> &dinput_stride,
                                                       const std::vector<size_t> &dgrid_stride,
                                                       const GridSamplerInterpolationMode interpolation_mode,
                                                       const GridSamplerPaddingMode padding_mode,
                                                       const bool align_corners,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void GridSampler2DGrad<float>(const size_t size, const size_t dinput_size,
                                                       const size_t dgrid_size, float *grad_addr, float *input_addr,
                                                       float *grid_addr, float *dinput_addr, float *dgrid_addr,
                                                       const std::vector<size_t> &grad_shape,
                                                       const std::vector<size_t> &input_shape,
                                                       const std::vector<size_t> &grid_shape,
                                                       const std::vector<size_t> &dinput_shape,
                                                       const std::vector<size_t> &dgrid_shape,
                                                       const std::vector<size_t> &grad_stride,
                                                       const std::vector<size_t> &input_stride,
                                                       const std::vector<size_t> &grid_stride,
                                                       const std::vector<size_t> &dinput_stride,
                                                       const std::vector<size_t> &dgrid_stride,
                                                       const GridSamplerInterpolationMode interpolation_mode,
                                                       const GridSamplerPaddingMode padding_mode,
                                                       const bool align_corners,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void GridSampler2DGrad<double>(const size_t size, const size_t dinput_size,
                                                        const size_t dgrid_size, double *grad_addr, double *input_addr,
                                                        double *grid_addr, double *dinput_addr, double *dgrid_addr,
                                                        const std::vector<size_t> &grad_shape,
                                                        const std::vector<size_t> &input_shape,
                                                        const std::vector<size_t> &grid_shape,
                                                        const std::vector<size_t> &dinput_shape,
                                                        const std::vector<size_t> &dgrid_shape,
                                                        const std::vector<size_t> &grad_stride,
                                                        const std::vector<size_t> &input_stride,
                                                        const std::vector<size_t> &grid_stride,
                                                        const std::vector<size_t> &dinput_stride,
                                                        const std::vector<size_t> &dgrid_stride,
                                                        const GridSamplerInterpolationMode interpolation_mode,
                                                        const GridSamplerPaddingMode padding_mode,
                                                        const bool align_corners,
                                                        cudaStream_t cuda_stream);

template <typename T>
__global__ void GridSampler3DGradKernel(const size_t size, T *grad_addr, T *input_addr,
                                        T *grid_addr, T *dinput_addr, T *dgrid_addr,
                                        const size_t C, const size_t inp_D,
                                        const size_t inp_H, const size_t inp_W,
                                        const size_t out_D, const size_t out_H, const size_t out_W,
                                        const size_t inp_sN, const size_t inp_sC, const size_t inp_sD,
                                        const size_t inp_sH, const size_t inp_sW,
                                        const size_t grid_sN, const size_t grid_sD, const size_t grid_sH,
                                        const size_t grid_sW, const size_t grid_sCoor,
                                        const size_t dinp_sN, const size_t dinp_sC, const size_t dinp_sD,
                                        const size_t dinp_sH, const size_t dinp_sW,
                                        const size_t grad_sN, const size_t grad_sC, const size_t grad_sD,
                                        const size_t grad_sH, const size_t grad_sW,
                                        const size_t dgrid_sW, GridSamplerInterpolationMode interpolation_mode,
                                        GridSamplerPaddingMode padding_mode, bool align_corners) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x) {
    const size_t w = index % out_W;
    const size_t h = (index / out_W) % out_H;
    const size_t d = (index / (out_H * out_W)) % out_D;
    const size_t n = index / (out_D * out_H * out_W);
    const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z coordinates from grid
    auto x = GetInput(grid_addr, grid_offset);
    auto y = GetInput(grid_addr, grid_offset + grid_sCoor);
    auto z = GetInput(grid_addr, grid_offset + 2 * grid_sCoor);

    // ItmType is the intermediate type for computing.
    // If input type T is fp16, ItmType represents the upcasting type fp32 of T. Otherwise, im_type is the same as T.
    using ItmType = decltype(x);

    // multipliers for gradients on ix, iy, and iz
    ItmType dix_mul, diy_mul, diz_mul;
    ItmType ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &dix_mul);
    ItmType iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &diy_mul);
    ItmType iz = grid_sampler_compute_source_index_set_grad(z, inp_D, padding_mode, align_corners, &diz_mul);

    if (interpolation_mode == GridSamplerInterpolationMode::BILINEAR) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      int64_t ix_tnw = static_cast<int64_t>(::floor(ix));
      int64_t iy_tnw = static_cast<int64_t>(::floor(iy));
      int64_t iz_tnw = static_cast<int64_t>(::floor(iz));

      int64_t ix_tne = ix_tnw + 1;
      int64_t iy_tne = iy_tnw;
      int64_t iz_tne = iz_tnw;

      int64_t ix_tsw = ix_tnw;
      int64_t iy_tsw = iy_tnw + 1;
      int64_t iz_tsw = iz_tnw;

      int64_t ix_tse = ix_tnw + 1;
      int64_t iy_tse = iy_tnw + 1;
      int64_t iz_tse = iz_tnw;

      int64_t ix_bnw = ix_tnw;
      int64_t iy_bnw = iy_tnw;
      int64_t iz_bnw = iz_tnw + 1;

      int64_t ix_bne = ix_tnw + 1;
      int64_t iy_bne = iy_tnw;
      int64_t iz_bne = iz_tnw + 1;

      int64_t ix_bsw = ix_tnw;
      int64_t iy_bsw = iy_tnw + 1;
      int64_t iz_bsw = iz_tnw + 1;

      int64_t ix_bse = ix_tnw + 1;
      int64_t iy_bse = iy_tnw + 1;
      int64_t iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      ItmType tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
      ItmType tne = (ix - ix_bsw)    * (iy_bsw - iy)    * (iz_bsw - iz);
      ItmType tsw = (ix_bne - ix)    * (iy - iy_bne)    * (iz_bne - iz);
      ItmType tse = (ix - ix_bnw)    * (iy - iy_bnw)    * (iz_bnw - iz);
      ItmType bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
      ItmType bne = (ix - ix_tsw)    * (iy_tsw - iy)    * (iz - iz_tsw);
      ItmType bsw = (ix_tne - ix)    * (iy - iy_tne)    * (iz - iz_tne);
      ItmType bse = (ix - ix_tnw)    * (iy - iy_tnw)    * (iz - iz_tnw);

      ItmType dix = 0, diy = 0, diz = 0;
      size_t grad_idx_NCDHW = n * grad_sN + d * grad_sD + h * grad_sH + w * grad_sW;
      size_t NC_offset = n * dinp_sN;
      T *inp_ptr_NC = input_addr + n * inp_sN;
      // calculate bilinear weighted pixel value and set output pixel
      for (size_t c = 0; c < C; ++c, grad_idx_NCDHW += grad_sC, NC_offset += dinp_sC, inp_ptr_NC += inp_sC) {
        auto grad = GetInput(grad_addr, grad_idx_NCDHW);

        // calculate and set grad_input.
        safe_add_3d(dinput_addr, iz_tnw, iy_tnw, ix_tnw, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(tnw * grad), NC_offset);
        safe_add_3d(dinput_addr, iz_tne, iy_tne, ix_tne, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(tne * grad), NC_offset);
        safe_add_3d(dinput_addr, iz_tsw, iy_tsw, ix_tsw, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(tsw * grad), NC_offset);
        safe_add_3d(dinput_addr, iz_tse, iy_tse, ix_tse, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(tse * grad), NC_offset);
        safe_add_3d(dinput_addr, iz_bnw, iy_bnw, ix_bnw, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(bnw * grad), NC_offset);
        safe_add_3d(dinput_addr, iz_bne, iy_bne, ix_bne, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(bne * grad), NC_offset);
        safe_add_3d(dinput_addr, iz_bsw, iy_bsw, ix_bsw, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(bsw * grad), NC_offset);
        safe_add_3d(dinput_addr, iz_bse, iy_bse, ix_bse, dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W,
                    static_cast<T>(bse * grad), NC_offset);

        // calculate grad_grid
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          auto tnw_val = GetInput(inp_ptr_NC, iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW);
          dix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * grad;
          diy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * grad;
          diz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * grad;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          auto tne_val = GetInput(inp_ptr_NC, iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW);
          dix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * grad;
          diy -= tne_val * (ix - ix_bsw)    * (iz_bsw - iz)    * grad;
          diz -= tne_val * (ix - ix_bsw)    * (iy_bsw - iy)    * grad;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          auto tsw_val = GetInput(inp_ptr_NC, iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW);
          dix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * grad;
          diy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * grad;
          diz -= tsw_val * (ix_bne - ix)    * (iy - iy_bne)    * grad;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          auto tse_val = GetInput(inp_ptr_NC, iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW);
          dix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * grad;
          diy += tse_val * (ix - ix_bnw)    * (iz_bnw - iz)    * grad;
          diz -= tse_val * (ix - ix_bnw)    * (iy - iy_bnw)    * grad;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          auto bnw_val = GetInput(inp_ptr_NC, iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW);
          dix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * grad;
          diy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * grad;
          diz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * grad;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          auto bne_val = GetInput(inp_ptr_NC, iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW);
          dix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * grad;
          diy -= bne_val * (ix - ix_tsw)    * (iz - iz_tsw)    * grad;
          diz += bne_val * (ix - ix_tsw)    * (iy_tsw - iy)    * grad;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          auto bsw_val = GetInput(inp_ptr_NC, iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW);
          dix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * grad;
          diy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * grad;
          diz += bsw_val * (ix_tne - ix)    * (iy - iy_tne)    * grad;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          auto bse_val = GetInput(inp_ptr_NC, iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW);
          dix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * grad;
          diy += bse_val * (ix - ix_tnw)    * (iz - iz_tnw)    * grad;
          diz += bse_val * (ix - ix_tnw)    * (iy - iy_tnw)    * grad;
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with dgrid_sW to directly compute dgrid_ptr_NDHW
      //   2. directly assign to dgrid_ptr_NDHW[0], dgrid_ptr_NDHW[1], dgrid_ptr_NDHW[2]
      T *dgrid_ptr_NDHW = dgrid_addr + index * dgrid_sW;
      dgrid_ptr_NDHW[0] = static_cast<T>(dix_mul * dix);
      dgrid_ptr_NDHW[1] = static_cast<T>(diy_mul * diy);
      dgrid_ptr_NDHW[2] = static_cast<T>(diz_mul * diz);
    } else if (interpolation_mode == GridSamplerInterpolationMode::NEAREST) {
      auto ix_nearest = static_cast<int64_t>(::round(ix));
      auto iy_nearest = static_cast<int64_t>(::round(iy));
      auto iz_nearest = static_cast<int64_t>(::round(iz));

      // assign nearest neighbor pixel value to output pixel
      T *grad_ptr_NCDHW = grad_addr + n * grad_sN + d * grad_sD + h * grad_sH + w * grad_sW;
      size_t NC_offset = n * dinp_sN;
      for (size_t c = 0; c < C; ++c, grad_ptr_NCDHW += grad_sC, NC_offset += dinp_sC) {
        // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
        safe_add_3d(dinput_addr, iz_nearest, iy_nearest, ix_nearest,
                    dinp_sD, dinp_sH, dinp_sW, inp_D, inp_H, inp_W, *grad_ptr_NCDHW,
                    NC_offset);
      }
      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with dgrid_sW to directly compute dgrid_ptr_NDHW
      //   2. directly assign to dgrid_ptr_NDHW[0], dgrid_ptr_NDHW[1], dgrid_ptr_NDHW[2]
      T *dgrid_ptr_NDHW = dgrid_addr + index * dgrid_sW;
      dgrid_ptr_NDHW[0] = static_cast<T>(0);
      dgrid_ptr_NDHW[1] = static_cast<T>(0);
      dgrid_ptr_NDHW[2] = static_cast<T>(0);
    }
  }
}

template <typename T>
void GridSampler3DGrad(const size_t size, const size_t dinput_size,
                       const size_t dgrid_size, T *grad_addr, T *input_addr,
                       T *grid_addr, T *dinput_addr, T *dgrid_addr,
                       const std::vector<size_t> &grad_shape,
                       const std::vector<size_t> &input_shape,
                       const std::vector<size_t> &grid_shape,
                       const std::vector<size_t> &dinput_shape,
                       const std::vector<size_t> &dgrid_shape,
                       const std::vector<size_t> &grad_stride,
                       const std::vector<size_t> &input_stride,
                       const std::vector<size_t> &grid_stride,
                       const std::vector<size_t> &dinput_stride,
                       const std::vector<size_t> &dgrid_stride,
                       const GridSamplerInterpolationMode interpolation_mode,
                       const GridSamplerPaddingMode padding_mode,
                       const bool align_corners,
                       cudaStream_t cuda_stream) {
  GridSamplerGradInitKernel<<<GET_BLOCKS(dinput_size), GET_THREADS_MAXSIZE(dinput_size), 0, cuda_stream>>>(
    dinput_size, dinput_addr);
  GridSamplerGradInitKernel<<<GET_BLOCKS(dgrid_size), GET_THREADS_MAXSIZE(dgrid_size), 0, cuda_stream>>>(
    dgrid_size, dgrid_addr);
  size_t thread_per_block = 256;
  size_t block_per_grid = (size + thread_per_block - 1) / thread_per_block;
  GridSampler3DGradKernel<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    size, grad_addr, input_addr, grid_addr, dinput_addr, dgrid_addr,
    input_shape[1], input_shape[2], input_shape[3], input_shape[4],
    grid_shape[1], grid_shape[2], grid_shape[3],
    input_stride[0], input_stride[1], input_stride[2], input_stride[3], input_stride[4],
    grid_stride[0], grid_stride[1], grid_stride[2], grid_stride[3], grid_stride[4],
    dinput_stride[0], dinput_stride[1], dinput_stride[2], dinput_stride[3], dinput_stride[4],
    grad_stride[0], grad_stride[1], grad_stride[2], grad_stride[3], grad_stride[4],
    dgrid_stride[3], interpolation_mode, padding_mode, align_corners);
}

template CUDA_LIB_EXPORT void GridSampler3DGrad<half>(const size_t size, const size_t dinput_size,
                                                       const size_t dgrid_size, half *grad_addr, half *input_addr,
                                                       half *grid_addr, half *dinput_addr, half *dgrid_addr,
                                                       const std::vector<size_t> &grad_shape,
                                                       const std::vector<size_t> &input_shape,
                                                       const std::vector<size_t> &grid_shape,
                                                       const std::vector<size_t> &dinput_shape,
                                                       const std::vector<size_t> &dgrid_shape,
                                                       const std::vector<size_t> &grad_stride,
                                                       const std::vector<size_t> &input_stride,
                                                       const std::vector<size_t> &grid_stride,
                                                       const std::vector<size_t> &dinput_stride,
                                                       const std::vector<size_t> &dgrid_stride,
                                                       const GridSamplerInterpolationMode interpolation_mode,
                                                       const GridSamplerPaddingMode padding_mode,
                                                       const bool align_corners,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void GridSampler3DGrad<float>(const size_t size, const size_t dinput_size,
                                                       const size_t dgrid_size, float *grad_addr, float *input_addr,
                                                       float *grid_addr, float *dinput_addr, float *dgrid_addr,
                                                       const std::vector<size_t> &grad_shape,
                                                       const std::vector<size_t> &input_shape,
                                                       const std::vector<size_t> &grid_shape,
                                                       const std::vector<size_t> &dinput_shape,
                                                       const std::vector<size_t> &dgrid_shape,
                                                       const std::vector<size_t> &grad_stride,
                                                       const std::vector<size_t> &input_stride,
                                                       const std::vector<size_t> &grid_stride,
                                                       const std::vector<size_t> &dinput_stride,
                                                       const std::vector<size_t> &dgrid_stride,
                                                       const GridSamplerInterpolationMode interpolation_mode,
                                                       const GridSamplerPaddingMode padding_mode,
                                                       const bool align_corners,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void GridSampler3DGrad<double>(const size_t size, const size_t dinput_size,
                                                        const size_t dgrid_size, double *grad_addr, double *input_addr,
                                                        double *grid_addr, double *dinput_addr, double *dgrid_addr,
                                                        const std::vector<size_t> &grad_shape,
                                                        const std::vector<size_t> &input_shape,
                                                        const std::vector<size_t> &grid_shape,
                                                        const std::vector<size_t> &dinput_shape,
                                                        const std::vector<size_t> &dgrid_shape,
                                                        const std::vector<size_t> &grad_stride,
                                                        const std::vector<size_t> &input_stride,
                                                        const std::vector<size_t> &grid_stride,
                                                        const std::vector<size_t> &dinput_stride,
                                                        const std::vector<size_t> &dgrid_stride,
                                                        const GridSamplerInterpolationMode interpolation_mode,
                                                        const GridSamplerPaddingMode padding_mode,
                                                        const bool align_corners,
                                                        cudaStream_t cuda_stream);
