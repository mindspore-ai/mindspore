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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/grid_sampler_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__inline__ __device__ T GetInput(const T *input, size_t index) {
  return input[index];
}
__inline__ __device__ float GetInput(const half *input, size_t index) { return __half2float(input[index]); }

template <typename T>
__global__ void GridSampler2DKernel(const size_t size, const T *input_addr, const T *grid_addr, T *output_addr,
                                    const size_t C, const size_t inp_H, const size_t inp_W, const size_t out_H,
                                    const size_t out_W, const size_t inp_sN, const size_t inp_sC, const size_t inp_sH,
                                    const size_t inp_sW, const size_t grid_sN, const size_t grid_sH,
                                    const size_t grid_sW, const size_t grid_sCoor, const size_t out_sN,
                                    const size_t out_sC, const size_t out_sH, const size_t out_sW,
                                    GridSamplerInterpolationMode interpolation_mode,
                                    GridSamplerPaddingMode padding_mode, bool align_corners) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x) {
    const size_t w = index % out_W;
    const size_t h = (index / out_W) % out_H;
    const size_t n = index / (out_H * out_W);
    const size_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    auto x = GetInput(grid_addr, grid_offset);
    auto y = GetInput(grid_addr, grid_offset + grid_sCoor);

    // ItmType is the intermediate type for computing.
    // If input type T is fp16, ItmType represents the upcasting type fp32 of T. Otherwise, im_type is the same as T.
    using ItmType = decltype(x);

    ItmType ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
    ItmType iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

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
      ItmType nw = (ix_se - ix) * (iy_se - iy);
      ItmType ne = (ix - ix_sw) * (iy_sw - iy);
      ItmType sw = (ix_ne - ix) * (iy - iy_ne);
      ItmType se = (ix - ix_nw) * (iy - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input_addr + n * inp_sN;
      auto out_ptr_NCHW = output_addr + n * out_sN + h * out_sH + w * out_sW;
      for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        ItmType intermediate_value = 0;
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iy_nw * inp_sH + ix_nw * inp_sW) * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iy_ne * inp_sH + ix_ne * inp_sW) * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iy_sw * inp_sH + ix_sw * inp_sW) * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iy_se * inp_sH + ix_se * inp_sW) * se;
        }
        *out_ptr_NCHW = static_cast<T>(intermediate_value);
      }
    } else if (interpolation_mode == GridSamplerInterpolationMode::NEAREST) {
      int64_t ix_nearest = static_cast<int64_t>(::round(ix));
      int64_t iy_nearest = static_cast<int64_t>(::round(iy));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input_addr + n * inp_sN;
      auto out_ptr_NCHW = output_addr + n * out_sN + h * out_sH + w * out_sW;
      for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<T>(0);
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolationMode::BICUBIC) {
      ix = grid_sampler_unnormalize(x, inp_W, align_corners);
      iy = grid_sampler_unnormalize(y, inp_H, align_corners);

      ItmType ix_nw = ::floor(ix);
      ItmType iy_nw = ::floor(iy);

      const ItmType tx = ix - ix_nw;
      const ItmType ty = iy - iy_nw;

      auto inp_ptr_NC = input_addr + n * inp_sN;
      auto out_ptr_NCHW = output_addr + n * out_sN + h * out_sH + w * out_sW;
      for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        T coefficients[4];

        for (size_t i = 0; i < 4; ++i) {
          coefficients[i] = cubic_interp1d(get_value_bounded<T>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H,
                                                                inp_sW, inp_sH, padding_mode, align_corners),
                                           get_value_bounded<T>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H,
                                                                inp_sW, inp_sH, padding_mode, align_corners),
                                           get_value_bounded<T>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H,
                                                                inp_sW, inp_sH, padding_mode, align_corners),
                                           get_value_bounded<T>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H,
                                                                inp_sW, inp_sH, padding_mode, align_corners),
                                           tx);
        }

        *out_ptr_NCHW = cubic_interp1d(coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty);
      }
    }
  }
}

template <typename T>
cudaError_t GridSampler2D(const size_t size, const T *input_addr, const T *grid_addr, T *output_addr,
                          const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                          const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                          const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                          const GridSamplerInterpolationMode interpolation_mode,
                          const GridSamplerPaddingMode padding_mode, const bool align_corners,
                          cudaStream_t cuda_stream) {
  size_t thread_per_block = 256;
  size_t block_per_grid = (size + thread_per_block - 1) / thread_per_block;
  GridSampler2DKernel<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    size, input_addr, grid_addr, output_addr, input_shape[1], input_shape[2], input_shape[3], grid_shape[1],
    grid_shape[2], input_stride[0], input_stride[1], input_stride[2], input_stride[3], grid_stride[0], grid_stride[1],
    grid_stride[2], grid_stride[3], output_stride[0], output_stride[1], output_stride[2], output_stride[3],
    interpolation_mode, padding_mode, align_corners);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t
GridSampler2D<half>(const size_t size, const half *input_addr, const half *grid_addr, half *output_addr,
                    const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                    const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                    const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                    const GridSamplerInterpolationMode interpolation_mode, const GridSamplerPaddingMode padding_mode,
                    const bool align_corners, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
GridSampler2D<float>(const size_t size, const float *input_addr, const float *grid_addr, float *output_addr,
                     const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                     const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                     const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                     const GridSamplerInterpolationMode interpolation_mode, const GridSamplerPaddingMode padding_mode,
                     const bool align_corners, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
GridSampler2D<double>(const size_t size, const double *input_addr, const double *grid_addr, double *output_addr,
                      const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                      const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                      const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                      const GridSamplerInterpolationMode interpolation_mode, const GridSamplerPaddingMode padding_mode,
                      const bool align_corners, cudaStream_t cuda_stream);

template <typename T>
__global__ void GridSampler3DKernel(const size_t size, const T *input_addr, const T *grid_addr, T *output_addr,
                                    const size_t C, const size_t inp_D, const size_t inp_H, const size_t inp_W,
                                    const size_t out_D, const size_t out_H, const size_t out_W, const size_t inp_sN,
                                    const size_t inp_sC, const size_t inp_sD, const size_t inp_sH, const size_t inp_sW,
                                    const size_t grid_sN, const size_t grid_sD, const size_t grid_sH,
                                    const size_t grid_sW, const size_t grid_sCoor, const size_t out_sN,
                                    const size_t out_sC, const size_t out_sD, const size_t out_sH, const size_t out_sW,
                                    GridSamplerInterpolationMode interpolation_mode,
                                    GridSamplerPaddingMode padding_mode, bool align_corners) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x) {
    const size_t w = index % out_W;
    const size_t h = (index / out_W) % out_H;
    const size_t d = (index / (out_H * out_W)) % out_D;
    const size_t n = index / (out_D * out_H * out_W);
    const size_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z coordinates from grid
    auto x = GetInput(grid_addr, grid_offset);
    auto y = GetInput(grid_addr, grid_offset + grid_sCoor);
    auto z = GetInput(grid_addr, grid_offset + 2 * grid_sCoor);

    // ItmType is the intermediate type for computing.
    // If input type T is fp16, ItmType represents the upcasting type fp32 of T. Otherwise, im_type is the same as T.
    using ItmType = decltype(x);

    ItmType ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
    ItmType iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);
    ItmType iz = grid_sampler_compute_source_index(z, inp_D, padding_mode, align_corners);

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
      ItmType tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      ItmType tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      ItmType tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      ItmType tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      ItmType bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      ItmType bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      ItmType bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      ItmType bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      auto inp_ptr_NC = input_addr + n * inp_sN;
      auto out_ptr_NCDHW = output_addr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
        ItmType intermediate_value = 0;
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW) * tnw;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW) * tne;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW) * tsw;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW) * tse;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW) * bnw;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW) * bne;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW) * bsw;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          intermediate_value += GetInput(inp_ptr_NC, iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW) * bse;
        }
        *out_ptr_NCDHW = static_cast<T>(intermediate_value);
      }
    } else if (interpolation_mode == GridSamplerInterpolationMode::NEAREST) {
      int64_t ix_nearest = static_cast<int64_t>(::round(ix));
      int64_t iy_nearest = static_cast<int64_t>(::round(iy));
      int64_t iz_nearest = static_cast<int64_t>(::round(iz));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input_addr + n * inp_sN;
      auto out_ptr_NCDHW = output_addr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (size_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCDHW = static_cast<T>(0);
        }
      }
    }
  }
}

template <typename T>
cudaError_t GridSampler3D(const size_t size, const T *input_addr, const T *grid_addr, T *output_addr,
                          const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                          const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                          const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                          const GridSamplerInterpolationMode interpolation_mode,
                          const GridSamplerPaddingMode padding_mode, const bool align_corners,
                          cudaStream_t cuda_stream) {
  size_t thread_per_block = 256;
  size_t block_per_grid = (size + thread_per_block - 1) / thread_per_block;
  GridSampler3DKernel<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    size, input_addr, grid_addr, output_addr, input_shape[1], input_shape[2], input_shape[3], input_shape[4],
    grid_shape[1], grid_shape[2], grid_shape[3], input_stride[0], input_stride[1], input_stride[2], input_stride[3],
    input_stride[4], grid_stride[0], grid_stride[1], grid_stride[2], grid_stride[3], grid_stride[4], output_stride[0],
    output_stride[1], output_stride[2], output_stride[3], output_stride[4], interpolation_mode, padding_mode,
    align_corners);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t
GridSampler3D<half>(const size_t size, const half *input_addr, const half *grid_addr, half *output_addr,
                    const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                    const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                    const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                    const GridSamplerInterpolationMode interpolation_mode, const GridSamplerPaddingMode padding_mode,
                    const bool align_corners, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t
GridSampler3D<float>(const size_t size, const float *input_addr, const float *grid_addr, float *output_addr,
                     const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                     const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                     const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                     const GridSamplerInterpolationMode interpolation_mode, const GridSamplerPaddingMode padding_mode,
                     const bool align_corners, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t
GridSampler3D<double>(const size_t size, const double *input_addr, const double *grid_addr, double *output_addr,
                      const std::vector<size_t> &input_shape, const std::vector<size_t> &grid_shape,
                      const std::vector<size_t> &output_shape, const std::vector<size_t> &input_stride,
                      const std::vector<size_t> &grid_stride, const std::vector<size_t> &output_stride,
                      const GridSamplerInterpolationMode interpolation_mode, const GridSamplerPaddingMode padding_mode,
                      const bool align_corners, cudaStream_t cuda_stream);
