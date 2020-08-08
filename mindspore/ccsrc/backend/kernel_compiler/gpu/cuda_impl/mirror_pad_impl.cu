/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include "backend/kernel_compiler/gpu/cuda_impl/mirror_pad_impl.cuh"

__inline__ __device__ bool range_check(int x, int y, int padded_width, int padded_height) {
  // check for existence in current padded array
  if (((x >= 0) && (x <= padded_width - 1)) && ((y >= 0) && (y <= padded_height - 1))) {
    return true;
  }
  return false;
}

template <typename T>
__global__ void MirrorPad(const size_t size, const T *input, const int num, const int channels, const int old_height,
                          const int old_width, const int padded_height, const int padded_width, const int padd_dim,
                          const int *paddings, int mode, T *output) {
  int padd_offset = 4 * (padd_dim - 2);
  int pad_left_ = paddings[padd_offset + 4];
  int pad_top_ = paddings[padd_offset + 0];

  // Create anchor points for old tensor positions inside new tensor
  int ap1_x = pad_left_;
  int ap1_y = pad_top_;
  int ap2_x = pad_left_ + old_width - 1;
  int ap2_y = pad_top_ + old_height - 1;

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int block_num = (pos / padded_width) / padded_height;
    const int padded_x = pos % padded_width;
    const int padded_y = (pos / padded_width) % padded_height;

    // distance to move from anchor point
    int x_dist = 0;
    int y_dist = 0;

    // x,y value to mirror in new tenspr
    int matchval_x_index = padded_x;
    int matchval_y_index = padded_y;

    if (padded_y - pad_top_ < 0 || padded_x - pad_left_ < 0 || padded_y - pad_top_ >= old_height ||
        padded_x - pad_left_ >= old_width) {
      if ((padded_x < ap1_x) || (padded_x > ap2_x)) {
        x_dist = (padded_x < ap1_x) ? (ap1_x - padded_x) : (padded_x - ap2_x);  // GEN DIST
        matchval_x_index = (padded_x < ap1_x) ? (ap1_x + x_dist - mode) : (ap2_x - x_dist + mode);
      }
      if ((padded_y < ap1_y) || (padded_y > ap2_y)) {
        y_dist = (padded_y < ap1_y) ? (ap1_y - padded_y) : (padded_y - ap2_y);
        matchval_y_index = (padded_y < ap1_y) ? (ap1_y + y_dist - mode) : (ap2_y - y_dist + mode);
      }
      output[pos] =
        input[(block_num * old_height + matchval_y_index - pad_top_) * old_width + matchval_x_index - pad_left_];
    } else {
      // existing values remain the same
      output[pos] = input[(block_num * old_height + padded_y - pad_top_) * old_width + padded_x - pad_left_];
    }
  }
  return;
}

template <typename T>
__global__ void MirrorPadGrad(const size_t size, const T *dy, const int num, const int channels,
                              const int padded_height, const int padded_width, const int old_height,
                              const int old_width, const int padd_dim, const int *paddings, int mode, T *dx) {
  int padd_offset = 4 * (padd_dim - 2);
  int pad_left_ = paddings[padd_offset + 4];
  int pad_top_ = paddings[padd_offset + 0];

  // Create anchor points for positions in the dy array
  int ap1_x = pad_left_;
  int ap1_y = pad_top_;
  int ap2_x = pad_left_ + old_width - 1;
  int ap2_y = pad_top_ + old_height - 1;

  int adjust = 0;  // adjust dist from reflection axis for symmetric padding
  if (mode == 1) {
    adjust = 1;
  }

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int block_num = (pos / old_width) / old_height;

    // refer to indices of original values inside padded array
    const int padded_x = (pos % old_width) + pad_left_;
    const int padded_y = ((pos / old_width) % old_height) + pad_top_;

    // copy positions own value into output
    dx[pos] = dx[pos] + dy[(block_num * padded_height + padded_y) * padded_width + padded_x];

    int x_dist_1 = (ap1_x - padded_x - adjust);
    int y_dist_1 = (ap1_y - padded_y - adjust);
    int x_dist_2 = (ap2_x - padded_x + adjust);
    int y_dist_2 = (ap2_y - padded_y + adjust);

    int axis_dist[] = {x_dist_1, x_dist_2, y_dist_1, y_dist_2};
    int anch_point[] = {ap1_x, ap2_x, ap1_y, ap2_y};
    bool x_axis_check[] = {true, true, false, false};  // true - update X , false - update Y

    int temp_x = 0;
    int temp_y = 0;

    // mirroring in axis lines
    for (int x = 0; x < 4; x++) {
      if (axis_dist[x] != 0) {
        if (x_axis_check[x]) {
          temp_y = padded_y;
          temp_x = anch_point[x] + axis_dist[x];
        } else {
          temp_x = padded_x;
          temp_y = anch_point[x] + axis_dist[x];
        }
        if (range_check(temp_x, temp_y, padded_width, padded_height)) {
          dx[pos] = dx[pos] + dy[(block_num * padded_height + temp_y) * padded_width + temp_x];
        }
      }
    }

    // mirroring at corners
    for (int x = 0; x < 2; x++) {
      for (int y = 2; y < 4; y++) {
        if ((axis_dist[x] != 0) && (axis_dist[y] != 0)) {
          temp_x = anch_point[x] + axis_dist[x];
          temp_y = anch_point[y] + axis_dist[y];
          if (range_check(temp_x, temp_y, padded_width, padded_height)) {
            dx[pos] = dx[pos] + dy[(block_num * padded_height + temp_y) * padded_width + temp_x];
          }
        }
      }
    }
  }
  return;
}

template <typename T>
void CalMirrorPad(const size_t size, const T *input, const int num, const int channels, const int old_height,
                  const int old_width, const int padded_height, const int padded_width, int padd_num,
                  const int *paddings, const int mode, T *output, cudaStream_t cuda_stream) {
  MirrorPad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, input, num, channels, old_height, old_width, padded_height, padded_width, padd_num, paddings, mode, output);
  return;
}

template <typename T>
void CalMirrorPadGrad(const size_t size, const T *dy, const int num, const int channels, const int padded_height,
                      const int padded_width, const int old_height, const int old_width, const int padd_dim,
                      const int *paddings, int mode, T *dx, cudaStream_t cuda_stream) {
  MirrorPadGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, channels, padded_height, padded_width,
                                                                   old_height, old_width, padd_dim, paddings, mode, dx);
  return;
}

template void CalMirrorPad<float>(const size_t size, const float *input, const int num, const int channels,
                                  const int old_height, const int old_width, const int padded_height,
                                  const int padded_width, int padd_num, const int *paddings, int mode, float *output,
                                  cudaStream_t cuda_stream);
template void CalMirrorPadGrad<float>(const size_t size, const float *dy, const int num, const int channels,
                                      const int old_height, const int old_width, const int padded_height,
                                      const int padded_width, const int padd_dim, const int *paddings, int mode,
                                      float *dx, cudaStream_t cuda_stream);
template void CalMirrorPad<half>(const size_t size, const half *input, const int num, const int channels,
                                 const int old_height, const int old_width, const int padded_height,
                                 const int padded_width, int padd_num, const int *paddings, int mode, half *output,
                                 cudaStream_t cuda_stream);
template void CalMirrorPadGrad<half>(const size_t size, const half *dy, const int num, const int channels,
                                     const int old_height, const int old_width, const int padded_height,
                                     const int padded_width, const int padd_dim, const int *paddings, int mode,
                                     half *dx, cudaStream_t cuda_stream);
