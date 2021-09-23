/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

// check for existence in current padded array on X and Y dims
__inline__ __device__ bool range_check(int x, int y, int padded_width, int padded_height) {
  if (((x >= 0) && (x <= padded_width - 1)) && ((y >= 0) && (y <= padded_height - 1))) {
    return true;
  }
  return false;
}

// extract paddings from correct positions given variable paddings_arg size
__inline__ __device__ void extract_paddings(const int64_t *paddings_arg, int padd_dim, int64_t *extracted_paddings) {
  const int paddings_offset = MAX_PADDINGS - padd_dim;
  for (int i = 0; i < padd_dim; i++) {
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE] = paddings_arg[i * PADDING_SIZE];
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE + 1] = paddings_arg[i * PADDING_SIZE + 1];
  }
}

// for every position, first calculate position it mirrors from in the new padded array
// adjust calculated position to origin dx array dimensions and copy value
template <typename T>
__global__ void MirrorPad(const size_t size, const T *input, const int old_batch, const int old_channel,
                          const int old_height, const int old_width, const int padded_height, const int padded_width,
                          const int padd_dim, const int64_t *paddings_arg, int mode, T *output) {
  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // Create anchor points for non mirrored data inside new tensor
  int ap1_x = paddings[WIDTH + LEFT];
  int ap2_x = paddings[WIDTH + LEFT] + old_width - 1;
  int ap1_y = paddings[HEIGHT + TOP];
  int ap2_y = paddings[HEIGHT + TOP] + old_height - 1;
  int ap1_channel = paddings[CHANNEL + LEFT];
  int ap2_channel = paddings[CHANNEL + LEFT] + old_channel - 1;
  int ap1_batch = paddings[BATCH + LEFT];
  int ap2_batch = paddings[BATCH + LEFT] + old_batch - 1;
  int channels_new = old_channel + paddings[CHANNEL + LEFT] + paddings[CHANNEL + RIGHT];

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int block_num = (pos / padded_width) / padded_height;
    // cur position
    const int padded_x = pos % padded_width;
    const int padded_y = (pos / padded_width) % padded_height;
    const int padded_channel = block_num % channels_new;
    const int padded_batch = block_num / channels_new;
    // distance from anchor points
    // can be +/- depending on position
    int x_dist = 0;
    int y_dist = 0;
    int channel_dist = 0;
    int batch_dist = 0;

    // data to mirror from in new tensor dims
    int matchval_x_index = padded_x;
    int matchval_y_index = padded_y;
    int matchval_channel_index = padded_channel;
    int matchval_batch_index = padded_batch;
    int equiv_block_num = 0;

    // update matching index in original tensor across all 4 dims
    if ((padded_x < ap1_x) || (padded_x > ap2_x)) {
      x_dist = (padded_x < ap1_x) ? (ap1_x - padded_x) : (padded_x - ap2_x);
      matchval_x_index = (padded_x < ap1_x) ? (ap1_x + x_dist - mode) : (ap2_x - x_dist + mode);
    }
    if ((padded_y < ap1_y) || (padded_y > ap2_y)) {
      y_dist = (padded_y < ap1_y) ? (ap1_y - padded_y) : (padded_y - ap2_y);
      matchval_y_index = (padded_y < ap1_y) ? (ap1_y + y_dist - mode) : (ap2_y - y_dist + mode);
    }
    if ((padded_channel < ap1_channel) || (padded_channel > ap2_channel)) {
      channel_dist = (padded_channel < ap1_channel) ? (ap1_channel - padded_channel) : (padded_channel - ap2_channel);
      matchval_channel_index =
        (padded_channel < ap1_channel) ? (ap1_channel + channel_dist - mode) : (ap2_channel - channel_dist + mode);
    }
    if ((padded_batch < ap1_batch) || (padded_batch > ap2_batch)) {
      batch_dist = (padded_batch < ap1_batch) ? (ap1_batch - padded_batch) : (padded_batch - ap2_batch);
      matchval_batch_index =
        (padded_batch < ap1_batch) ? (ap1_batch + batch_dist - mode) : (ap2_batch - batch_dist + mode);
    }

    // calculate equivalent block in input
    equiv_block_num = ((matchval_batch_index - paddings[BATCH + LEFT]) * old_channel) +
                      (matchval_channel_index - paddings[CHANNEL + LEFT]);

    // copy data from equiv block and adjusted x and y values in unpadded tensor
    output[pos] = input[(equiv_block_num * old_height + matchval_y_index - paddings[HEIGHT + TOP]) * old_width +
                        matchval_x_index - paddings[WIDTH + LEFT]];
  }
}

// Accumulates mirrored values across batch and channels into an interim workspace array
// One thread for every output value and a sweeping add logic allows kernel to avoid using
// slower locked based atomic adds
template <typename T>
__global__ void MirrorPadGradBatchChannel(const size_t size, T *dy, T *interim_dy, const int dx_batches,
                                          const int dx_channels, const int dx_height, const int dx_width,
                                          const int dy_height, const int dy_width, const int padd_dim,
                                          const int64_t *paddings_arg, int mode, T *dx) {
  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;  // init all to 0
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // Create anchor points for non mirrored data inside new tensor
  int ap1_channel = paddings[CHANNEL + LEFT];
  int ap2_channel = paddings[CHANNEL + LEFT] + dx_channels - 1;
  int ap1_batch = paddings[BATCH + LEFT];
  int ap2_batch = paddings[BATCH + LEFT] + dx_batches - 1;
  int dy_channels = dx_channels + paddings[CHANNEL + LEFT] + paddings[CHANNEL + RIGHT];
  int dy_batches = dx_batches + paddings[BATCH + LEFT] + paddings[BATCH + RIGHT];

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int block_num = (pos / dy_width) / dy_height;
    // Select exact position inside the dy_interim array
    const int interim_x = pos % dy_width;
    const int interim_y = (pos / dy_width) % dy_height;
    const int interim_channel = block_num % dx_channels;
    const int interim_batch = block_num / dx_channels;
    interim_dy[pos] = 0;  // init
    // map cur interim channel and batch to equivalent in padded dy array
    const int equiv_dy_channel = interim_channel + paddings[CHANNEL + LEFT];
    const int equiv_dy_batch = interim_batch + paddings[BATCH + LEFT];
    int target_batch = 0;
    int target_channel = 0;
    int equiv_block_num = 0;
    equiv_block_num = ((equiv_dy_batch * dy_channels) + equiv_dy_channel);
    // generate values to sweep over all possible mirrored points
    auto batch_offsets = {2 * (ap1_batch - equiv_dy_batch) - mode, 0, 2 * (ap2_batch - equiv_dy_batch) + mode};
    auto channel_offsets = {2 * (ap1_channel - equiv_dy_channel) - mode, 0,
                            2 * (ap2_channel - equiv_dy_channel) + mode};
    for (auto b_adjust : batch_offsets) {
      for (auto c_adjust : channel_offsets) {
        target_batch = equiv_dy_batch + b_adjust;
        target_channel = equiv_dy_channel + c_adjust;
        // bounds check - if within bounds, mirrored value exists - copy dy
        if ((target_batch < 0) || (target_batch > (dy_batches - 1)) || (target_channel < 0) ||
            (target_channel > (dy_channels - 1))) {
          continue;  // no mirrored value with these target values
        }
        equiv_block_num = ((target_batch * dy_channels) + target_channel);
        // Copy data and set value at input to 0 to avoid duplicates in reflect mode
        interim_dy[pos] = interim_dy[pos] + dy[(equiv_block_num * dy_height + interim_y) * dy_width + interim_x];
        dy[(equiv_block_num * dy_height + interim_y) * dy_width + interim_x] = 0;
      }
    }
  }
}

// Accumulate mirrored values across width and height from the interim dy array into output array
// Similar sweep logic again allows for a no lock based logic
template <typename T>
__global__ void MirrorPadGrad_Width_Height(const size_t size, const T *dy, T *interim_dy, const int dx_batches,
                                           const int dx_channels, const int dx_height, const int dx_width,
                                           const int dy_height, const int dy_width, const int padd_dim,
                                           const int64_t *paddings_arg, int mode, T *dx) {
  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;  // init all to 0
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // Create required anchor points for non-mirrored data inside new tensor
  int ap1_x = paddings[WIDTH + LEFT];
  int ap2_x = paddings[WIDTH + LEFT] + dx_width - 1;
  int ap1_y = paddings[HEIGHT + TOP];
  int ap2_y = paddings[HEIGHT + TOP] + dx_height - 1;

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int dx_block_num = (pos / dx_width) / dx_height;
    const int grad_x = (pos % dx_width) + paddings[WIDTH + LEFT];
    const int grad_y = ((pos / dx_width) % dx_height) + paddings[HEIGHT + TOP];
    // copy position's own value into output
    dx[pos] = interim_dy[(dx_block_num * dy_height + grad_y) * dy_width + grad_x];

    int x_dist_1 = (ap1_x - grad_x - mode);
    int y_dist_1 = (ap1_y - grad_y - mode);
    int x_dist_2 = (ap2_x - grad_x + mode);
    int y_dist_2 = (ap2_y - grad_y + mode);
    int axis_dist[] = {x_dist_1, x_dist_2, y_dist_1, y_dist_2};
    int anch_point[] = {ap1_x, ap2_x, ap1_y, ap2_y};
    bool x_axis_check[] = {true, true, false, false};  // true - update X , false - update Y

    int temp_x = 0;
    int temp_y = 0;
    // mirroring in axis lines
    for (int x = 0; x < 4; x++) {
      if (axis_dist[x] != 0) {
        if (x_axis_check[x]) {
          temp_y = grad_y;
          temp_x = anch_point[x] + axis_dist[x];
        } else {
          temp_x = grad_x;
          temp_y = anch_point[x] + axis_dist[x];
        }
        if (range_check(temp_x, temp_y, dy_width, dy_height)) {
          dx[pos] = dx[pos] + interim_dy[(dx_block_num * dy_height + temp_y) * dy_width + temp_x];
        }
      }
    }
    // mirroring at corners
    for (int x = 0; x < 2; x++) {
      for (int y = 2; y < 4; y++) {
        if ((axis_dist[x] != 0) && (axis_dist[y] != 0)) {
          temp_x = anch_point[x] + axis_dist[x];
          temp_y = anch_point[y] + axis_dist[y];
          if (range_check(temp_x, temp_y, dy_width, dy_height)) {
            dx[pos] = dx[pos] + interim_dy[(dx_block_num * dy_height + temp_y) * dy_width + temp_x];
          }
        }
      }
    }
  }
}

template <typename T>
void CalMirrorPad(const size_t size, const T *input, const int old_batch, const int old_channel, const int old_height,
                  const int old_width, const int padded_height, const int padded_width, int padd_num,
                  const int64_t *paddings, const int mode, T *output, cudaStream_t cuda_stream) {
  MirrorPad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, old_batch, old_channel, old_height,
                                                               old_width, padded_height, padded_width, padd_num,
                                                               paddings, mode, output);
}
template <typename T>
void CalMirrorPadGrad(const size_t dx_size, const size_t interim_dy_size, T *dy, T *interim_dy, const int dx_batches,
                      const int dx_channels, const int dx_height, const int dx_width, const int dy_height,
                      const int dy_width, const int padd_dim, const int64_t *paddings, int mode, T *dx,
                      cudaStream_t cuda_stream) {
  MirrorPadGradBatchChannel<<<GET_BLOCKS(interim_dy_size), GET_THREADS, 0, cuda_stream>>>(
    interim_dy_size, dy, interim_dy, dx_batches, dx_channels, dx_height, dx_width, dy_height, dy_width, padd_dim,
    paddings, mode, dx);
  MirrorPadGrad_Width_Height<<<GET_BLOCKS(dx_size), GET_THREADS, 0, cuda_stream>>>(
    dx_size, dy, interim_dy, dx_batches, dx_channels, dx_height, dx_width, dy_height, dy_width, padd_dim, paddings,
    mode, dx);
}

template void CalMirrorPad<float>(const size_t size, const float *input, const int old_batch, const int old_channel,
                                  const int old_height, const int old_width, const int padded_height,
                                  const int padded_width, int padd_num, const int64_t *paddings, int mode,
                                  float *output, cudaStream_t cuda_stream);
template void CalMirrorPad<half>(const size_t size, const half *input, const int old_batch, const int old_channel,
                                 const int old_height, const int old_width, const int padded_height,
                                 const int padded_width, int padd_num, const int64_t *paddings, int mode, half *output,
                                 cudaStream_t cuda_stream);
template void CalMirrorPad<int>(const size_t size, const int *input, const int old_batch, const int old_channel,
                                const int old_height, const int old_width, const int padded_height,
                                const int padded_width, int padd_num, const int64_t *paddings, int mode, int *output,
                                cudaStream_t cuda_stream);
template void CalMirrorPadGrad<float>(const size_t dx_size, const size_t dy_size, float *dy, float *interim_dy,
                                      const int dx_batches, const int dx_channels, const int dx_height,
                                      const int dx_width, const int dy_height, const int dy_width, const int padd_dim,
                                      const int64_t *paddings, int mode, float *dx, cudaStream_t cuda_stream);
template void CalMirrorPadGrad<half>(const size_t dx_size, const size_t dy_size, half *dy, half *interim_dy,
                                     const int dx_batches, const int dx_channels, const int dx_height,
                                     const int dx_width, const int dy_height, const int dy_width, const int padd_dim,
                                     const int64_t *paddings, int mode, half *dx, cudaStream_t cuda_stream);
template void CalMirrorPadGrad<int>(const size_t dx_size, const size_t dy_size, int *dy, int *interim_dy,
                                    const int dx_batches, const int dx_channels, const int dx_height,
                                    const int dx_width, const int dy_height, const int dy_width, const int padd_dim,
                                    const int64_t *paddings, int mode, int *dx, cudaStream_t cuda_stream);
