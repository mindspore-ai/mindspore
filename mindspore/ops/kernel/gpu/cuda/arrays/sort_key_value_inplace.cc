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

#include "kernel/gpu/cuda/arrays/sort_key_value_inplace.h"
#include <vector>
#include "kernel/gpu/cuda_impl/cuda_ops/sort_fixed_size.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/broadcast_to_impl.cuh"
#include "kernel/gpu/cuda/math/broadcast_public.h"
#include "kernel/gpu/gpu_common.h"

constexpr int MAX_DIMS = 8;

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
static uint64_t NextHighestPowerOf2(uint64_t n) {
  const int pow0of2 = 1;
  const int pow1of2 = 2;
  const int pow2of2 = 4;
  const int pow3of2 = 8;
  const int pow4of2 = 16;
  const int pow5of2 = 32;
  n--;
  n |= n >> pow0of2;
  n |= n >> pow1of2;
  n |= n >> pow2of2;
  n |= n >> pow3of2;
  n |= n >> pow4of2;
#ifndef _MSC_VER
  n |= n >> pow5of2;
#endif
  n++;

  return n;
}

template <typename K, typename V>
bool SegSort(const int key_dims, const TensorLayoutHelper &key_info, K *key_data, int64_t key_slices,
             int64_t key_slice_size, int64_t key_slice_stride, const TensorLayoutHelper &value_info, V *value_data,
             int64_t value_slice_stride, bool descending, cudaStream_t stream) {
  int64_t ceil_power_of2 = NextHighestPowerOf2(key_slice_size);

#define HANDLE_CASE(SIZE, ITEMS_PER_THREAD, STATUS)                                                                  \
  STATUS = SortFixedSize<SIZE, ITEMS_PER_THREAD, K, V>(key_dims, key_info, key_data, key_slices, key_slice_size,     \
                                                       key_slice_stride, value_info, value_data, value_slice_stride, \
                                                       descending, stream);
  constexpr int kFixedSizeLevel3SubThreshold1 = 512;
  constexpr int kFixedSizeLevel3SubThreshold2 = 256;
  constexpr int kFixedSizeLevel4SubThreshold = 64;
  constexpr int kFixedSizeLevel5SubThreshold1 = 16;
  constexpr int kFixedSizeLevel5SubThreshold2 = 8;
  constexpr int kFixedSizeLevel5SubThreshold3 = 4;
  constexpr int kFixedSizeLevel5SubThreshold4 = 2;
  cudaError_t status = cudaErrorNotReady;
  switch (ceil_power_of2) {
    case kFixedSizeLevel1:
      HANDLE_CASE(kFixedSizeLevel1, kFixedSizeLevel1ItemPreThread, status);
      break;
    case kFixedSizeLevel2:
      HANDLE_CASE(kFixedSizeLevel2, kFixedSizeLevel2ItemPreThread, status);
      break;
    case kFixedSizeLevel3:
    case kFixedSizeLevel3SubThreshold1:
    case kFixedSizeLevel3SubThreshold2:
      HANDLE_CASE(kFixedSizeLevel3, kFixedSizeLevel3ItemPreThread, status);
      break;
    case kFixedSizeLevel4:
    case kFixedSizeLevel4SubThreshold:
      HANDLE_CASE(kFixedSizeLevel4, kFixedSizeLevel4ItemPreThread, status);
      break;
    case kFixedSizeLevel5:
    case kFixedSizeLevel5SubThreshold1:
    case kFixedSizeLevel5SubThreshold2:
    case kFixedSizeLevel5SubThreshold3:
    case kFixedSizeLevel5SubThreshold4:
      HANDLE_CASE(kFixedSizeLevel5, kFixedSizeLevel5ItemPreThread, status);
      break;
    case 1:
      return true;
    default:
      MS_LOG(ERROR) << "SortKeyValueInplace only support sort size less than or equal to 4096, but got "
                    << key_slice_size;
      return false;
  }
  CHECK_CUDA_STATUS(status, "SegSort called by FastSort");
  return true;
#undef HANDLE_CASE
}

template <typename K>
bool InitIndexBySlice(const TensorLayoutHelper &t, int64_t axis, K *data, cudaStream_t cuda_stream) {
  if (t.shape_size_ <= 0) {
    return true;
  }

  if (axis < 0) {
    axis += t.dim_size_;
  }
  if (axis >= t.dim_size_ || axis < 0) {
    MS_LOG(ERROR) << "axis out of range of dim_size_.";
    return false;
  }

  // implement cuda method to init slice data and avoiding temp-data malloc and cudaMemcpy in future.
  int64_t slice_size = t.sizes_[axis];
  K *slice_data_host = reinterpret_cast<K *>(malloc(slice_size * sizeof(K)));
  if (slice_data_host == nullptr) {
    MS_LOG(ERROR) << "Malloc slice index data failed.";
    return false;
  }
  for (int64_t i = 0; i < slice_size; i++) {
    slice_data_host[i] = i;
  }

  K *slice_data_device = nullptr;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMalloc(reinterpret_cast<void **>(&slice_data_device), slice_size * sizeof(K)),
                                     "Malloc slice data failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(slice_data_device, slice_data_host, slice_size * sizeof(K), cudaMemcpyHostToDevice, cuda_stream),
    "Memcpy slice data from host to device failed.");
  free(slice_data_host);
  std::vector<int64_t> in_size(MAX_DIMS, 1);
  std::vector<int64_t> out_size(MAX_DIMS, 1);
  in_size[MAX_DIMS - t.dim_size_ + axis] = t.sizes_[axis];
  for (int i = t.dim_size_ - 1; i >= 0; i--) {
    out_size[i + MAX_DIMS - t.dim_size_] = t.sizes_[i];
  }
  std::vector<int64_t> simplified_inp_shape;
  std::vector<int64_t> simplified_out_shape;
  SimplifyBroadcastToShape(in_size, out_size, &simplified_inp_shape, &simplified_out_shape);
  auto status =
    BroadcastTo<K>(simplified_inp_shape, simplified_out_shape, slice_data_device, data, GET_CTX_DEVICE_ID, cuda_stream);
  CHECK_CUDA_STATUS(status, "InitIndexBySlice called by FastSort");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaFree(slice_data_device), "Free slice data failed.");
  return true;
}

template bool InitIndexBySlice<int64_t>(const TensorLayoutHelper &t, int64_t axis, int64_t *data,
                                        cudaStream_t cuda_stream);

template bool InitIndexBySlice<int32_t>(const TensorLayoutHelper &t, int64_t axis, int32_t *data,
                                        cudaStream_t cuda_stream);

template <typename K, typename V>
bool SortKeyValueInplace(const TensorLayoutHelper &key, K *key_data, const TensorLayoutHelper &value, V *value_data,
                         int64_t axis, bool descending, cudaStream_t cuda_stream) {
  if (key.dim_size_ != value.dim_size_) {
    MS_LOG(ERROR) << "dim_size of key(" << key.dim_size_ << ") should be equal to dim_size of value(" << value.dim_size_
                  << ").";
    return false;
  }
  int dims = value.dim_size_;
  if (dims > MAX_DIMS) {
    MS_LOG(ERROR) << "dim_size should be less than or equal to " << MAX_DIMS << ", but got " << dims << ".";
    return false;
  }

  int in_elements = key.shape_size_;
  if (in_elements == 0) {
    return true;
  }

  int key_slice_size = key.sizes_[axis];
  int key_slices = in_elements / key_slice_size;

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis

  TensorLayoutHelper key_info(key.sizes_, key.dim_size_);
  TensorLayoutHelper value_info(value.sizes_, value.dim_size_);

  auto stride_key = key_info.strides_[axis];
  key_info.sizes_[axis] = 1;
  int collapse_key_dim = key_info.CollapseDims(axis);
  key_info.strides_[collapse_key_dim] = stride_key;
  auto stride_value = value_info.strides_[axis];
  value_info.sizes_[axis] = 1;
  int collapse_value_dim = value_info.CollapseDims(axis);
  value_info.strides_[collapse_value_dim] = stride_value;

  if (key_info.IsContiguous()) {
    return SegSort<K, V>(kFixedSizeSortKeyDimsLastSecond, key_info, key_data, (int64_t)key_slices,
                         (int64_t)key_slice_size, (int64_t)key_info.strides_[collapse_key_dim], value_info, value_data,
                         (int64_t)value_info.strides_[collapse_value_dim], descending, cuda_stream);
  } else {
    constexpr int kDimSize = 2;
    switch (key_info.dim_size_) {
      case kDimSize:  // if sort dim == -1:
        return SegSort<K, V>(kFixedSizeSortKeyDimsSecond, key_info, key_data, (unsigned int)key_slices,
                             (unsigned int)key_slice_size, (unsigned int)key_info.strides_[collapse_key_dim],
                             value_info, value_data, (unsigned int)value_info.strides_[collapse_value_dim], descending,
                             cuda_stream);
      default:  // if sort dim != -1:
        return SegSort<K, V>(kFixedSizeSortKeyDimsLast, key_info, key_data, (unsigned int)key_slices,
                             (unsigned int)key_slice_size, (unsigned int)key_info.strides_[collapse_key_dim],
                             value_info, value_data, (unsigned int)value_info.strides_[collapse_value_dim], descending,
                             cuda_stream);
    }
  }
}

#define SortKeyValueInplace(K, V)                                                                                      \
  template bool SortKeyValueInplace<K, V>(const TensorLayoutHelper &key, K *key_data, const TensorLayoutHelper &value, \
                                          V *value_data, int64_t axis, bool descending, cudaStream_t cuda_stream);

SortKeyValueInplace(bool, int64_t);
SortKeyValueInplace(int8_t, int64_t);
SortKeyValueInplace(int16_t, int64_t);
SortKeyValueInplace(int32_t, int64_t);
SortKeyValueInplace(int64_t, int64_t);
SortKeyValueInplace(uint8_t, int64_t);
SortKeyValueInplace(half, int64_t);
SortKeyValueInplace(float, int64_t);
SortKeyValueInplace(double, int64_t);

SortKeyValueInplace(bool, int32_t);
SortKeyValueInplace(int8_t, int32_t);
SortKeyValueInplace(int16_t, int32_t);
SortKeyValueInplace(int32_t, int32_t);
SortKeyValueInplace(int64_t, int32_t);
SortKeyValueInplace(uint8_t, int32_t);
SortKeyValueInplace(half, int32_t);
SortKeyValueInplace(float, int32_t);
SortKeyValueInplace(double, int32_t);
