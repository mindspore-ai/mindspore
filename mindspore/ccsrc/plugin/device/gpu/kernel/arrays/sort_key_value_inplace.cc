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

#include "plugin/device/gpu/kernel/arrays/sort_key_value_inplace.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sort_fixed_size.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/hal/device/gpu_common.h"

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

template <int A, typename K, typename V>
bool SegSort(const TensorLayoutHelper &key_info, K *key_data, int64_t key_slices, int64_t key_slice_size,
             int64_t key_slice_stride, const TensorLayoutHelper &value_info, V *value_data, int64_t value_slice_stride,
             bool descending, cudaStream_t stream) {
  int64_t ceil_power_of2 = NextHighestPowerOf2(key_slice_size);

#define HANDLE_CASE(SIZE, ITEMS_PER_THREAD)                                                                           \
  return SortFixedSize<A, SIZE, ITEMS_PER_THREAD, K, V>(key_info, key_data, key_slices, key_slice_size,               \
                                                        key_slice_stride, value_info, value_data, value_slice_stride, \
                                                        descending, stream)
  constexpr int kFixedSizeLevel3SubThreshold1 = 512;
  constexpr int kFixedSizeLevel3SubThreshold2 = 256;
  constexpr int kFixedSizeLevel4SubThreshold = 64;
  constexpr int kFixedSizeLevel5SubThreshold1 = 16;
  constexpr int kFixedSizeLevel5SubThreshold2 = 8;
  constexpr int kFixedSizeLevel5SubThreshold3 = 4;
  constexpr int kFixedSizeLevel5SubThreshold4 = 2;
  switch (ceil_power_of2) {
    case kFixedSizeLevel1:
      HANDLE_CASE(kFixedSizeLevel1, kFixedSizeLevel1ItemPreThread);
    case kFixedSizeLevel2:
      HANDLE_CASE(kFixedSizeLevel2, kFixedSizeLevel2ItemPreThread);
    case kFixedSizeLevel3:
    case kFixedSizeLevel3SubThreshold1:
    case kFixedSizeLevel3SubThreshold2:
      HANDLE_CASE(kFixedSizeLevel3, kFixedSizeLevel3ItemPreThread);
    case kFixedSizeLevel4:
    case kFixedSizeLevel4SubThreshold:
      HANDLE_CASE(kFixedSizeLevel4, kFixedSizeLevel4ItemPreThread);
    case kFixedSizeLevel5:
    case kFixedSizeLevel5SubThreshold1:
    case kFixedSizeLevel5SubThreshold2:
    case kFixedSizeLevel5SubThreshold3:
    case kFixedSizeLevel5SubThreshold4:
      HANDLE_CASE(kFixedSizeLevel5, kFixedSizeLevel5ItemPreThread);
    case 1:
      return true;
    default:
      MS_LOG(ERROR) << "SortKeyValueInplace only support sort size less than or equal to 4096, but got "
                    << key_slice_size;
      return false;
  }
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

  int in_size[MAX_DIMS];
  int out_size[MAX_DIMS];
  for (int i = 0; i < MAX_DIMS; i++) {
    in_size[i] = 1;
  }
  in_size[MAX_DIMS - t.dim_size_ + axis] = t.sizes_[axis];
  for (int i = t.dim_size_ - 1; i >= 0; i--) {
    out_size[i + MAX_DIMS - t.dim_size_] = t.sizes_[i];
  }
  for (int i = MAX_DIMS - t.dim_size_ - 1; i >= 0; i--) {
    out_size[i] = 1;
  }

  constexpr size_t kIndex0 = 0;
  constexpr size_t kIndex1 = 1;
  constexpr size_t kIndex2 = 2;
  constexpr size_t kIndex3 = 3;
  constexpr size_t kIndex4 = 4;
  constexpr size_t kIndex5 = 5;
  constexpr size_t kIndex6 = 6;
  constexpr size_t kIndex7 = 7;
  BroadcastTo<K>(in_size[kIndex0], in_size[kIndex1], in_size[kIndex2], in_size[kIndex3], in_size[kIndex4],
                 in_size[kIndex5], in_size[kIndex6], in_size[kIndex7], out_size[kIndex0], out_size[kIndex1],
                 out_size[kIndex2], out_size[kIndex3], out_size[kIndex4], out_size[kIndex5], out_size[kIndex6],
                 out_size[kIndex7], slice_data_device, data, cuda_stream);
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

#define HANDLE_SORT_CASE(TYPE, A)                                                            \
  return SegSort<A, K, V>(key_info, key_data, (TYPE)key_slices, (TYPE)key_slice_size,        \
                          (TYPE)key_info.strides_[collapse_key_dim], value_info, value_data, \
                          (TYPE)value_info.strides_[collapse_value_dim], descending, cuda_stream)

  if (key_info.IsContiguous()) {
    HANDLE_SORT_CASE(int64_t, kFixedSizeSortKeyDimsLastSecond);
  } else {
    constexpr int kDimSize = 2;
    switch (key_info.dim_size_) {
      case kDimSize:  // if sort dim == -1:
        HANDLE_SORT_CASE(unsigned int, kFixedSizeSortKeyDimsSecond);
      default:  // if sort dim != -1:
        HANDLE_SORT_CASE(unsigned int, kFixedSizeSortKeyDimsLast);
    }
  }
#undef HANDLE_SORT_CASE
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
