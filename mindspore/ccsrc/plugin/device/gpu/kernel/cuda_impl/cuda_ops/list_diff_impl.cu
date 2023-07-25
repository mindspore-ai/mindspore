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

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include "include/cuda_fp16.h"
#include "include/cuda_runtime.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/list_diff_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

struct is_selected {
  __host__ __device__ bool operator()(const bool x) { return x == false; }
};
template <typename T, typename S>
cudaError_t CalListDiff(size_t x_size, size_t y_size, const T *x, const T *y, T *out, S *idx, T *workspace_y,
                        S *workspace_xidx, bool *workspace_flag, const uint32_t &device_id, cudaStream_t cuda_stream,
                        int *count) {
  int count_out = 0;
  auto policy = thrust::cuda::par.on(cuda_stream);
  cudaMemcpy(workspace_y, y, y_size * sizeof(T), cudaMemcpyDeviceToDevice);
  thrust::sequence(policy, thrust::device_pointer_cast(workspace_xidx),
                   thrust::device_pointer_cast(workspace_xidx) + x_size);
  thrust::stable_sort(policy, thrust::device_pointer_cast(workspace_y),
                      thrust::device_pointer_cast(workspace_y) + y_size);
  thrust::binary_search(thrust::device_pointer_cast(workspace_y), thrust::device_pointer_cast(workspace_y) + y_size,
                        thrust::device_pointer_cast(x), thrust::device_pointer_cast(x) + x_size,
                        thrust::device_pointer_cast(workspace_flag));
  count_out = thrust::count(policy, thrust::device_pointer_cast(workspace_flag),
                            thrust::device_pointer_cast(workspace_flag) + x_size, false);
  thrust::copy_if(
    policy,
    thrust::make_zip_iterator(
      thrust::make_tuple(thrust::device_pointer_cast(workspace_xidx), thrust::device_pointer_cast(x))),
    thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(workspace_xidx) + x_size,
                                                 thrust::device_pointer_cast(x) + x_size)),
    thrust::device_pointer_cast(workspace_flag),
    thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(idx), thrust::device_pointer_cast(out))),
    is_selected());
  *count = count_out;
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const half *, const half *, half *, int64_t *, half *,
                                                 int64_t *, bool *, const uint32_t &, cudaStream_t cuda_stream,
                                                 int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const float *, const float *, float *, int64_t *,
                                                 float *, int64_t *, bool *, const uint32_t &, cudaStream_t cuda_stream,
                                                 int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const double *, const double *, double *, int64_t *,
                                                 double *, int64_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const uint8_t *, const uint8_t *, uint8_t *, int64_t *,
                                                 uint8_t *, int64_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const uint16_t *, const uint16_t *, uint16_t *,
                                                 int64_t *, uint16_t *, int64_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int8_t *, const int8_t *, int8_t *, int64_t *,
                                                 int8_t *, int64_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int16_t *, const int16_t *, int16_t *, int64_t *,
                                                 int16_t *, int64_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int32_t *, const int32_t *, int32_t *, int64_t *,
                                                 int32_t *, int64_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int64_t *, const int64_t *, int64_t *, int64_t *,
                                                 int64_t *, int64_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const half *, const half *, half *, int32_t *, half *,
                                                 int32_t *, bool *, const uint32_t &, cudaStream_t cuda_stream,
                                                 int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const float *, const float *, float *, int32_t *,
                                                 float *, int32_t *, bool *, const uint32_t &, cudaStream_t cuda_stream,
                                                 int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const double *, const double *, double *, int32_t *,
                                                 double *, int32_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const uint8_t *, const uint8_t *, uint8_t *, int32_t *,
                                                 uint8_t *, int32_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const uint16_t *, const uint16_t *, uint16_t *,
                                                 int32_t *, uint16_t *, int32_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int8_t *, const int8_t *, int8_t *, int32_t *,
                                                 int8_t *, int32_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int16_t *, const int16_t *, int16_t *, int32_t *,
                                                 int16_t *, int32_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int32_t *, const int32_t *, int32_t *, int32_t *,
                                                 int32_t *, int32_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
template CUDA_LIB_EXPORT cudaError_t CalListDiff(size_t, size_t, const int64_t *, const int64_t *, int64_t *, int32_t *,
                                                 int64_t *, int32_t *, bool *, const uint32_t &,
                                                 cudaStream_t cuda_stream, int *count);
