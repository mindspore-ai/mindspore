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

#include "include/cuda_fp16.h"
#include "include/cuda_runtime.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/list_diff_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void InitListDiff(size_t x_size, size_t y_size, const T *x, const T *y, int *workspace_flag) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < x_size * y_size; pos += blockDim.x * gridDim.x) {
    size_t x_pos = pos / y_size;
    size_t y_pos = pos % y_size;
    if (x[x_pos] == y[y_pos]) {
      workspace_flag[x_pos] = 1;
    }
  }
}

template <typename T, typename S>
__global__ void ShrinkRes(int count_out, const T *x, int *workspace_flag, T *out, S *idx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < count_out; pos += blockDim.x * gridDim.x) {
    out[pos] = x[workspace_flag[pos]];
    idx[pos] = static_cast<S>(workspace_flag[pos]);
  }
}

template <typename T, typename S>
int ListDiff(int *count_number, size_t x_size, size_t y_size, const T *x, const T *y, T *out, S *idx,
             int *workspace_flag, const uint32_t &device_id, cudaStream_t cuda_stream) {
  int count_out = 0;
  cudaMemset(workspace_flag, 0, sizeof(int) * x_size);
  InitListDiff<<<CUDA_BLOCKS(device_id, x_size * y_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(x_size, y_size, x,
                                                                                                     y, workspace_flag);
  std::vector<int> workspace_flag_host(x_size);
  cudaMemcpy(workspace_flag_host.data(), workspace_flag, x_size * sizeof(int), cudaMemcpyDeviceToHost);
  for (size_t idx = 0; idx < x_size; ++idx) {
    if (workspace_flag_host[idx] == 0) {
      workspace_flag_host[count_out] = idx;
      ++count_out;
    }
  }
  cudaMemcpy(workspace_flag, workspace_flag_host.data(), count_out * sizeof(int), cudaMemcpyHostToDevice);
  ShrinkRes<<<CUDA_BLOCKS(device_id, count_out), CUDA_THREADS(device_id), 0, cuda_stream>>>(count_out, x,
                                                                                            workspace_flag, out, idx);
  return count_out;
}

template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const half *, const half *, half *, int64_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const float *, const float *, float *, int64_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const double *, const double *, double *, int64_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const uint8_t *, const uint8_t *, uint8_t *, int64_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const uint16_t *, const uint16_t *, uint16_t *, int64_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int8_t *, const int8_t *, int8_t *, int64_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int16_t *, const int16_t *, int16_t *, int64_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int32_t *, const int32_t *, int32_t *, int64_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int64_t *, const int64_t *, int64_t *, int64_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const half *, const half *, half *, int32_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const float *, const float *, float *, int32_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const double *, const double *, double *, int32_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const uint8_t *, const uint8_t *, uint8_t *, int32_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const uint16_t *, const uint16_t *, uint16_t *, int32_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int8_t *, const int8_t *, int8_t *, int32_t *, int *,
                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int16_t *, const int16_t *, int16_t *, int32_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int32_t *, const int32_t *, int32_t *, int32_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT int ListDiff(int *, size_t, size_t, const int64_t *, const int64_t *, int64_t *, int32_t *,
                                      int *, const uint32_t &, cudaStream_t cuda_stream);
