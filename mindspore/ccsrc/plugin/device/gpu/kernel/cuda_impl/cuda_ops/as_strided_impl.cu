/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/as_strided_impl.cuh"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace {
// dim will be 9, when op is pixel shuffle
constexpr size_t kMaxDim = 9;
}  // namespace

template <size_t N>
class VectorWrapper {
 public:
  explicit VectorWrapper(std::vector<int64_t> v) {
    std::reverse(v.begin(), v.end());
    std::copy(v.begin(), v.end(), data);
  }
  ~VectorWrapper() {}
  __device__ int64_t &operator[](size_t index) { return data[index]; }

 private:
  int64_t data[N];
};

template <typename DataType>
__global__ void AsStridedKernel(const size_t input_size, const DataType *input_ptr, DataType *output_ptr, size_t ndim,
                                VectorWrapper<kMaxDim> output_shape, VectorWrapper<kMaxDim> strides,
                                size_t storage_offset) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < input_size; idx += blockDim.x * gridDim.x) {
    int64_t tmp_idx = idx;
    int64_t offset = 0;
    for (size_t dim = 0; dim < ndim; dim++) {
      int64_t mod = tmp_idx % output_shape[dim];
      tmp_idx = tmp_idx / output_shape[dim];
      offset += mod * strides[dim];
    }
    output_ptr[idx] = input_ptr[offset + storage_offset];
  }
}

template <typename DataType>
cudaError_t CalAsStrided(const size_t &input_size, const DataType *input_ptr, DataType *output_ptr,
                         const mindspore::TensorStorageInfoPtr &input_storage_info, cudaStream_t cuda_stream) {
  size_t ndim = input_storage_info->shape.size();
  VectorWrapper<kMaxDim> output_shape(input_storage_info->shape);
  VectorWrapper<kMaxDim> strides(input_storage_info->strides);

  AsStridedKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
    input_size, input_ptr, output_ptr, ndim, output_shape, strides, input_storage_info->storage_offset);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalAsStrided<uint8_t>(const size_t &input_size, const uint8_t *input_ptr,
                                                           uint8_t *output_ptr,
                                                           const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<uint16_t>(const size_t &input_size, const uint16_t *input_ptr,
                                                            uint16_t *output_ptr,
                                                            const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<uint32_t>(const size_t &input_size, const uint32_t *input_ptr,
                                                            uint32_t *output_ptr,
                                                            const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<uint64_t>(const size_t &input_size, const uint64_t *input_ptr,
                                                            uint64_t *output_ptr,
                                                            const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalAsStrided<half>(const size_t &input_size, const half *input_ptr,
                                                        half *output_ptr,
                                                        const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<float>(const size_t &input_size, const float *input_ptr,
                                                         float *output_ptr,
                                                         const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<double>(const size_t &input_size, const double *input_ptr,
                                                          double *output_ptr,
                                                          const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<bool>(const size_t &input_size, const bool *input_ptr,
                                                        bool *output_ptr,
                                                        const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<int8_t>(const size_t &input_size, const int8_t *input_ptr,
                                                          int8_t *output_ptr,
                                                          const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<int16_t>(const size_t &input_size, const int16_t *input_ptr,
                                                           int16_t *output_ptr,
                                                           const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<int32_t>(const size_t &input_size, const int32_t *input_ptr,
                                                           int32_t *output_ptr,
                                                           const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAsStrided<int64_t>(const size_t &input_size, const int64_t *input_ptr,
                                                           int64_t *output_ptr,
                                                           const mindspore::TensorStorageInfoPtr &input_storage_info,
                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t
CalAsStrided<Complex<float>>(const size_t &input_size, const Complex<float> *input_ptr, Complex<float> *output_ptr,
                             const mindspore::TensorStorageInfoPtr &input_storage_info, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalAsStrided<Complex<double>>(const size_t &input_size, const Complex<double> *input_ptr, Complex<double> *output_ptr,
                              const mindspore::TensorStorageInfoPtr &input_storage_info, cudaStream_t cuda_stream);
