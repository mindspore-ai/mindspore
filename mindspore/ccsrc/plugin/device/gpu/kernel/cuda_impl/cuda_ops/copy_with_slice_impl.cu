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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/copy_with_slice_impl.cuh"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace {
constexpr size_t kMaxDim = 8;
}

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
__global__ void CopyWithSliceKernelDFST(const size_t input_size, const DataType *src_addr, DataType *self_addr,
                                        size_t ndim, VectorWrapper<kMaxDim> output_shape,
                                        VectorWrapper<kMaxDim> strides, size_t src_storage_offset,
                                        size_t dst_storage_offset) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < input_size; idx += blockDim.x * gridDim.x) {
    int64_t tmp_idx = idx;
    int64_t offset = 0;
    for (size_t dim = 0; dim < ndim; dim++) {
      int64_t mod = tmp_idx % output_shape[dim];
      tmp_idx = tmp_idx / output_shape[dim];
      offset += mod * strides[dim];
    }
    self_addr[offset + dst_storage_offset] = src_addr[idx + src_storage_offset];
  }
}

template <typename DataType>
__global__ void CopyWithSliceKernelDTSF(const size_t input_size, const DataType *src_addr, DataType *self_addr,
                                        size_t src_ndim, VectorWrapper<kMaxDim> input_shape,
                                        VectorWrapper<kMaxDim> src_strides, size_t src_storage_offset,
                                        size_t dst_storage_offset) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < input_size; idx += blockDim.x * gridDim.x) {
    int64_t tmp_idx = idx;
    int64_t offset = 0;
    for (size_t dim = 0; dim < src_ndim; dim++) {
      int64_t mod = tmp_idx % input_shape[dim];
      tmp_idx = tmp_idx / input_shape[dim];
      offset += mod * src_strides[dim];
    }
    self_addr[idx + dst_storage_offset] = src_addr[offset + src_storage_offset];
  }
}

template <typename DataType>
__global__ void CopyWithSliceKernelDFSF(const size_t input_size, const DataType *src_addr, DataType *self_addr,
                                        size_t src_ndim, size_t dst_ndim, VectorWrapper<kMaxDim> input_shape,
                                        VectorWrapper<kMaxDim> output_shape, VectorWrapper<kMaxDim> src_strides,
                                        VectorWrapper<kMaxDim> dst_strides, size_t src_storage_offset,
                                        size_t dst_storage_offset) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < input_size; idx += blockDim.x * gridDim.x) {
    int64_t src_tmp_idx = idx;
    int64_t src_offset = 0;
    for (size_t dim = 0; dim < src_ndim; dim++) {
      int64_t mod = src_tmp_idx % input_shape[dim];
      src_tmp_idx = src_tmp_idx / input_shape[dim];
      src_offset += mod * src_strides[dim];
    }

    int64_t dst_tmp_idx = idx;
    int64_t dst_offset = 0;
    for (size_t dim = 0; dim < dst_ndim; dim++) {
      int64_t mod = dst_tmp_idx % output_shape[dim];
      dst_tmp_idx = dst_tmp_idx / output_shape[dim];
      dst_offset += mod * dst_strides[dim];
    }
    self_addr[dst_offset + dst_storage_offset] = src_addr[src_offset + src_storage_offset];
  }
}

template <typename DataType>
cudaError_t CalCopyWithSlice(const size_t &input_size, const DataType *src_addr,
                             const mindspore::TensorStorageInfoPtr &src_storage_info, DataType *self_addr,
                             const mindspore::TensorStorageInfoPtr &dst_storage_info, cudaStream_t cuda_stream) {
  size_t dst_ndim = dst_storage_info->shape.size();
  VectorWrapper<kMaxDim> output_shape(dst_storage_info->shape);
  VectorWrapper<kMaxDim> dst_strides(dst_storage_info->strides);
  bool src_is_contiguous = src_storage_info == nullptr || src_storage_info->is_contiguous;
  bool dst_is_contiguous = dst_storage_info->is_contiguous;
  if (dst_is_contiguous && !src_is_contiguous) {
    VectorWrapper<kMaxDim> input_shape(src_storage_info->shape);
    VectorWrapper<kMaxDim> src_strides(src_storage_info->strides);
    size_t src_ndim = src_storage_info->shape.size();
    CopyWithSliceKernelDTSF<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
      input_size, src_addr, self_addr, src_ndim, input_shape, src_strides, src_storage_info->storage_offset,
      dst_storage_info->storage_offset);
  } else if (!dst_is_contiguous && src_is_contiguous) {
    CopyWithSliceKernelDFST<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
      input_size, src_addr, self_addr, dst_ndim, output_shape, dst_strides, src_storage_info->storage_offset,
      dst_storage_info->storage_offset);
  } else {
    VectorWrapper<kMaxDim> input_shape(src_storage_info->shape);
    VectorWrapper<kMaxDim> src_strides(src_storage_info->strides);
    size_t src_ndim = src_storage_info->shape.size();
    CopyWithSliceKernelDFSF<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
      input_size, src_addr, self_addr, src_ndim, dst_ndim, input_shape, output_shape, src_strides, dst_strides,
      src_storage_info->storage_offset, dst_storage_info->storage_offset);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<uint8_t>(const size_t &input_size, const uint8_t *src_addr,
                                                               const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                               uint8_t *self_addr,
                                                               const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<uint16_t>(const size_t &input_size, const uint16_t *src_addr,
                                                                const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                                uint16_t *self_addr,
                                                                const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<uint32_t>(const size_t &input_size, const uint32_t *src_addr,
                                                                const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                                uint32_t *self_addr,
                                                                const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<uint64_t>(const size_t &input_size, const uint64_t *src_addr,
                                                                const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                                uint64_t *self_addr,
                                                                const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                                cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<half>(const size_t &input_size, const half *src_addr,
                                                            const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                            half *self_addr,
                                                            const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<float>(const size_t &input_size, const float *src_addr,
                                                             const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                             float *self_addr,
                                                             const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<double>(const size_t &input_size, const double *src_addr,
                                                              const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                              double *self_addr,
                                                              const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<bool>(const size_t &input_size, const bool *src_addr,
                                                            const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                            bool *self_addr,
                                                            const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<int8_t>(const size_t &input_size, const int8_t *src_addr,
                                                              const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                              int8_t *self_addr,
                                                              const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<int16_t>(const size_t &input_size, const int16_t *src_addr,
                                                               const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                               int16_t *self_addr,
                                                               const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<int32_t>(const size_t &input_size, const int32_t *src_addr,
                                                               const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                               int32_t *self_addr,
                                                               const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<int64_t>(const size_t &input_size, const int64_t *src_addr,
                                                               const mindspore::TensorStorageInfoPtr &src_storage_info,
                                                               int64_t *self_addr,
                                                               const mindspore::TensorStorageInfoPtr &dst_storage_info,
                                                               cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<Complex<float>>(
  const size_t &input_size, const Complex<float> *src_addr, const mindspore::TensorStorageInfoPtr &src_storage_info,
  Complex<float> *self_addr, const mindspore::TensorStorageInfoPtr &dst_storage_info, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCopyWithSlice<Complex<double>>(
  const size_t &input_size, const Complex<double> *src_addr, const mindspore::TensorStorageInfoPtr &src_storage_info,
  Complex<double> *self_addr, const mindspore::TensorStorageInfoPtr &dst_storage_info, cudaStream_t cuda_stream);
