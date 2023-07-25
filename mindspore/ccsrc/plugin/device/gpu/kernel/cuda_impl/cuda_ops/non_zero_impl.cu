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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/non_zero_impl.cuh"
#include <cub/cub.cuh>

constexpr size_t kNonZeroMaxDim = 10;

struct TensorShape {
  size_t data[kNonZeroMaxDim];
  size_t rank;
};

template <typename DataType>
struct IsZero {
  __host__ __device__ __forceinline__ size_t operator()(const DataType &x) const { return x == DataType(0) ? 0 : 1; }
};

// Inspired by cub library
template <typename IndexType>
class NonZeroOutputIterator {
 public:
  // Required iterator traits
  typedef NonZeroOutputIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef void value_type;
  typedef void *pointer;
  typedef IndexType &reference;

#if (THRUST_VERSION >= 100700)
  // Use Thrust's iterator categories so we can use these iterators in Thrust 1.7 (or newer) methods
  typedef
    typename thrust::detail::iterator_facade_category<thrust::any_system_tag, thrust::random_access_traversal_tag,
                                                      value_type,
                                                      reference>::type iterator_category;  ///< The iterator category
#else
  typedef std::random_access_iterator_tag iterator_category;  ///< The iterator category
#endif  // THRUST_VERSION

  NonZeroOutputIterator(IndexType *ptr, size_t rank) : ptr_(ptr), rank_(rank) {}

  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    // To avoid data conflict in NonZeroKernel.
    return *(ptr_ + rank_ * n);
  }

 private:
  IndexType *ptr_;
  const size_t rank_;
};

template <typename IndexType>
__global__ void NonZeroKernel(IndexType *output_ptr, const size_t *output_size_ptr, const TensorShape shape) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < *output_size_ptr; tid += blockDim.x * gridDim.x) {
    size_t fill_value = output_ptr[tid * shape.rank];
    for (size_t i = 0, j = shape.rank, k = (tid + 1) * shape.rank; i < shape.rank; ++i) {
      size_t base = shape.data[--j];
      output_ptr[--k] = fill_value % base;
      fill_value /= base;
    }
  }
}

template <typename DataType, typename IndexType>
cudaError_t NonZero(const DataType *input_ptr, IndexType *output_ptr, size_t *output_size_ptr,
                    const std::vector<size_t> &input_shape, size_t input_size, const uint32_t &device_id,
                    cudaStream_t cuda_stream) {
  // Set the index (1-D base) for non-zero elements and place them into output.
  // To support in place operation later, we use custom output iterator,
  // which is inspired by cub library. And output_size_ptr stores the number of non-zero elements.
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::CountingInputIterator<IndexType> count_iter(0);
  cub::TransformInputIterator<size_t, IsZero<DataType>, const DataType *> trans_iter(input_ptr, IsZero<DataType>());
  NonZeroOutputIterator<IndexType> out_iter(output_ptr, input_shape.size());
  (void)cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, count_iter, trans_iter, out_iter, output_size_ptr,
                                   input_size, cuda_stream);
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, count_iter, trans_iter, out_iter,
                                   output_size_ptr, input_size, cuda_stream);

  if (input_shape.size() > 1) {
    TensorShape shape;
    shape.rank = input_shape.size();
    for (size_t i = 0; i < input_shape.size(); i++) {
      shape.data[i] = input_shape[i];
    }
    // Transform output index (1-D base) to N-D base in place.
    // e.g., [0, 2, 3] -> [(0, 0), (1, 0), (1, 1)] when shape is (2, 2)
    NonZeroKernel<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      output_ptr, output_size_ptr, shape);
  }

  // Since cudaGetLastError can return the last error from a runtime call,
  // we catch the error in Launch function.
  (void)cudaFree(d_temp_storage);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t NonZero<bool, int64_t>(const bool *input_ptr, int64_t *output_ptr,
                                                            size_t *output_size_ptr,
                                                            const std::vector<size_t> &input_shape, size_t input_size,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<uint8_t, int64_t>(const uint8_t *input_ptr, int64_t *output_ptr,
                                                               size_t *output_size_ptr,
                                                               const std::vector<size_t> &input_shape,
                                                               size_t input_size, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<uint16_t, int64_t>(const uint16_t *input_ptr, int64_t *output_ptr,
                                                                size_t *output_size_ptr,
                                                                const std::vector<size_t> &input_shape,
                                                                size_t input_size, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<uint32_t, int64_t>(const uint32_t *input_ptr, int64_t *output_ptr,
                                                                size_t *output_size_ptr,
                                                                const std::vector<size_t> &input_shape,
                                                                size_t input_size, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<uint64_t, int64_t>(const uint64_t *input_ptr, int64_t *output_ptr,
                                                                size_t *output_size_ptr,
                                                                const std::vector<size_t> &input_shape,
                                                                size_t input_size, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<int8_t, int64_t>(const int8_t *input_ptr, int64_t *output_ptr,
                                                              size_t *output_size_ptr,
                                                              const std::vector<size_t> &input_shape, size_t input_size,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<int16_t, int64_t>(const int16_t *input_ptr, int64_t *output_ptr,
                                                               size_t *output_size_ptr,
                                                               const std::vector<size_t> &input_shape,
                                                               size_t input_size, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<int32_t, int64_t>(const int32_t *input_ptr, int64_t *output_ptr,
                                                               size_t *output_size_ptr,
                                                               const std::vector<size_t> &input_shape,
                                                               size_t input_size, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<int64_t, int64_t>(const int64_t *input_ptr, int64_t *output_ptr,
                                                               size_t *output_size_ptr,
                                                               const std::vector<size_t> &input_shape,
                                                               size_t input_size, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<half, int64_t>(const half *input_ptr, int64_t *output_ptr,
                                                            size_t *output_size_ptr,
                                                            const std::vector<size_t> &input_shape, size_t input_size,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<float, int64_t>(const float *input_ptr, int64_t *output_ptr,
                                                             size_t *output_size_ptr,
                                                             const std::vector<size_t> &input_shape, size_t input_size,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NonZero<double, int64_t>(const double *input_ptr, int64_t *output_ptr,
                                                              size_t *output_size_ptr,
                                                              const std::vector<size_t> &input_shape, size_t input_size,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
