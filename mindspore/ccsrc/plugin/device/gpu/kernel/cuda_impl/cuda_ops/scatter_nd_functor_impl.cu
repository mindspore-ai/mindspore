/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scatter_nd_functor_impl.cuh"
template <typename T, typename S>
__global__ void ScatterNdUpdate(const size_t unit_size, const size_t index_depth, const size_t updates_size,
                                const S *out_strides, const S *indices, const S *work_shape, const T *updates,
                                T *input) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < (updates_size);
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = read_index / unit_size;
    j = read_index % unit_size;

    for (size_t k = 0; k < index_depth; k++) {
      S indices_i = indices[i * index_depth + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * out_strides[k] * unit_size;
    }

    write_index += j;
    CUDA_KERNEL_ASSERT(!out_bound);
    input[write_index] = updates[read_index];
  }
}

template <typename T, typename S, typename Functor>
__global__ void ScatterNdBinaryOp(Functor func, const size_t unit_size, const size_t index_depth,
                                  const size_t updates_size, const S *out_strides, const S *indices,
                                  const S *work_shape, const T *updates, T *input) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < (updates_size);
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = read_index / unit_size;
    j = read_index % unit_size;

    for (size_t k = 0; k < index_depth; k++) {
      S indices_i = indices[i * index_depth + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * out_strides[k] * unit_size;
    }

    write_index += j;
    CUDA_KERNEL_ASSERT(!out_bound);
    func(&input[write_index], updates[read_index]);
  }
}

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T, typename S>
cudaError_t CalScatterNdFunctor(enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units,
                                const size_t &index_depth, const S *out_strides, const S *indices, const S *work_shape,
                                const T *updates, T *input, uint32_t device_id, cudaStream_t cuda_stream) {
  const size_t updates_size = unit_size * num_units;
  switch (func_type) {
    case SCATTER_ND_FUNC_UPDATE:
      ScatterNdUpdate<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
      break;
    case SCATTER_ND_FUNC_ADD:
      ScatterNdBinaryOp<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        MsAtomicAddFunctor{}, unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
      break;
    case SCATTER_ND_FUNC_SUB:
      ScatterNdBinaryOp<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        MsAtomicSubFunctor{}, unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
      break;
    case SCATTER_ND_FUNC_MUL:
      ScatterNdBinaryOp<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        MsAtomicMulFunctor{}, unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
      break;
    case SCATTER_ND_FUNC_DIV:
      ScatterNdBinaryOp<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        MsAtomicDivFunctor{}, unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
      break;
    case SCATTER_ND_FUNC_MAX:
      ScatterNdBinaryOp<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        MsAtomicMaxFunctor{}, unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
      break;
    case SCATTER_ND_FUNC_MIN:
      ScatterNdBinaryOp<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        MsAtomicMinFunctor{}, unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
      break;
    default:
      break;
  }
  return GetCudaStatus();
}

template <>
cudaError_t CalScatterNdFunctor(enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units,
                                const size_t &index_depth, const int32_t *out_strides, const int32_t *indices,
                                const int32_t *work_shape, const std::complex<float> *updates,
                                std::complex<float> *input, uint32_t device_id, cudaStream_t cuda_stream) {
  const size_t updates_size = unit_size * num_units;
  ScatterNdUpdate<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
  return GetCudaStatus();
}

template <>
cudaError_t CalScatterNdFunctor(enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units,
                                const size_t &index_depth, const int64_t *out_strides, const int64_t *indices,
                                const int64_t *work_shape, const std::complex<float> *updates,
                                std::complex<float> *input, uint32_t device_id, cudaStream_t cuda_stream) {
  const size_t updates_size = unit_size * num_units;
  ScatterNdUpdate<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
  return GetCudaStatus();
}

template <>
cudaError_t CalScatterNdFunctor(enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units,
                                const size_t &index_depth, const int32_t *out_strides, const int32_t *indices,
                                const int32_t *work_shape, const std::complex<double> *updates,
                                std::complex<double> *input, uint32_t device_id, cudaStream_t cuda_stream) {
  const size_t updates_size = unit_size * num_units;
  ScatterNdUpdate<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
  return GetCudaStatus();
}

template <>
cudaError_t CalScatterNdFunctor(enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units,
                                const size_t &index_depth, const int64_t *out_strides, const int64_t *indices,
                                const int64_t *work_shape, const std::complex<double> *updates,
                                std::complex<double> *input, uint32_t device_id, cudaStream_t cuda_stream) {
  const size_t updates_size = unit_size * num_units;
  ScatterNdUpdate<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    unit_size, index_depth, updates_size, out_strides, indices, work_shape, updates, input);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<double, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const double *updates, double *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<double, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const double *updates, double *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<float, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const float *updates, float *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<float, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const float *updates, float *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<half, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const half *updates, half *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<half, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const half *updates, half *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int64_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const int64_t *updates, int64_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int64_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const int64_t *updates, int64_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint64_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const uint64_t *updates,
  uint64_t *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint64_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const uint64_t *updates,
  uint64_t *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int32_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const int32_t *updates, int32_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int32_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const int32_t *updates, int32_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint32_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const uint32_t *updates,
  uint32_t *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint32_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const uint32_t *updates,
  uint32_t *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int16_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const int16_t *updates, int16_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int16_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const int16_t *updates, int16_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint16_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const uint16_t *updates,
  uint16_t *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint16_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const uint16_t *updates,
  uint16_t *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint8_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const uint8_t *updates, uint8_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<uint8_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const uint8_t *updates, uint8_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int8_t, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const int8_t *updates, int8_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<int8_t, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const int8_t *updates, int8_t *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<bool, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const bool *updates, bool *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<bool, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const bool *updates, bool *input,
  uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<std::complex<float>, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const std::complex<float> *updates,
  std::complex<float> *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<std::complex<float>, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const std::complex<float> *updates,
  std::complex<float> *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<std::complex<double>, int64_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int64_t *out_strides, const int64_t *indices, const int64_t *work_shape, const std::complex<double> *updates,
  std::complex<double> *input, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalScatterNdFunctor<std::complex<double>, int32_t>(
  enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units, const size_t &index_depth,
  const int32_t *out_strides, const int32_t *indices, const int32_t *work_shape, const std::complex<double> *updates,
  std::complex<double> *input, uint32_t device_id, cudaStream_t cuda_stream);
