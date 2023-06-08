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
 * WITH WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_PUB_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_PUB_CUH_
#include <math.h>
#include <vector>
#include <iostream>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_types.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_common.cuh"

struct BinaryBroadcastStrideInfo {
  size_t in0_stride[8];
  size_t in1_stride[8];
  size_t out_stride[8];
};

template <typename T, size_t VecSize>
struct Vec {
  T data[VecSize];
};
constexpr size_t kMaxVecBytes = 128 / 8;
constexpr size_t kMaxVecSize = 4;
constexpr size_t MsMin(size_t a, size_t b) { return a < b ? a : b; }

template <typename T>
constexpr size_t VecSize() {
  return MsMin(kMaxVecBytes / sizeof(T), kMaxVecSize);
}

template <typename T, typename U, typename... Args>
constexpr size_t VecSize() {
  return MsMin(VecSize<T>(), VecSize<U, Args...>());
}
enum class ScalarOption {
  NoScalar = 0,
  In0Scalar = 1,
  In1Scalar = 2,
};

template <BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t, size_t vec_num>
__device__ void ApplyVec(BinaryFunc<OP, In0_t, In1_t, Out_t> func, ScalarOption scalar_option, In0_t *in0_addr,
                         In1_t *in1_addr, Out_t *out_addr) {
  Vec<Out_t, vec_num> out_vec;
  if (scalar_option == ScalarOption::NoScalar) {
    Vec<In0_t, vec_num> in0_vec = reinterpret_cast<Vec<In0_t, vec_num> *>(in0_addr)[0];
    Vec<In1_t, vec_num> in1_vec = reinterpret_cast<Vec<In1_t, vec_num> *>(in1_addr)[0];
#pragma unroll
    for (size_t idx = 0; idx < vec_num; ++idx) {
      out_vec.data[idx] = func(in0_vec.data[idx], in1_vec.data[idx]);
    }
  } else if (scalar_option == ScalarOption::In0Scalar) {
    In0_t in0_data = in0_addr[0];
    Vec<In1_t, vec_num> in1_vec = reinterpret_cast<Vec<In1_t, vec_num> *>(in1_addr)[0];
#pragma unroll
    for (size_t idx = 0; idx < vec_num; ++idx) {
      out_vec.data[idx] = func(in0_data, in1_vec.data[idx]);
    }
  } else {
    Vec<In0_t, vec_num> in0_vec = reinterpret_cast<Vec<In0_t, vec_num> *>(in0_addr)[0];
    In1_t in1_data = in1_addr[0];
#pragma unroll
    for (size_t idx = 0; idx < vec_num; ++idx) {
      out_vec.data[idx] = func(in0_vec.data[idx], in1_data);
    }
  }
  Vec<Out_t, vec_num> *out_data = reinterpret_cast<Vec<Out_t, vec_num> *>(out_addr);
  out_data[0] = out_vec;
}
static __device__ Vec<size_t, 2> CalInposByOutPos(size_t out_pos, size_t dim_size,
                                                  const BinaryBroadcastStrideInfo &strides) {
  Vec<size_t, 2> in_pos = {0, 0};
  size_t tmp_idx = 0;
  for (int idx = 0; idx < dim_size; ++idx) {
    tmp_idx = out_pos / strides.out_stride[idx];
    in_pos.data[0] += tmp_idx * strides.in0_stride[idx];
    in_pos.data[1] += tmp_idx * strides.in1_stride[idx];
    out_pos -= tmp_idx * strides.out_stride[idx];
  }
  return in_pos;
}
template <BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t>
__global__ void BinaryWithBroadcastNoVecCuda(BinaryFunc<OP, In0_t, In1_t, Out_t> func, size_t dim_size,
                                             size_t total_threads, BinaryBroadcastStrideInfo strides, In0_t *in0,
                                             In1_t *in1, Out_t *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_threads; pos += blockDim.x * gridDim.x) {
    Vec<size_t, 2> in_pos = CalInposByOutPos(pos, dim_size, strides);
    out[pos] = func(in0[in_pos.data[0]], in1[in_pos.data[1]]);
  }
}
template <BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t>
__global__ void BinaryWithoutBroadcastNoVecCuda(BinaryFunc<OP, In0_t, In1_t, Out_t> func, ScalarOption scalar_option,
                                                size_t total_threads, In0_t *in0, In1_t *in1, Out_t *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_threads; pos += blockDim.x * gridDim.x) {
    In0_t in0_data = (scalar_option == ScalarOption::In0Scalar) ? in0[0] : in0[pos];
    In1_t in1_data = (scalar_option == ScalarOption::In1Scalar) ? in1[0] : in1[pos];
    out[pos] = func(in0_data, in1_data);
  }
}
template <BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t, size_t vec_num>
__global__ void BinaryBroadcastVecWithoutTailCuda(BinaryFunc<OP, In0_t, In1_t, Out_t> func, ScalarOption scalar_option,
                                                  size_t dim_size, size_t total_threads,
                                                  BinaryBroadcastStrideInfo strides, In0_t *in0, In1_t *in1,
                                                  Out_t *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_threads; pos += blockDim.x * gridDim.x) {
    size_t out_pos = pos * vec_num;
    Vec<size_t, 2> in_pos = CalInposByOutPos(out_pos, dim_size, strides);
    ApplyVec<OP, In0_t, In1_t, Out_t, vec_num>(func, scalar_option, in0 + in_pos.data[0], in1 + in_pos.data[1],
                                               out + out_pos);
  }
}
template <BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t, size_t vec_num>
__global__ void BinaryBroadcastVecWithTailCuda(BinaryFunc<OP, In0_t, In1_t, Out_t> func, ScalarOption scalar_option,
                                               size_t dim_size, size_t total_threads, size_t step, size_t tail_num,
                                               BinaryBroadcastStrideInfo strides, In0_t *in0, In1_t *in1, Out_t *out) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_threads; pos += blockDim.x * gridDim.x) {
    size_t out_pos = pos * vec_num + pos / step * tail_num;
    Vec<size_t, 2> in_pos = CalInposByOutPos(out_pos, dim_size, strides);
    if ((pos + 1) % step != 0) {
      ApplyVec<OP, In0_t, In1_t, Out_t, vec_num>(func, scalar_option, in0 + in_pos.data[0], in1 + in_pos.data[1],
                                                 out + out_pos);
    } else {
      switch (tail_num) {
        case 1:
          ApplyVec<OP, In0_t, In1_t, Out_t, vec_num + 1>(func, scalar_option, in0 + in_pos.data[0],
                                                         in1 + in_pos.data[1], out + out_pos);
          break;
        case 2:
          ApplyVec<OP, In0_t, In1_t, Out_t, vec_num + 2>(func, scalar_option, in0 + in_pos.data[0],
                                                         in1 + in_pos.data[1], out + out_pos);
          break;
        case 3:
          ApplyVec<OP, In0_t, In1_t, Out_t, vec_num + 3>(func, scalar_option, in0 + in_pos.data[0],
                                                         in1 + in_pos.data[1], out + out_pos);
          break;
      }
    }
  }
}
static BinaryBroadcastStrideInfo BinaryBroadcastCalStride(const size_t dim_size, const std::vector<int64_t> &in0_shape,
                                                          const std::vector<int64_t> &in1_shape,
                                                          const std::vector<int64_t> &out_shape, const size_t vec_num) {
  BinaryBroadcastStrideInfo strides;
  strides.in0_stride[dim_size - 1] = 1;
  strides.in1_stride[dim_size - 1] = 1;
  strides.out_stride[dim_size - 1] = 1;
  for (int64_t idx = dim_size - 2; idx >= 0; --idx) {
    strides.out_stride[idx] = out_shape[idx + 1] * strides.out_stride[idx + 1];
    strides.in0_stride[idx] = in0_shape[idx + 1] * strides.in0_stride[idx + 1];
    strides.in1_stride[idx] = in1_shape[idx + 1] * strides.in1_stride[idx + 1];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    strides.in0_stride[idx] = (in0_shape[idx] == 1) ? 0 : strides.in0_stride[idx];
    strides.in1_stride[idx] = (in1_shape[idx] == 1) ? 0 : strides.in1_stride[idx];
  }
  return strides;
}
template <BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t>
cudaError_t BinaryWithBroadcast(BinaryFunc<OP, In0_t, In1_t, Out_t> func, ScalarOption scalar_option,
                                const size_t out_num, const std::vector<int64_t> &in0_shape,
                                const std::vector<int64_t> &in1_shape, const std::vector<int64_t> &out_shape,
                                In0_t *in0, In1_t *in1, Out_t *out, size_t device_id, cudaStream_t cuda_stream) {
  size_t vec_thread_num = 8 * 8 * 32;
  const size_t dim_size = out_shape.size();
  constexpr size_t vec_num = VecSize<In0_t, In1_t, Out_t>();
  size_t total_threads = out_num / out_shape.back();
  if (out_num > vec_thread_num && vec_num > 1) {
    if (out_shape.back() == 2) {
      BinaryBroadcastStrideInfo strides = BinaryBroadcastCalStride(dim_size, in0_shape, in1_shape, out_shape, 2);
      size_t thread_num = total_threads > 1024 ? 1024 : total_threads;
      BinaryBroadcastVecWithoutTailCuda<OP, In0_t, In1_t, Out_t, 2>
        <<<CUDA_BLOCKS_CAL(device_id, total_threads, thread_num), thread_num, 0, cuda_stream>>>(
          func, scalar_option, dim_size, total_threads, strides, in0, in1, out);
      CHECK_CUDA_LAUNCH_SUCCESS();
    } else if (out_shape.back() == 3) {
      BinaryBroadcastStrideInfo strides = BinaryBroadcastCalStride(dim_size, in0_shape, in1_shape, out_shape, 3);
      size_t total_threads = out_shape[0] * strides.out_stride[0];
      size_t thread_num = total_threads > 1024 ? 1024 : total_threads;
      BinaryBroadcastVecWithoutTailCuda<OP, In0_t, In1_t, Out_t, 3>
        <<<CUDA_BLOCKS_CAL(device_id, total_threads, thread_num), thread_num, 0, cuda_stream>>>(
          func, scalar_option, dim_size, total_threads, strides, in0, in1, out);
      CHECK_CUDA_LAUNCH_SUCCESS();
    } else {
      BinaryBroadcastStrideInfo strides = BinaryBroadcastCalStride(dim_size, in0_shape, in1_shape, out_shape, vec_num);
      size_t step = out_shape.back() / vec_num;
      total_threads *= step;
      size_t tail_num = out_shape.back() % vec_num;
      size_t thread_num = total_threads > 1024 ? 1024 : total_threads;
      if (tail_num == 0) {
        BinaryBroadcastVecWithoutTailCuda<OP, In0_t, In1_t, Out_t, vec_num>
          <<<CUDA_BLOCKS_CAL(device_id, total_threads, thread_num), thread_num, 0, cuda_stream>>>(
            func, scalar_option, dim_size, total_threads, strides, in0, in1, out);
        CHECK_CUDA_LAUNCH_SUCCESS();
      } else {
        BinaryBroadcastVecWithTailCuda<OP, In0_t, In1_t, Out_t, vec_num>
          <<<CUDA_BLOCKS_CAL(device_id, total_threads, thread_num), thread_num, 0, cuda_stream>>>(
            func, scalar_option, dim_size, total_threads, step, tail_num, strides, in0, in1, out);
        CHECK_CUDA_LAUNCH_SUCCESS();
      }
    }
  } else {
    BinaryBroadcastStrideInfo strides = BinaryBroadcastCalStride(dim_size, in0_shape, in1_shape, out_shape, 1);
    total_threads *= out_shape.back();
    size_t thread_num = total_threads > 1024 ? 1024 : total_threads;
    BinaryWithBroadcastNoVecCuda<OP, In0_t, In1_t, Out_t>
      <<<CUDA_BLOCKS_CAL(device_id, total_threads, thread_num), thread_num, 0, cuda_stream>>>(
        func, dim_size, total_threads, strides, in0, in1, out);
    CHECK_CUDA_LAUNCH_SUCCESS();
  }
}

template <BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t>
cudaError_t BinaryWithoutBroadcast(BinaryFunc<OP, In0_t, In1_t, Out_t> func, ScalarOption scalar_option, size_t nums,
                                   Out_t *out, In0_t *in0, In1_t *in1, size_t device_id, cudaStream_t cuda_stream) {
  size_t thread_num = nums > 1024 ? 1024 : nums;
  BinaryWithoutBroadcastNoVecCuda<OP, In0_t, In1_t, Out_t>
    <<<CUDA_BLOCKS_CAL(device_id, nums, thread_num), thread_num, 0, cuda_stream>>>(func, scalar_option, nums, in0, in1,
                                                                                   out);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template <enum BinaryOpType OP, typename In0_t, typename In1_t, typename Out_t>
cudaError_t BinaryOpWithBroadcastCudaFunc(const bool is_broadcast, const std::vector<int64_t> &in0_shape,
                                          const std::vector<int64_t> &in1_shape, const std::vector<int64_t> &out_shape,
                                          In0_t *in0, In1_t *in1, Out_t *out, size_t device_id,
                                          cudaStream_t cuda_stream) {
  BinaryFunc<OP, In0_t, In1_t, Out_t> func;
  size_t out_num = 1;
  for (auto val : out_shape) {
    out_num *= val;
  }
  ScalarOption scalar_option = ScalarOption::NoScalar;
  if (is_broadcast) {
    if (in0_shape.back() == 1) {
      scalar_option = ScalarOption::In0Scalar;
    } else if (in1_shape.back() == 1) {
      scalar_option = ScalarOption::In1Scalar;
    }
    return BinaryWithBroadcast<OP, In0_t, In1_t, Out_t>(func, scalar_option, out_num, in0_shape, in1_shape, out_shape,
                                                        in0, in1, out, device_id, cuda_stream);
  } else {
    if (in0_shape.size() == 1 && in0_shape[0] == 1) {
      scalar_option = ScalarOption::In0Scalar;
    }
    if (in1_shape.size() == 1 && in1_shape[0] == 1) {
      scalar_option = ScalarOption::In1Scalar;
    }
    return BinaryWithoutBroadcast<OP, In0_t, In1_t, Out_t>(func, scalar_option, out_num, out, in0, in1, device_id,
                                                           cuda_stream);
  }
}

#define REGISTER_BINARY_OP_CUDA_FUNC_BOOL_TYPE(op)                                                         \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, bool, bool, bool>(                \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, bool *in0, bool *in1, bool *out, size_t device_id,              \
    cudaStream_t cuda_stream);

#define REGISTER_BINARY_OP_CUDA_FUNC_INT_TYPE(op)                                                          \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int8_t, int8_t, int8_t>(          \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, int8_t *in0, int8_t *in1, int8_t *out, size_t device_id,        \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint8_t, uint8_t, uint8_t>(       \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, uint8_t *in0, uint8_t *in1, uint8_t *out, size_t device_id,     \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int16_t, int16_t, int16_t>(       \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, int16_t *in0, int16_t *in1, int16_t *out, size_t device_id,     \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint16_t, uint16_t, uint16_t>(    \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, uint16_t *in0, uint16_t *in1, uint16_t *out, size_t device_id,  \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int32_t, int32_t, int32_t>(       \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, int32_t *in0, int32_t *in1, int32_t *out, size_t device_id,     \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint32_t, uint32_t, uint32_t>(    \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, uint32_t *in0, uint32_t *in1, uint32_t *out, size_t device_id,  \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, int64_t, int64_t, int64_t>(       \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, int64_t *in0, int64_t *in1, int64_t *out, size_t device_id,     \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, uint64_t, uint64_t, uint64_t>(    \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, uint64_t *in0, uint64_t *in1, uint64_t *out, size_t device_id,  \
    cudaStream_t cuda_stream)

#define REGISTER_BINARY_OP_CUDA_FUNC_FLOAT_TYPE(op)                                                        \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, double, double, double>(          \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, double *in0, double *in1, double *out, size_t device_id,        \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, float, float, float>(             \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, float *in0, float *in1, float *out, size_t device_id,           \
    cudaStream_t cuda_stream);                                                                             \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, half, half, half>(                \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape, \
    const std::vector<int64_t> &out_shape, half *in0, half *in1, half *out, size_t device_id,              \
    cudaStream_t cuda_stream);

#define REGISTER_BINARY_OP_CUDA_FUNC_COMPLEX_TYPE(op)                                                                 \
  template CUDA_LIB_EXPORT cudaError_t                                                                                \
  BinaryOpWithBroadcastCudaFunc<op, Complex<float>, Complex<float>, Complex<float>>(                                  \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, Complex<float> *in0, Complex<float> *in1, Complex<float> *out,             \
    size_t device_id, cudaStream_t cuda_stream);                                                                      \
  template CUDA_LIB_EXPORT cudaError_t                                                                                \
  BinaryOpWithBroadcastCudaFunc<op, Complex<double>, Complex<double>, Complex<double>>(                               \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, Complex<double> *in0, Complex<double> *in1, Complex<double> *out,          \
    size_t device_id, cudaStream_t cuda_stream);                                                                      \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, Complex<float>, float, Complex<float>>(      \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, Complex<float> *in0, float *in1, Complex<float> *out, size_t device_id,    \
    cudaStream_t cuda_stream);                                                                                        \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, float, Complex<float>, Complex<float>>(      \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, float *in0, Complex<float> *in1, Complex<float> *out, size_t device_id,    \
    cudaStream_t cuda_stream);                                                                                        \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, double, Complex<double>, Complex<double>>(   \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, double *in0, Complex<double> *in1, Complex<double> *out, size_t device_id, \
    cudaStream_t cuda_stream);                                                                                        \
  template CUDA_LIB_EXPORT cudaError_t BinaryOpWithBroadcastCudaFunc<op, Complex<double>, double, Complex<double>>(   \
    const bool is_broadcast, const std::vector<int64_t> &in0_shape, const std::vector<int64_t> &in1_shape,            \
    const std::vector<int64_t> &out_shape, Complex<double> *in0, double *in1, Complex<double> *out, size_t device_id, \
    cudaStream_t cuda_stream)
#endif
