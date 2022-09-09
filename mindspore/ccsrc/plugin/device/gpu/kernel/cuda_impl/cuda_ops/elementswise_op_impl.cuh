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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_UTILS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_UTILS_IMPL_CUH_

#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
namespace cuda {
namespace elementwise {
// An empirical parameter
// In the mainstream GPU architecture, the maximum number of registers per block is 64K,
// the maximum number of registers that can be used by each thread is 255.
// So, kThreadsPerBlock = 64 * 1024 / 255 = 256.
// Refer from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
constexpr uint kThreadsPerBlock = 256;
// An empirical parameter
constexpr uint kWaves = 32;
constexpr uint kStride = 2;

struct CudaConfig {
  int dev_{0};
  int sm_nums_{1};
  int max_threads_{1};
};

// Get some necessary hardware config.
inline cudaError_t GetCurrentConfig(CudaConfig *config) {
  // 1. Get current device.
  // 2. Get current sm_nums
  // 3. Get the maximum resident threads in per multiprocessor.
  int dev;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) {
    return err;
  }
  int sm_nums;
  err = cudaDeviceGetAttribute(&sm_nums, cudaDevAttrMultiProcessorCount, dev);
  if (err != cudaSuccess) {
    return err;
  }
  int max_threads;
  err = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
  if (err != cudaSuccess) {
    return err;
  }
  config->dev_ = dev;
  config->sm_nums_ = sm_nums;
  config->max_threads_ = max_threads;
  return err;
}

// Get best blocks basing on parallel data size for current hardware, adaptively.
inline uint GetBestBlocks(uint n, const CudaConfig &config) {
  uint best_blocks =
    std::max<uint>(1, std::min<uint>((n + kThreadsPerBlock - 1) / kThreadsPerBlock,
                                     config.sm_nums_ * config.max_threads_ / kThreadsPerBlock * kWaves));
  return best_blocks;
}

template <typename T, uint vec_size>
struct VectorizedTraitType {
  using type = typename std::aligned_storage<vec_size * sizeof(T), vec_size * sizeof(T)>::type;
};

template <typename T, uint vec_size>
using VectorizedType = typename VectorizedTraitType<T, vec_size>::type;

template <typename T, uint VecSize>
union Vec {
  static_assert(sizeof(VectorizedType<T, VecSize>) == sizeof(T) * VecSize, "data can not be aligned.");
  __device__ Vec() {}
  VectorizedType<T, VecSize> storage_;
  T elements_[VecSize];
};

template <typename T, uint VecSize>
struct alignas(sizeof(T) * VecSize) AlignVec {
  T elements_[VecSize];
};

constexpr uint kMaxVecBytes = 128 / 8;
constexpr uint kMaxVecSize = 8;

constexpr uint MsMin(uint a, uint b) { return a < b ? a : b; }

template <typename T>
constexpr uint VecSize() {
  return MsMin(kMaxVecBytes / sizeof(T), kMaxVecSize);
}

template <typename T, typename U, typename... Args>
constexpr uint VecSize() {
  return MsMin(VecSize<T>(), VecSize<U, Args...>());
}

template <typename T>
class CheckApply2 {
  typedef char apply_unit;
  struct apply_struct {
    char x_[2];
  };

  template <typename IN3>
  static apply_unit check(decltype(&IN3::Apply2));
  template <typename IN3>
  static apply_struct check(...);

 public:
  enum { value = sizeof(check<T>(0)) == sizeof(char) };
};

template <uint vec_size>
bool IsAligned() {
  return true;
}

template <uint vec_size, typename T, typename... Args>
bool IsAligned(const T *ptr, const Args *...others) {
  return reinterpret_cast<uintptr_t>(ptr) % sizeof(Vec<T, vec_size>) == 0 && IsAligned<vec_size, Args...>(others...);
}

template <uint vec_size, typename FunctorT, typename OUT, typename... IN>
__device__ typename std::enable_if<CheckApply2<FunctorT>::value == true && vec_size % kStride == 0,
                                   AlignVec<OUT, vec_size>>::type
ApplyVec(const FunctorT &functor, const IN... in[vec_size]) {
  AlignVec<OUT, vec_size> ret;

#pragma unroll
  for (uint j = 0; j < vec_size; j += kStride) {
    functor.Apply2(ret.elements_ + j, (in + j)...);
  }
  return ret;
}

template <uint vec_size, typename FunctorT, typename OUT, typename... IN>
__device__ typename std::enable_if<CheckApply2<FunctorT>::value == false || vec_size % kStride != 0,
                                   AlignVec<OUT, vec_size>>::type
ApplyVec(const FunctorT &functor, const IN... in[vec_size]) {
  AlignVec<OUT, vec_size> ret;
#pragma unroll
  for (uint j = 0; j < vec_size; ++j) {
    ret.elements_[j] = functor((in[j])...);
  }
  return ret;
}

template <uint vec_size, bool tail, typename Factory, typename OUT, typename... IN>
__global__ void __launch_bounds__(kThreadsPerBlock)
  DoApply(Factory factory, uint vec_nums, AlignVec<OUT, vec_size> *vec_out, const AlignVec<IN, vec_size> *...vec_in,
          uint tail_nums, OUT *tail_out, const IN *...tail_in) {
  auto functor = factory();
  const uint global_tid = blockIdx.x * kThreadsPerBlock + threadIdx.x;
  for (uint i = global_tid; i < vec_nums; i += blockDim.x * gridDim.x) {
    vec_out[i] = ApplyVec<vec_size, decltype(functor), OUT, IN...>(functor, (vec_in[i].elements_)...);
  }
  if (tail && global_tid < tail_nums) {
    tail_out[global_tid] = functor((tail_in[global_tid])...);
  }
}

template <uint vec_size, typename Factory, typename OUT, typename... IN>
cudaError_t LaunchKernel(Factory factory, uint nums, OUT *out, const IN *...in, cudaStream_t stream) {
  const uint vec_nums = nums / vec_size;
  const uint tail_offset = vec_nums * vec_size;
  const uint tail_nums = nums - tail_offset;
  CudaConfig config;
  cudaError_t err = GetCurrentConfig(&config);
  if (err != cudaSuccess) {
    return err;
  }
  uint num_blocks = GetBestBlocks(vec_nums, config);
  dim3 block{kThreadsPerBlock};
  dim3 grid{uint(num_blocks)};
  if (tail_nums > 0) {
    auto func = DoApply<vec_size, true, Factory, OUT, IN...>;
    func<<<grid, block, 0, stream>>>(factory, vec_nums, reinterpret_cast<AlignVec<OUT, vec_size> *>(out),
                                    (reinterpret_cast<const AlignVec<IN, vec_size> *>(in))..., tail_nums,
                                    out + tail_offset, (in + tail_offset)...);
  } else {
    auto func = DoApply<vec_size, false, Factory, OUT, IN...>;
    func<<<grid, block, 0, stream>>>(factory, vec_nums, reinterpret_cast<AlignVec<OUT, vec_size> *>(out),
                                    (reinterpret_cast<const AlignVec<IN, vec_size> *>(in))..., tail_nums,
                                    out + tail_offset, (in + tail_offset)...);
  }
  return cudaPeekAtLastError();
}

template <typename Factory, typename OUT, typename... IN>
struct DoLaunch {
  static cudaError_t Launch(Factory factory, uint n, OUT *out, const IN *...in, cudaStream_t stream) {
    constexpr uint max_pack_size = VecSize<OUT, IN...>();
    if (IsAligned<max_pack_size, OUT, IN...>(out, in...)) {
      return LaunchKernel<max_pack_size, Factory, OUT, IN...>(factory, n, out, in..., stream);
    }
    return LaunchKernel<1, Factory, OUT, IN...>(factory, n, out, in..., stream);
  }
};

template <typename FunctorT>
struct TransitFactory {
  explicit TransitFactory(FunctorT functor) : transit_impl_(functor) {}
  __device__ FunctorT operator()() const { return transit_impl_; }

 private:
  FunctorT transit_impl_;
};

// API elementwise for input: a, output: out.
template <typename Factory, typename OUT, typename IN>
inline cudaError_t UnaryTransit(Factory factory, uint n, OUT *out, const IN *in, cudaStream_t stream) {
  return DoLaunch<Factory, OUT, IN>::Launch(factory, n, out, in, stream);
}

template <typename FunctorT, typename OUT, typename IN>
inline cudaError_t Unary(FunctorT functor, uint n, OUT *out, const IN *in, cudaStream_t stream) {
  return UnaryTransit(TransitFactory<FunctorT>(functor), n, out, in, stream);
}

template <typename Factory, typename OUT, typename IN, typename IN2>
inline cudaError_t BinaryTransit(Factory factory, uint n, OUT *out, const IN *in, const IN2 *in2, cudaStream_t stream) {
  return DoLaunch<Factory, OUT, IN, IN2>::Launch(factory, n, out, in, in2, stream);
}

// API elementwise for input: [a, b], output: out.
template <typename FunctorT, typename OUT, typename IN, typename IN2>
inline cudaError_t Binary(FunctorT functor, uint n, OUT *out, const IN *in, const IN2 *in2, cudaStream_t stream) {
  return BinaryTransit(TransitFactory<FunctorT>(functor), n, out, in, in2, stream);
}

template <typename Factory, typename OUT, typename IN, typename IN2, typename IN3>
inline cudaError_t TernaryTransit(Factory factory, uint n, OUT *out, const IN *in, const IN2 *in2, const IN3 *in3,
                                  cudaStream_t stream) {
  return DoLaunch<Factory, OUT, IN, IN2, IN3>::Launch(factory, n, out, in, in2, in3, stream);
}

// API elementwise for input: [a, b, c], output: out.
template <typename FunctorT, typename OUT, typename IN, typename IN2, typename IN3>
inline cudaError_t Ternary(FunctorT functor, uint n, OUT *out, const IN *in, const IN2 *in2, const IN3 *in3,
                           cudaStream_t stream) {
  return TernaryTransit(TransitFactory<FunctorT>(functor), n, out, in, in2, in3, stream);
}
}  // namespace elementwise
}  // namespace cuda
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ELEMENTWISE_UTILS_IMPL_CUH_
