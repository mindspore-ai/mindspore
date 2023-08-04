/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include "loss_with_reduction_impl.cuh"
#include "util.cuh"

inline __device__ float logT(float x) { return logf(x); }
inline __device__ half logT(half x) { return hlog(x); }
inline __device__ float castT(float ref, int x) { return __int2float_rd(x); }
inline __device__ half castT(half ref, int x) { return __int2half_rd(x); }
inline __device__ float maxT(float a, float b) { return fmaxf(a, b); }
inline __device__ half maxT(half a, half b) { return a > b ? a : b; }

template <typename T>
__global__ void Copy(T *loss, T *tmp_loss, ReductionMode reduction, int input_size) {
  loss[0] += tmp_loss[0];
  if (reduction == ReductionMode::kMean) {
    loss[0] /= castT(loss[0], input_size);
  }
}

template <typename T>
__global__ void AddTile(T *tmp_loss, int index) {
  tmp_loss[0] += tmp_loss[index];
}
template <typename T>
__global__ void PartialSum(T *tmp_loss, int stride) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += blockDim.x * gridDim.x) {
    tmp_loss[i] += tmp_loss[i + stride];
  }
}

template <typename T, typename S>
__device__ void MultiplyDevice(const S a, const T b, T *out) {
  *out = a * b;
}

template <>
__device__ void MultiplyDevice(const half a, const float b, float *out) {
  // cast a to float for calculation
  float a_float = __half2float(a);
  *out = a_float * b;
}

template <>
__device__ void MultiplyDevice(const float a, const half b, half *out) {
  // cast b to float for calculation
  float b_float = __half2float(b);
  float out_float = a * b_float;
  *out = __float2half(out_float);
}

template <typename T, typename S>
__device__ void Divide(const T *numerator, const S *denominator, T *result) {
  result[0] = numerator[0] / denominator[0];
}

template <>
__device__ void Divide(const float *numerator, const half *denominator, float *result) {
  float denom_float = __half2float(denominator[0]);

  result[0] = numerator[0] / denom_float;
}

template <>
__device__ void Divide(const half *numerator, const float *denominator, half *result) {
  float numer_float = __half2float(numerator[0]);

  float result_float = numer_float / denominator[0];

  result[0] = __float2half(result_float);
}

template <unsigned int BlockDimX, typename T>
__device__ __forceinline__ void WarpReduce(T *shared_data, const int tid) {
  T local_data = shared_data[tid];
  if (BlockDimX >= 32) {
    local_data = local_data + __shfl_down_sync(0xFFFFFFFF, local_data, 16);
  }
  if (BlockDimX >= 16) {
    local_data = local_data + __shfl_down_sync(0xFFFFFFFF, local_data, 8);
  }
  if (BlockDimX >= 8) {
    local_data = local_data + __shfl_down_sync(0xFFFFFFFF, local_data, 4);
  }
  if (BlockDimX >= 4) {
    local_data = local_data + __shfl_down_sync(0xFFFFFFFF, local_data, 2);
  }
  if (BlockDimX >= 2) {
    local_data = local_data + __shfl_down_sync(0xFFFFFFFF, local_data, 1);
  }
  if (tid == 0) {
    shared_data[tid] = local_data;
  }
}

template <unsigned int BlockDimX, typename T, typename S>
__device__ __forceinline__ void BinaryWarpReduce(T *shared_data0, S *shared_data1, const int tid) {
  T local_data0 = shared_data0[tid];
  S local_data1 = shared_data1[tid];
  if (BlockDimX >= 32) {
    local_data0 = local_data0 + __shfl_down_sync(0xFFFFFFFF, local_data0, 16);
    local_data1 = local_data1 + __shfl_down_sync(0xFFFFFFFF, local_data1, 16);
  }
  if (BlockDimX >= 16) {
    local_data0 = local_data0 + __shfl_down_sync(0xFFFFFFFF, local_data0, 8);
    local_data1 = local_data1 + __shfl_down_sync(0xFFFFFFFF, local_data1, 8);
  }
  if (BlockDimX >= 8) {
    local_data0 = local_data0 + __shfl_down_sync(0xFFFFFFFF, local_data0, 4);
    local_data1 = local_data1 + __shfl_down_sync(0xFFFFFFFF, local_data1, 4);
  }
  if (BlockDimX >= 4) {
    local_data0 = local_data0 + __shfl_down_sync(0xFFFFFFFF, local_data0, 2);
    local_data1 = local_data1 + __shfl_down_sync(0xFFFFFFFF, local_data1, 2);
  }
  if (BlockDimX >= 2) {
    local_data0 = local_data0 + __shfl_down_sync(0xFFFFFFFF, local_data0, 1);
    local_data1 = local_data1 + __shfl_down_sync(0xFFFFFFFF, local_data1, 1);
  }
  if (tid == 0) {
    shared_data0[tid] = local_data0;
    shared_data1[tid] = local_data1;
  }
}

template <unsigned int BlockDimX, typename T>
__device__ __forceinline__ void BlockReduce(T *shared_data, const unsigned int tid) {
  if (BlockDimX >= 1024) {
    if (tid < 512) {
      shared_data[tid] = shared_data[tid] + shared_data[tid + 512];
    }
    __syncthreads();
  }
  if (BlockDimX >= 512) {
    if (tid < 256) {
      shared_data[tid] = shared_data[tid] + shared_data[tid + 256];
    }
    __syncthreads();
  }
  if (BlockDimX >= 256) {
    if (tid < 128) {
      shared_data[tid] = shared_data[tid] + shared_data[tid + 128];
    }
    __syncthreads();
  }
  if (BlockDimX >= 128) {
    if (tid < 64) {
      shared_data[tid] = shared_data[tid] + shared_data[tid + 64];
    }
    __syncthreads();
  }
  if (BlockDimX >= 64) {
    if (tid < 32) {
      shared_data[tid] = shared_data[tid] + shared_data[tid + 32];
    }
  }
  __syncthreads();

  if (tid < 32) WarpReduce<BlockDimX>(shared_data, tid);

  __syncthreads();
}

template <unsigned int BlockDimX, typename T, typename S>
__device__ __forceinline__ void BinaryBlockReduce(T *shared_data0, S *shared_data1, const unsigned int tid) {
  if (BlockDimX >= 1024) {
    if (tid < 512) {
      shared_data0[tid] = shared_data0[tid] + shared_data0[tid + 512];
      shared_data1[tid] = shared_data1[tid] + shared_data1[tid + 512];
    }
    __syncthreads();
  }
  if (BlockDimX >= 512) {
    if (tid < 256) {
      shared_data0[tid] = shared_data0[tid] + shared_data0[tid + 256];
      shared_data1[tid] = shared_data1[tid] + shared_data1[tid + 256];
    }
    __syncthreads();
  }
  if (BlockDimX >= 256) {
    if (tid < 128) {
      shared_data0[tid] = shared_data0[tid] + shared_data0[tid + 128];
      shared_data1[tid] = shared_data1[tid] + shared_data1[tid + 128];
    }
    __syncthreads();
  }
  if (BlockDimX >= 128) {
    if (tid < 64) {
      shared_data0[tid] = shared_data0[tid] + shared_data0[tid + 64];
      shared_data1[tid] = shared_data1[tid] + shared_data1[tid + 64];
    }
    __syncthreads();
  }
  if (BlockDimX >= 64) {
    if (tid < 32) {
      shared_data0[tid] = shared_data0[tid] + shared_data0[tid + 32];
      shared_data1[tid] = shared_data1[tid] + shared_data1[tid + 32];
    }
  }
  __syncthreads();

  if (tid < 32) BinaryWarpReduce<BlockDimX>(shared_data0, shared_data1, tid);

  __syncthreads();
}

template <unsigned int BlockDimX, typename T>
__inline__ __device__ void Reduce(T *output, T *shared_data, const unsigned int tid) {
  BlockReduce<BlockDimX>(shared_data, tid);

  if (tid == 0) {
    MsAtomicAdd(output, shared_data[0]);
  }
}

template <unsigned int BlockDimX, typename T, typename S>
__inline__ __device__ void BinaryReduce(T *output0, S *output1, T *shared_data0, S *shared_data1,
                                        const unsigned int tid) {
  BinaryBlockReduce<BlockDimX>(shared_data0, shared_data1, tid);

  if (tid == 0) {
    MsAtomicAdd(output0, shared_data0[0]);
    MsAtomicAdd(output1, shared_data1[0]);
  }
}

template <unsigned int BlockDimX, typename T, typename S, unsigned int sharedSize>
__global__ void NLLLossNativeKernel(const T *logits, const int32_t *labels, const S *weights, T *loss, S *total_weight,
                                    unsigned int label_size, unsigned int num_classes, int32_t ignore_index) {
  unsigned int tid = threadIdx.x;
  const S zero = static_cast<S>(0);
  const S one = static_cast<S>(1);
  __shared__ S shared_total_weight[sharedSize];
  shared_total_weight[tid] = zero;
  if (tid == 0 && blockIdx.x == 0) {
    total_weight[0] = zero;
  }

  for (unsigned int gid = blockIdx.x * BlockDimX + tid, gridSize = BlockDimX * gridDim.x; gid < label_size;
       gid += gridSize) {
    int32_t label = labels[gid];
    if (label != ignore_index) {
      CUDA_KERNEL_ASSERT(label >= 0 && label < num_classes);
      S weight = weights ? weights[label] : one;
      T logit;
      MultiplyDevice(weight, -(logits[gid * num_classes + label]), &logit);
      loss[gid] = logit;
      shared_total_weight[tid] = shared_total_weight[tid] + weight;
    }
  }
  __syncthreads();
  Reduce<BlockDimX>(total_weight, shared_total_weight, tid);
}

template <unsigned int BlockDimX, typename T, typename S, unsigned int sharedSize0, unsigned int sharedSize1>
__global__ void NLLLossReduceKernel(const T *logits, const int32_t *labels, const S *weights, T *loss, S *total_weight,
                                    unsigned int label_size, unsigned int num_classes, int32_t ignore_index,
                                    bool mean) {
  unsigned int tid = threadIdx.x;
  const S one = static_cast<S>(1);
  __shared__ T shared_loss[sharedSize0];
  __shared__ S shared_total_weight[sharedSize1];
  shared_loss[tid] = static_cast<T>(0);
  shared_total_weight[tid] = static_cast<S>(0);
  if (tid == 0 && blockIdx.x == 0) {
    loss[0] = static_cast<S>(0);
    total_weight[0] = static_cast<S>(0);
  }

  for (unsigned int gid = blockIdx.x * BlockDimX + tid, gridSize = BlockDimX * gridDim.x; gid < label_size;
       gid += gridSize) {
    int32_t label = labels[gid];
    if (label != ignore_index) {
      CUDA_KERNEL_ASSERT(label >= 0 && label < num_classes);
      S weight = weights ? weights[label] : one;
      T logit;
      MultiplyDevice(weight, -(logits[gid * num_classes + label]), &logit);
      shared_loss[tid] = shared_loss[tid] + logit;
      shared_total_weight[tid] = shared_total_weight[tid] + weight;
    }
  }
  __syncthreads();
  BinaryReduce<BlockDimX>(loss, total_weight, shared_loss, shared_total_weight, tid);
  if (mean && tid == 0) {
    __syncthreads();
    Divide(loss, total_weight, loss);
  }
}

template <typename T>
__global__ void LossInitKernel(T *loss) {
  loss[0] = static_cast<T>(0.);
}

template <typename T>
__global__ void InitZero(T *array, int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    array[i] = static_cast<T>(0.);
  }
}

template <typename T>
__global__ void KLDivLossKernel(const int input_size, const ReductionMode reduction, const T *input_x, const T *input_y,
                                T *loss, T *tmp_loss) {
  T epsilon = 1e-6;
  if (reduction == ReductionMode::kNone) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      T value = input_y[i] * (logT(denominator) - input_x[i]);
      loss[i] = value;
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      T value = input_y[i] * (logT(denominator) - input_x[i]);
      tmp_loss[i] = value;
    }
  }
}

template <typename T>
cudaError_t KLDivLoss(const int &input_size, const ReductionMode &reduction, const T *input_x, const T *input_y,
                      T *loss, T *tmp_loss, cudaStream_t stream) {
  LossInitKernel<<<1, 1, 0, stream>>>(loss);
  KLDivLossKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x, input_y, loss,
                                                                      tmp_loss);
  if (reduction != ReductionMode::kNone) {
    if (input_size % 2 == 1) {
      AddTile<<<1, 1, 0, stream>>>(tmp_loss, input_size - 1);
    }
    for (int stride = input_size / 2; stride > 0; stride >>= 1) {
      PartialSum<<<GET_BLOCKS(stride), GET_THREADS, 0, stream>>>(tmp_loss, stride);
      if (stride > 2 && stride % 2 == 1) {
        AddTile<<<1, 1, 0, stream>>>(tmp_loss, stride - 1);
      }
    }
    Copy<<<1, 1, 0, stream>>>(loss, tmp_loss, reduction, input_size);
  }
  return GetCudaStatus();
}

template <typename T>
__global__ void KLDivLossGradKernel(const int input_size, const ReductionMode reduction, const T *input_x,
                                    const T *input_y, const T *dloss, T *dx) {
  T epsilon = 1e-6;
  if (reduction == ReductionMode::kNone) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      dx[i] = -input_y[i] * dloss[i];
    }
  } else {
    T dloss1 = dloss[0];
    if (reduction == ReductionMode::kMean) {
      dloss1 = dloss[0] / castT(dloss[0], input_size);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      dx[i] = -input_y[i] * dloss1;
    }
  }
}

template <typename T>
cudaError_t KLDivLossGrad(const int &input_size, const ReductionMode &reduction, const T *input_x, const T *input_y,
                          const T *dloss, T *dx, cudaStream_t stream) {
  KLDivLossGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x, input_y,
                                                                          dloss, dx);
  return GetCudaStatus();
}

template <typename T>
__global__ void BinaryCrossEntropyLossKernel(const int input_size, const ReductionMode reduction, const T *input_x,
                                             const T *input_y, const T *weight, T *loss, T *tmp_loss) {
  T epsilon = 1e-12;
  T zero = static_cast<T>(0);
  T one = static_cast<T>(1);
  if (reduction == ReductionMode::kNone && weight != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      CUDA_KERNEL_ASSERT(input_x[i] >= zero && input_x[i] <= one);
      T value =
        -weight[i] * (input_y[i] * logT(input_x[i] + epsilon) + (one - input_y[i]) * logT(one - input_x[i] + epsilon));
      loss[i] = value;
    }
  } else if (reduction == ReductionMode::kNone && weight == nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      CUDA_KERNEL_ASSERT(input_x[i] >= zero && input_x[i] <= one);
      T value = -(input_y[i] * logT(input_x[i] + epsilon) + (one - input_y[i]) * logT(one - input_x[i] + epsilon));
      loss[i] = value;
    }
  } else if (reduction != ReductionMode::kNone && weight != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      CUDA_KERNEL_ASSERT(input_x[i] >= zero && input_x[i] <= one);
      T value =
        -weight[i] * (input_y[i] * logT(input_x[i] + epsilon) + (one - input_y[i]) * logT(one - input_x[i] + epsilon));
      tmp_loss[i] = value;
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      CUDA_KERNEL_ASSERT(input_x[i] >= zero && input_x[i] <= one);
      T value = -(input_y[i] * logT(input_x[i] + epsilon) + (one - input_y[i]) * logT(one - input_x[i] + epsilon));
      tmp_loss[i] = value;
    }
  }
}

template <typename T>
cudaError_t BinaryCrossEntropyLoss(const int &input_size, const ReductionMode &reduction, const T *input_x,
                                   const T *input_y, const T *weight, T *loss, T *tmp_loss, cudaStream_t stream) {
  LossInitKernel<<<1, 1, 0, stream>>>(loss);
  BinaryCrossEntropyLossKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x,
                                                                                   input_y, weight, loss, tmp_loss);
  if (reduction != ReductionMode::kNone) {
    if (input_size % 2 == 1) {
      AddTile<<<1, 1, 0, stream>>>(tmp_loss, input_size - 1);
    }
    for (int stride = input_size / 2; stride > 0; stride >>= 1) {
      PartialSum<<<GET_BLOCKS(stride), GET_THREADS, 0, stream>>>(tmp_loss, stride);
      if (stride > 2 && stride % 2 == 1) {
        AddTile<<<1, 1, 0, stream>>>(tmp_loss, stride - 1);
      }
    }
    Copy<<<1, 1, 0, stream>>>(loss, tmp_loss, reduction, input_size);
  }
  return GetCudaStatus();
}

template <typename T>
__global__ void BinaryCrossEntropyLossGradKernel(const int input_size, const ReductionMode reduction, const T *input_x,
                                                 const T *input_y, const T *weight, const T *dloss, T *dx) {
  T epsilon = 1e-12;
  T one = static_cast<T>(1);
  if (reduction == ReductionMode::kNone) {
    if (weight != nullptr) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
        T denominator = maxT(input_x[i] * (one - input_x[i]), epsilon);
        T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss[i];
      }
    } else {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
        T denominator = maxT(input_x[i] * (one - input_x[i]), epsilon);
        T value = (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss[i];
      }
    }
  } else {
    T dloss1 = dloss[0];
    if (reduction == ReductionMode::kMean) {
      dloss1 = dloss[0] / castT(dloss[0], input_size);
    }
    if (weight != nullptr) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
        T denominator = maxT(input_x[i] * (one - input_x[i]), epsilon);
        T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss1;
      }
    } else {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
        T denominator = maxT(input_x[i] * (one - input_x[i]), epsilon);
        T value = (input_x[i] - input_y[i]) / denominator;
        dx[i] = value * dloss1;
      }
    }
  }
}

template <typename T>
cudaError_t BinaryCrossEntropyLossGrad(const int &input_size, const ReductionMode &reduction, const T *input_x,
                                       const T *input_y, const T *weight, const T *dloss, T *dx, cudaStream_t stream) {
  BinaryCrossEntropyLossGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x,
                                                                                       input_y, weight, dloss, dx);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t NLLLoss(const T *logits, const int32_t *labels, const S *weights, T *loss, S *total_weight,
                    unsigned int label_size, unsigned int num_classes, const ReductionMode reduction,
                    int32_t ignore_index, cudaStream_t stream) {
  const unsigned int Threads = 512;
  if (reduction == ReductionMode::kNone) {
    const unsigned int sharedSize = Threads * sizeof(S) + 1;
    NLLLossNativeKernel<Threads, T, S, sharedSize><<<GET_BLOCKS(label_size), Threads, 0, stream>>>(
      logits, labels, weights, loss, total_weight, label_size, num_classes, ignore_index);
  } else {
    bool mean = (reduction == ReductionMode::kMean);
    const unsigned int sharedSize0 = Threads * sizeof(T) + 1;
    const unsigned int sharedSize1 = Threads * sizeof(S) + 1;
    NLLLossReduceKernel<Threads, T, S, sharedSize0, sharedSize1><<<GET_BLOCKS(label_size), Threads, 0, stream>>>(
      logits, labels, weights, loss, total_weight, label_size, num_classes, ignore_index, mean);
  }
  cudaStreamSynchronize(stream);
  return GetCudaStatus();
}

template <typename T, typename S>
__global__ void NLLLossGradKernel(const int n, const int c, const ReductionMode reduction, const T *input,
                                  const int32_t *target, const S *weight, const S *total_weight, int32_t ignore_index,
                                  const T *dloss, T *dinput) {
  int input_idx;
  int target_class;
  S tmp_quot;
  if (reduction == ReductionMode::kNone) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      target_class = static_cast<int>(target[i]);
      if (target_class == ignore_index) {
        continue;
      }

      input_idx = (i * c) + target_class;

      MultiplyDevice(-weight[target_class], dloss[i], dinput + input_idx);
    }
  } else if (reduction == ReductionMode::kMean) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      target_class = static_cast<int>(target[i]);
      if (target_class == ignore_index) {
        continue;
      }

      input_idx = (i * c) + target_class;

      tmp_quot = (-weight[target_class]) / *total_weight;
      MultiplyDevice(tmp_quot, dloss[0], dinput + input_idx);
    }
  } else if (reduction == ReductionMode::kSum) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      target_class = static_cast<int>(target[i]);
      if (target_class == ignore_index) {
        continue;
      }

      input_idx = (i * c) + target_class;

      MultiplyDevice(-weight[target_class], dloss[0], dinput + input_idx);
    }
  }
}

template <typename T, typename S>
cudaError_t NLLLossGrad(const int n, const int c, const ReductionMode reduction, const T *input, const int32_t *target,
                        const S *weight, const S *total_weight, const T *dloss, T *dinput, int32_t ignore_index,
                        cudaStream_t stream) {
  int input_size = n * c;
  InitZero<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(dinput, input_size);

  NLLLossGradKernel<<<GET_BLOCKS(n), GET_THREADS, 0, stream>>>(n, c, reduction, input, target, weight, total_weight,
                                                               ignore_index, dloss, dinput);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t NLLLoss<half, half>(const half *logits, const int32_t *labels, const half *weights,
                                                         half *loss, half *total_weight, const unsigned int label_size,
                                                         const unsigned int num_classes, const ReductionMode reduction,
                                                         int32_t ignore_index, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t NLLLoss<half, float>(const half *logits, const int32_t *labels,
                                                          const float *weights, half *loss, float *total_weight,
                                                          unsigned int label_size, unsigned int num_classes,
                                                          const ReductionMode reduction, int32_t ignore_index,
                                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t NLLLoss<float, half>(const float *logits, const int32_t *labels,
                                                          const half *weights, float *loss, half *total_weight,
                                                          unsigned int label_size, unsigned int num_classes,
                                                          const ReductionMode reduction, int32_t ignore_index,
                                                          cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t NLLLoss<float, float>(const float *logits, const int32_t *labels,
                                                           const float *weights, float *loss, float *total_weight,
                                                           unsigned int label_size, unsigned int num_classes,
                                                           const ReductionMode reduction, int32_t ignore_index,
                                                           cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t KLDivLoss<float>(const int &input_size, const ReductionMode &reduction,
                                                      const float *input_x, const float *input_y, float *loss,
                                                      float *tmp_loss, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t KLDivLossGrad<float>(const int &input_size, const ReductionMode &reduction,
                                                          const float *input_x, const float *input_y,
                                                          const float *dloss, float *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t KLDivLoss<double>(const int &input_size, const ReductionMode &reduction,
                                                       const double *input_x, const double *input_y, double *loss,
                                                       double *tmp_loss, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t KLDivLossGrad<double>(const int &input_size, const ReductionMode &reduction,
                                                           const double *input_x, const double *input_y,
                                                           const double *dloss, double *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BinaryCrossEntropyLoss<float>(const int &input_size,
                                                                   const ReductionMode &reduction, const float *input_x,
                                                                   const float *input_y, const float *weight,
                                                                   float *loss, float *tmp_loss, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BinaryCrossEntropyLossGrad<float>(const int &input_size,
                                                                       const ReductionMode &reduction,
                                                                       const float *input_x, const float *input_y,
                                                                       const float *weight, const float *dloss,
                                                                       float *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t NLLLossGrad<float, float>(const int n, const int c, const ReductionMode reduction,
                                                               const float *input, const int32_t *target,
                                                               const float *weight, const float *total_weight,
                                                               const float *dloss, float *dinput, int32_t ignore_index,
                                                               cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t NLLLossGrad<float, half>(const int n, const int c, const ReductionMode reduction,
                                                              const float *input, const int32_t *target,
                                                              const half *weight, const half *total_weight,
                                                              const float *dloss, float *dinput, int32_t ignore_index,
                                                              cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t KLDivLoss<half>(const int &input_size, const ReductionMode &reduction,
                                                     const half *input_x, const half *input_y, half *loss,
                                                     half *tmp_loss, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t KLDivLossGrad<half>(const int &input_size, const ReductionMode &reduction,
                                                         const half *input_x, const half *input_y, const half *dloss,
                                                         half *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BinaryCrossEntropyLoss<half>(const int &input_size, const ReductionMode &reduction,
                                                                  const half *input_x, const half *input_y,
                                                                  const half *weight, half *loss, half *tmp_loss,
                                                                  cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BinaryCrossEntropyLossGrad<half>(const int &input_size,
                                                                      const ReductionMode &reduction,
                                                                      const half *input_x, const half *input_y,
                                                                      const half *weight, const half *dloss, half *dx,
                                                                      cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t NLLLossGrad<half, half>(const int n, const int c, const ReductionMode reduction,
                                                             const half *input, const int32_t *target,
                                                             const half *weight, const half *total_weight,
                                                             const half *dloss, half *dinput, int32_t ignore_index,
                                                             cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t NLLLossGrad<half, float>(const int n, const int c, const ReductionMode reduction,
                                                              const half *input, const int32_t *target,
                                                              const float *weight, const float *total_weight,
                                                              const half *dloss, half *dinput, int32_t ignore_index,
                                                              cudaStream_t stream);
