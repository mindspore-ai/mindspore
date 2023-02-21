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

// copy array of equal size
template <typename T>
__global__ void CopyEqual(const T *src, T *dest, const int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    dest[i] = src[i];
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
__global__ void Divide(const T *numerator, const S *denominator, T *result) {
  result[0] = numerator[0] / denominator[0];
}

template <>
__global__ void Divide(const float *numerator, const half *denominator, float *result) {
  float denom_float = __half2float(denominator[0]);

  result[0] = numerator[0] / denom_float;
}

template <>
__global__ void Divide(const half *numerator, const float *denominator, half *result) {
  float numer_float = __half2float(numerator[0]);

  float result_float = numer_float / denominator[0];

  result[0] = __float2half(result_float);
}

template <typename T>
void Sum(T *array, const int &size, cudaStream_t stream) {
  if (size % 2 == 1) {
    AddTile<<<1, 1, 0, stream>>>(array, size - 1);
  }
  for (int stride = size / 2; stride > 0; stride >>= 1) {
    PartialSum<<<GET_BLOCKS(stride), GET_THREADS, 0, stream>>>(array, stride);
    if (stride > 2 && stride % 2 == 1) {
      AddTile<<<1, 1, 0, stream>>>(array, stride - 1);
    }
  }
}

template <typename T, typename S>
void Reduce(T *tmp_loss, const int &size, S *denom, const ReductionMode &reduction, T *output, cudaStream_t stream) {
  // sum losses together
  Sum(tmp_loss, size, stream);

  if (reduction == ReductionMode::kMean) {
    // mean reduction, divide sum by denominator, store result in output
    Divide<<<1, 1, 0, stream>>>(tmp_loss, denom, output);
  } else if (reduction == ReductionMode::kSum) {
    // sum reduction, copy sum to output
    CopyEqual<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(tmp_loss, output, size);
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
void KLDivLoss(const int &input_size, const ReductionMode &reduction, const T *input_x, const T *input_y, T *loss,
               T *tmp_loss, cudaStream_t stream) {
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
void KLDivLossGrad(const int &input_size, const ReductionMode &reduction, const T *input_x, const T *input_y,
                   const T *dloss, T *dx, cudaStream_t stream) {
  KLDivLossGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x, input_y,
                                                                          dloss, dx);
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
void BinaryCrossEntropyLoss(const int &input_size, const ReductionMode &reduction, const T *input_x, const T *input_y,
                            const T *weight, T *loss, T *tmp_loss, cudaStream_t stream) {
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
void BinaryCrossEntropyLossGrad(const int &input_size, const ReductionMode &reduction, const T *input_x,
                                const T *input_y, const T *weight, const T *dloss, T *dx, cudaStream_t stream) {
  BinaryCrossEntropyLossGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x,
                                                                                       input_y, weight, dloss, dx);
}

// helper function to calculate single negative log likelihood
template <typename T, typename S>
__global__ void NLLLossKernel(const int n, const int c, const T *input, const int32_t *target, const S *weight,
                              S *tmp_target_weight, T *output, int *ret_flag) {
  int target_class;
  int input_idx;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    target_class = static_cast<int>(target[i]);
    if (target_class < 0 || target_class > c) {
      *ret_flag = -1;
      return;
    }

    tmp_target_weight[i] = weight[target_class];  // fill tmp_target_weight for later summation

    input_idx = c * i + target_class;

    MultiplyDevice(-weight[target_class], input[input_idx], output + i);
  }
}

template <typename T, typename S>
int NLLLoss(const int n, const int c, const ReductionMode reduction, const T *input, const int32_t *target,
            const S *weight, T *loss, S *total_weight, T *tmp_loss, S *tmp_target_weight, cudaStream_t stream) {
  int *ret_flag_device = nullptr;
  (void)cudaMalloc(&ret_flag_device, sizeof(int));
  (void)cudaMemset(ret_flag_device, 0, sizeof(int));
  int ret_flag_host;
  if (reduction != ReductionMode::kNone) {
    NLLLossKernel<<<GET_BLOCKS(n), GET_THREADS, 0, stream>>>(n, c, input, target, weight, tmp_target_weight, tmp_loss,
                                                             ret_flag_device);
    cudaDeviceSynchronize();
    (void)cudaMemcpy(&ret_flag_host, ret_flag_device, sizeof(int), cudaMemcpyDeviceToHost);
    // sum target weights after populating them
    Sum(tmp_target_weight, n, stream);
    // reduce tmp_loss
    Reduce(tmp_loss, n, tmp_target_weight, reduction, loss, stream);
  } else {
    // no reduction, output directly to loss
    NLLLossKernel<<<GET_BLOCKS(n), GET_THREADS, 0, stream>>>(n, c, input, target, weight, tmp_target_weight, loss,
                                                             ret_flag_device);
    (void)cudaMemcpy(&ret_flag_host, ret_flag_device, sizeof(int), cudaMemcpyDeviceToHost);
    // sum target weights after populatin them
    Sum(tmp_target_weight, n, stream);
  }

  if (ret_flag_host == -1) {
    return ret_flag_host;
  }

  // copy sum of weight (tmp_target_weight[0]) to total_weight
  CopyEqual<<<1, 1, 0, stream>>>(tmp_target_weight, total_weight, 1);
  return 0;
}

template <typename T, typename S>
__global__ void NLLLossGradKernel(const int n, const int c, const ReductionMode reduction, const T *input,
                                  const int32_t *target, const S *weight, const S *total_weight, const T *dloss,
                                  T *dinput) {
  int input_idx;
  int target_class;
  S tmp_quot;
  if (reduction == ReductionMode::kNone) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      target_class = static_cast<int>(target[i]);

      input_idx = (i * c) + target_class;

      MultiplyDevice(-weight[target_class], dloss[i], dinput + input_idx);
    }
  } else if (reduction == ReductionMode::kMean) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      target_class = static_cast<int>(target[i]);

      input_idx = (i * c) + target_class;

      tmp_quot = (-weight[target_class]) / *total_weight;
      MultiplyDevice(tmp_quot, dloss[0], dinput + input_idx);
    }
  } else if (reduction == ReductionMode::kSum) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      target_class = static_cast<int>(target[i]);

      input_idx = (i * c) + target_class;

      MultiplyDevice(-weight[target_class], dloss[0], dinput + input_idx);
    }
  }
}

template <typename T, typename S>
void NLLLossGrad(const int n, const int c, const ReductionMode reduction, const T *input, const int32_t *target,
                 const S *weight, const S *total_weight, const T *dloss, T *dinput, cudaStream_t stream) {
  int input_size = n * c;
  InitZero<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(dinput, input_size);

  NLLLossGradKernel<<<GET_BLOCKS(n), GET_THREADS, 0, stream>>>(n, c, reduction, input, target, weight, total_weight,
                                                               dloss, dinput);
}

template CUDA_LIB_EXPORT void KLDivLoss<float>(const int &input_size, const ReductionMode &reduction,
                                               const float *input_x, const float *input_y, float *loss, float *tmp_loss,
                                               cudaStream_t stream);

template CUDA_LIB_EXPORT void KLDivLossGrad<float>(const int &input_size, const ReductionMode &reduction,
                                                   const float *input_x, const float *input_y, const float *dloss,
                                                   float *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT void KLDivLoss<double>(const int &input_size, const ReductionMode &reduction,
                                                const double *input_x, const double *input_y, double *loss,
                                                double *tmp_loss, cudaStream_t stream);

template CUDA_LIB_EXPORT void KLDivLossGrad<double>(const int &input_size, const ReductionMode &reduction,
                                                    const double *input_x, const double *input_y, const double *dloss,
                                                    double *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT void BinaryCrossEntropyLoss<float>(const int &input_size, const ReductionMode &reduction,
                                                            const float *input_x, const float *input_y,
                                                            const float *weight, float *loss, float *tmp_loss,
                                                            cudaStream_t stream);

template CUDA_LIB_EXPORT void BinaryCrossEntropyLossGrad<float>(const int &input_size, const ReductionMode &reduction,
                                                                const float *input_x, const float *input_y,
                                                                const float *weight, const float *dloss, float *dx,
                                                                cudaStream_t stream);

template CUDA_LIB_EXPORT int NLLLoss<float, float>(const int n, const int c, const ReductionMode reduction,
                                                   const float *input, const int32_t *target, const float *weight,
                                                   float *loss, float *total_weight, float *tmp_loss,
                                                   float *tmp_target_weight, cudaStream_t stream);

template CUDA_LIB_EXPORT int NLLLoss<float, half>(const int n, const int c, const ReductionMode reduction,
                                                  const float *input, const int32_t *target, const half *weight,
                                                  float *loss, half *total_weight, float *tmp_loss,
                                                  half *tmp_target_weight, cudaStream_t stream);

template CUDA_LIB_EXPORT void NLLLossGrad<float, float>(const int n, const int c, const ReductionMode reduction,
                                                        const float *input, const int32_t *target, const float *weight,
                                                        const float *total_weight, const float *dloss, float *dinput,
                                                        cudaStream_t stream);

template CUDA_LIB_EXPORT void NLLLossGrad<float, half>(const int n, const int c, const ReductionMode reduction,
                                                       const float *input, const int32_t *target, const half *weight,
                                                       const half *total_weight, const float *dloss, float *dinput,
                                                       cudaStream_t stream);

template CUDA_LIB_EXPORT void KLDivLoss<half>(const int &input_size, const ReductionMode &reduction,
                                              const half *input_x, const half *input_y, half *loss, half *tmp_loss,
                                              cudaStream_t stream);

template CUDA_LIB_EXPORT void KLDivLossGrad<half>(const int &input_size, const ReductionMode &reduction,
                                                  const half *input_x, const half *input_y, const half *dloss, half *dx,
                                                  cudaStream_t stream);

template CUDA_LIB_EXPORT void BinaryCrossEntropyLoss<half>(const int &input_size, const ReductionMode &reduction,
                                                           const half *input_x, const half *input_y, const half *weight,
                                                           half *loss, half *tmp_loss, cudaStream_t stream);

template CUDA_LIB_EXPORT void BinaryCrossEntropyLossGrad<half>(const int &input_size, const ReductionMode &reduction,
                                                               const half *input_x, const half *input_y,
                                                               const half *weight, const half *dloss, half *dx,
                                                               cudaStream_t stream);

template CUDA_LIB_EXPORT int NLLLoss<half, half>(const int n, const int c, const ReductionMode reduction,
                                                 const half *input, const int32_t *target, const half *weight,
                                                 half *loss, half *total_weight, half *tmp_loss,
                                                 half *tmp_target_weight, cudaStream_t stream);

template CUDA_LIB_EXPORT int NLLLoss<half, float>(const int n, const int c, const ReductionMode reduction,
                                                  const half *input, const int32_t *target, const float *weight,
                                                  half *loss, float *total_weight, half *tmp_loss,
                                                  float *tmp_target_weight, cudaStream_t stream);

template CUDA_LIB_EXPORT void NLLLossGrad<half, half>(const int n, const int c, const ReductionMode reduction,
                                                      const half *input, const int32_t *target, const half *weight,
                                                      const half *total_weight, const half *dloss, half *dinput,
                                                      cudaStream_t stream);

template CUDA_LIB_EXPORT void NLLLossGrad<half, float>(const int n, const int c, const ReductionMode reduction,
                                                       const half *input, const int32_t *target, const float *weight,
                                                       const float *total_weight, const half *dloss, half *dinput,
                                                       cudaStream_t stream);
