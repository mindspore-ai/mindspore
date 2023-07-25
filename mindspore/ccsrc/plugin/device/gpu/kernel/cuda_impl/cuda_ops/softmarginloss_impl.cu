/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "softmarginloss_impl.cuh"

inline __device__ double logT(double x) { return log(x); }
inline __device__ float logT(float x) { return logf(x); }
inline __device__ half logT(half x) { return hlog(x); }
inline __device__ double expT(double x) { return exp(x); }
inline __device__ float expT(float x) { return expf(x); }
inline __device__ half expT(half x) { return hexp(x); }
inline __device__ double castT(double ref, int x) { return __int2double_rn(x); }
inline __device__ float castT(float ref, int x) { return __int2float_rd(x); }
inline __device__ half castT(half ref, int x) { return __int2half_rd(x); }

template <typename T>
__global__ void Divide(T *numerator, int denominator) {
  numerator[0] = numerator[0] / castT(numerator[0], denominator);
}

template <typename T>
__global__ void AddTile(T *loss, int index) {
  loss[0] += loss[index];
}

template <typename T>
__global__ void Assign(T *loss, T *loss_work) {
  loss[0] = loss_work[0];
}

template <typename T>
__global__ void PartialSum(T *loss, int stride) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += blockDim.x * gridDim.x) {
    loss[i] += loss[i + stride];
  }
}

template <typename T>
__global__ void SoftMarginLoss(const T *prediction, const T *target, const size_t input_size,
                               const ReductionMode reduction, T *loss) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    loss[i] = logT(castT(loss[0], 1) + expT(-target[i] * prediction[i]));
  }
  return;
}

template <typename T>
cudaError_t Sum(T *loss, const size_t input_size, const uint32_t &device_id, cudaStream_t cuda_stream) {
  if (input_size % 2 == 1 && input_size != 1) {
    AddTile<<<1, 1, 0, cuda_stream>>>(loss, input_size - 1);
  }
  for (int stride = input_size / 2; stride > 0; stride >>= 1) {
    PartialSum<<<CUDA_BLOCKS(device_id, stride), CUDA_THREADS(device_id), 0, cuda_stream>>>(loss, stride);
    if (stride > 2 && stride % 2 == 1) {
      AddTile<<<1, 1, 0, cuda_stream>>>(loss, stride - 1);
    }
  }
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalSoftMarginLoss(const T *prediction, const T *target, const size_t input_size,
                              const ReductionMode &reduction, T *loss, T *loss_work, const uint32_t &device_id,
                              cudaStream_t cuda_stream) {
  if (reduction == ReductionMode::kNone) {
    SoftMarginLoss<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      prediction, target, input_size, reduction, loss);
    return GetCudaStatus();
  }

  SoftMarginLoss<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    prediction, target, input_size, reduction, loss_work);

  Sum(loss_work, input_size, device_id, cuda_stream);
  Assign<<<1, 1, 0, cuda_stream>>>(loss, loss_work);
  if (reduction == ReductionMode::kMean) {
    Divide<<<1, 1, 0, cuda_stream>>>(loss, static_cast<int>(input_size));
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSoftMarginLoss(const float *prediction, const float *target,
                                                       const size_t input_size, const ReductionMode &reduction,
                                                       float *loss, float *loss_work, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSoftMarginLoss(const half *prediction, const half *target,
                                                       const size_t input_size, const ReductionMode &reduction,
                                                       half *loss, half *loss_work, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSoftMarginLoss(const double *prediction, const double *target,
                                                       const size_t input_size, const ReductionMode &reduction,
                                                       double *loss, double *loss_work, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
