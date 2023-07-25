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

#define EIGEN_USE_GPU
#include "betainc_impl.cuh"
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "unsupported/Eigen/CXX11/Tensor"

template <typename T>
cudaError_t CalBetainc(const size_t size, T *input_a, T *input_b, T *input_x, T *output, const uint32_t &device_id,
                       cudaStream_t cuda_stream) {
  int num = static_cast<int>(size);
  T *agpu = input_a, *bgpu = input_b, *xgpu = input_x;
  int gpudevice = device_id;
  Eigen::GpuStreamDevice stream(&cuda_stream, gpudevice);
  Eigen::GpuDevice gpu_device(&stream);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> Eigen_a(agpu, num);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> Eigen_b(bgpu, num);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> Eigen_x(xgpu, num);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> Eigen_z(output, num);
  Eigen_z.device(gpu_device) = Eigen::betainc(Eigen_a, Eigen_b, Eigen_x);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBetainc<float>(const size_t size, float *input_a, float *input_b,
                                                       float *input_x, float *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalBetainc<double>(const size_t size, double *input_a, double *input_b,
                                                        double *input_x, double *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
