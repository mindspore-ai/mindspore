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

#include "mindspore/ccsrc/plugin/gpu/res_manager/device_context_conf/op_precision_conf.h"
#include "kernel/gpu/cublas/matmul/matmul_wrapper.h"

namespace mindspore {
namespace kernel {
#if CUDA_VERSION >= 11000
std::map<cudaDataType_t, cublasComputeType_t> data_compute_type_map_ = {
  {CUDA_R_8I, CUBLAS_COMPUTE_32I},  {CUDA_R_32I, CUBLAS_COMPUTE_32I}, {CUDA_R_16F, CUBLAS_COMPUTE_32F},
  {CUDA_R_32F, CUBLAS_COMPUTE_32F}, {CUDA_C_32F, CUBLAS_COMPUTE_32F}, {CUDA_R_64F, CUBLAS_COMPUTE_64F},
  {CUDA_C_64F, CUBLAS_COMPUTE_64F},
};
cublasComputeType_t GetComputeType(cudaDataType_t data_type) {
  if (data_type == CUDA_R_32F) {
    // tf32 only speed up in Ampere Architecture, such as 3090 or A100
    auto op_precision_conf = device::gpu::GPUOpPrecisionConf::GetInstance();
    auto matmul_allow_tf32 = op_precision_conf->matmul_allow_tf32();
    if (matmul_allow_tf32) {
      return CUBLAS_COMPUTE_32F_FAST_TF32;
    }
  }
  cublasComputeType_t compute_type = data_compute_type_map_[data_type];
  auto handle = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();
  auto math_mode = static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetMathMode(handle, math_mode), "cublasSetMathMode failed.");
  return compute_type;
}
#else
std::map<cudaDataType_t, cudaDataType_t> data_compute_type_map_ = {
  {CUDA_R_8I, CUDA_R_32I},  {CUDA_R_32I, CUDA_R_32I}, {CUDA_R_16F, CUDA_R_32F}, {CUDA_R_32F, CUDA_R_32F},
  {CUDA_C_32F, CUDA_C_32F}, {CUDA_R_64F, CUDA_R_64F}, {CUDA_C_64F, CUDA_C_64F},
};
cudaDataType_t GetComputeType(cudaDataType_t data_type) {
  cudaDataType_t compute_type = data_compute_type_map_[data_type];
  return compute_type;
}
#endif
}  // namespace kernel
}  // namespace mindspore
