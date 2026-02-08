/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_ALL_GATHER_MATMUL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_ALL_GATHER_MATMUL_H_

#include <memory>
#include <vector>
#include "kernel/ascend/hccl/hccl_kernel.h"

namespace mindspore {
namespace kernel {

constexpr uint32_t kAllGatherMatMulInputNum = 2;
constexpr uint32_t kAllGatherMatMulOutputNum = 2;
constexpr char kAttrNameTransposeA[] = "transpose_a";
constexpr char kAttrNameTransposeB[] = "transpose_b";

class HcomAllGatherMatMulKernel : public HcclKernel {
 public:
  HcomAllGatherMatMulKernel() = default;
  ~HcomAllGatherMatMulKernel() override = default;

  /* Inherit from kernelmod */
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  bool transpose_a_{false};
  bool transpose_b_{false};
#ifdef ENABLE_INTERNAL_KERNELS
  Lcal_ms::CoCDataTypeDesc lcoc_dtype_{Lcal_ms::CoCDataTypeDesc::FP16FP16_FP32_FP16};
  Lcal_ms::LcalType lcoc_type_{Lcal_ms::LcalType::ALL_GATHER_MATMUL_V2};
  Lcal_ms::QuantInfo quant_info_{};
  Lcal_ms::CoCTiling tiling_{};
  Lcal_ms::MatMulInfo matmul_info_{};
  Lcal_ms::CoCParamDesc param_desc_{};

  SetParamForLcocFunPtr set_param_for_lcoc_func_{nullptr};
  GetLcocWorkspaceSizeFunPtr get_lcoc_workspace_func_{nullptr};
  AllGatherMatmulFunPtr all_gather_matmul_func_{nullptr};
#endif
};

MS_HCCL_REG_KERNEL(AllGatherMatmul, HcomAllGatherMatMulKernel);
}  // namespace kernel
}  // namespace mindspore
#endif
