/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/acme/matmul_elemwise.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {

constexpr auto matmul_elemwise_fusion_relu_str = "relu";
constexpr auto matmul_elemwise_fusion_gelu_str = "gelu";
constexpr auto matmul_elemwise_fusion_biasadd_str = "bias_add";
constexpr auto matmul_elemwise_fusion_biasadd_fastgelu_str = "bias_add_fastgelu";

acme::AcmeOpPtr AcmeFusedMatMulElemUnary::CreateKernel(const acme::InputsImmutableInfoList &inputs,
                                                       const acme::OutputsImmutableInfoList &outputs,
                                                       const std::vector<KernelTensor *> &ms_inputs,
                                                       const std::vector<KernelTensor *> &ms_outputs) {
  acme::MatmulParam param;
  param.transpose_a = primitive_->HasAttr("is_trans_a") ? GetValue<bool>(primitive_->GetAttr("is_trans_a")) : false;
  param.transpose_b = primitive_->HasAttr("is_trans_b") ? GetValue<bool>(primitive_->GetAttr("is_trans_b")) : false;
  auto value_str = primitive_->GetAttr("ElemwiseType");
  MS_EXCEPTION_IF_NULL(value_str);
  std::string elemwise_type = GetValue<std::string>(value_str);
  if (elemwise_type == matmul_elemwise_fusion_relu_str) {
    param.with_relu = true;
  } else if (elemwise_type == matmul_elemwise_fusion_gelu_str) {
    param.with_gelu = true;
  }
  param.enable_shuffle = false;  // the real definition is in acme
  param.enable_dequant = false;
  return acme::CreateMatmulOp(inputs, outputs, param, acme::kAcmeMatMulOpName);
}

acme::AcmeOpPtr AcmeFusedMatMulElemBinary::CreateKernel(const acme::InputsImmutableInfoList &inputs,
                                                        const acme::OutputsImmutableInfoList &outputs,
                                                        const std::vector<KernelTensor *> &ms_inputs,
                                                        const std::vector<KernelTensor *> &ms_outputs) {
  acme::MatmulParam param;
  param.transpose_a = primitive_->HasAttr("is_trans_a") ? GetValue<bool>(primitive_->GetAttr("is_trans_a")) : false;
  param.transpose_b = primitive_->HasAttr("is_trans_b") ? GetValue<bool>(primitive_->GetAttr("is_trans_b")) : false;
  auto value_str = primitive_->GetAttr("ElemwiseType");
  MS_EXCEPTION_IF_NULL(value_str);
  std::string elemwise_type = GetValue<std::string>(value_str);
  if (elemwise_type == matmul_elemwise_fusion_biasadd_str) {
    param.with_bias = true;
  } else if (elemwise_type == matmul_elemwise_fusion_biasadd_fastgelu_str) {
    param.with_bias_fastgelu = true;
  }
  param.enable_shuffle = false;  // the real definition is in acme
  param.enable_dequant = false;
  return acme::CreateMatmulOp(inputs, outputs, param, acme::kAcmeMatMulOpName);
}

MS_ACME_KERNEL_FACTORY_REG(FusedMatMulElemBinary, acme::kAcmeMatMulOpName, AcmeFusedMatMulElemBinary);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedMatMulElemBinary, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FusedMatMulElemBinary, OUTPUT_NUM_1, INDEX_0);

MS_ACME_KERNEL_FACTORY_REG(FusedMatMulElemUnary, acme::kAcmeMatMulOpName, AcmeFusedMatMulElemUnary);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedMatMulElemUnary, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FusedMatMulElemUnary, OUTPUT_NUM_1, INDEX_0);

}  // namespace kernel
}  // namespace mindspore
