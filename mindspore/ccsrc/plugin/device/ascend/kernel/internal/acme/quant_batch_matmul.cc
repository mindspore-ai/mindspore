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

#include "plugin/device/ascend/kernel/internal/acme/quant_batch_matmul.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeQuantBatchMatmul::CreateKernel(acme::InputsImmutableInfoList inputs, acme::OutputsImmutableInfoList outputs,
                                         const std::vector<KernelTensor *> &ms_inputs,
                                         const std::vector<KernelTensor *> &ms_outputs) {
  acme::MatmulParam param;
  param.transpose_a = ms_inputs[kIndex5]->GetValueWithCheck<bool>();
  param.transpose_b = ms_inputs[kIndex6]->GetValueWithCheck<bool>();
  param.with_bias = !(ms_inputs[kIndex4]->GetType()->isa<TypeNone>());
  param.enable_shuffle = false; // the real definition is in acme
  param.enable_dequant = true;
  const std::string op_name = "QuantBatchMatmul";
  return acme::CreateMatmulOp(inputs, outputs, param, op_name);
}
MS_ACME_KERNEL_FACTORY_REG(QuantBatchMatmul, AcmeQuantBatchMatmul);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantBatchMatmul, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_4, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantBatchMatmul, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
