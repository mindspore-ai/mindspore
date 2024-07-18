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

#include "plugin/device/ascend/kernel/internal/acme/logical_not.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeLogicalNot::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                             const acme::OutputsImmutableInfoList &outputs_ii,
                                             const std::vector<KernelTensor *> &ms_inputs,
                                             const std::vector<KernelTensor *> &ms_outputs) {
  if (inputs_ii[kIndex0].GetDtype() == acme::kTypeBool) {
    // for now, logical_not is a asd op and it only support int8 input
    // int8 and bool make the same result
    auto inputs_ii_new = inputs_ii;
    auto outputs_ii_new = outputs_ii;
    inputs_ii_new[kIndex0].SetDtype(acme::kTypeInt8);
    outputs_ii_new[kIndex0].SetDtype(acme::kTypeInt8);
    return acme::CreateLogicalNotOp(inputs_ii_new, outputs_ii_new, acme::kAcmeLogicalNotOpName);
  }
  return acme::CreateLogicalNotOp(inputs_ii, outputs_ii, acme::kAcmeLogicalNotOpName);
}

MS_ACME_KERNEL_FACTORY_REG(LogicalNot, acme::kAcmeLogicalNotOpName, AcmeLogicalNot);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(LogicalNot, INPUT_NUM_1, INDEX_0);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(LogicalNot, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
