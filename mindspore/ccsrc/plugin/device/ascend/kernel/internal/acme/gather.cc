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

#include "plugin/device/ascend/kernel/internal/acme/gather.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeGather::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                         const acme::OutputsImmutableInfoList &outputs_ii,
                                         const std::vector<KernelTensor *> &ms_inputs,
                                         const std::vector<KernelTensor *> &ms_outputs) {
  acme::GatherParam param;
  param.axes.emplace_back(ms_inputs[kIndex2]->GetValueWithCheck<int64_t>());
  param.batch_dims = ms_inputs[kIndex3]->GetValueWithCheck<int64_t>();
  return acme::CreateGatherOp(inputs_ii, outputs_ii, param, acme::kAcmeGatherOpName);
}

MS_ACME_KERNEL_FACTORY_REG(Gather, acme::kAcmeGatherOpName, AcmeGather);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(Gather, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(Gather, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
