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

#include "kernel/ascend/symmetric_memory/signal_op.h"
#include <vector>
#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
symmetricmemory::SymmetricMemoryOpPtr SymmetricMemorySignalOp::CreateKernel(
  const symmetricmemory::InputsImmutableInfoList &inputs_ii,
  const symmetricmemory::OutputsImmutableInfoList &outputs_ii, const std::vector<KernelTensor *> &ms_inputs,
  const std::vector<KernelTensor *> &ms_outputs) {
  param_.signal_op = ms_inputs[INDEX_3]->GetValueWithCheck<int64_t>();
  param_.target_pe = ms_inputs[INDEX_4]->GetValueWithCheck<int64_t>();
  return symmetricmemory::CreateSignalOpOp(inputs_ii, outputs_ii, param_,
                                           symmetricmemory::kSymmetricMemorySignalOpOpName);
}

MS_SYMMETRIC_MEMORY_KERNEL_FACTORY_REG(SignalOp, symmetricmemory::kSymmetricMemorySignalOpOpName,
                                       SymmetricMemorySignalOp);
REG_MS_TO_SYMMETRIC_MEMORY_IN_TENSOR_IDX_MAP(SignalOp, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
}  // namespace kernel
}  // namespace mindspore
