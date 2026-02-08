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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_SIGNAL_OP_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_SIGNAL_OP_H_

#include <vector>
#include "kernel/ascend/symmetric_memory/symmetric_memory_kernel_mod.h"

namespace mindspore {
namespace kernel {
class SymmetricMemorySignalOp : public SymmetricMemoryKernelMod {
 public:
  SymmetricMemorySignalOp() : SymmetricMemoryKernelMod() {}
  ~SymmetricMemorySignalOp() = default;

 protected:
  symmetricmemory::SymmetricMemoryOpPtr CreateKernel(const symmetricmemory::InputsImmutableInfoList &inputs,
                                                     const symmetricmemory::OutputsImmutableInfoList &outputs,
                                                     const std::vector<KernelTensor *> &ms_inputs,
                                                     const std::vector<KernelTensor *> &ms_outputs) override;

 private:
  symmetricmemory::SignalOpParam param_;
};

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_SIGNAL_OP_H_
