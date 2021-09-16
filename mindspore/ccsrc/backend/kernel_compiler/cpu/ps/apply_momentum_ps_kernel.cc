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

#include "backend/kernel_compiler/cpu/ps/apply_momentum_ps_kernel.h"

namespace mindspore {
namespace kernel {
namespace ps {
bool ApplyMomentumPSKernel::Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  return Launch(inputs, workspace, outputs);
}

const std::vector<size_t> &ApplyMomentumPSKernel::input_sizes() const { return GetInputSizeList(); }

const std::vector<size_t> &ApplyMomentumPSKernel::output_sizes() const { return GetOutputSizeList(); }

const std::vector<size_t> &ApplyMomentumPSKernel::workspace_sizes() const { return GetWorkspaceSizeList(); }
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore
