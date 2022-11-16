/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/control_flow/kernel/entrance_subgraph_kernel.h"
#include "src/tensor.h"

namespace mindspore::kernel {
int EntranceSubGraphKernel::Execute(const KernelCallBack &, const KernelCallBack &) { return lite::RET_OK; }

SubGraphKernel *EntranceSubGraphKernel::Create(Kernel *kernel) {
  auto sub_kernel = new kernel::EntranceSubGraphKernel(kernel);
  if (sub_kernel == nullptr) {
    MS_LOG(ERROR) << "create entrance subgraph failed!";
  }
  return sub_kernel;
}
}  // namespace mindspore::kernel
