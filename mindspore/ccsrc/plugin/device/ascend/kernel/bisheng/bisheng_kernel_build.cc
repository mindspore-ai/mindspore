/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/bisheng/bisheng_kernel_build.h"

#include <memory>
#include "plugin/device/ascend/kernel/bisheng/bisheng_kernel_mod.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
KernelModPtr BiShengOpBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_mod_ptr = std::make_shared<BiShengKernelMod>(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  if (!kernel_mod_ptr->InitKernel(anf_node)) {
    MS_LOG(ERROR) << "BiSheng Kernel initialize failed!";
    return nullptr;
  }
  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
