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

#include "backend/kernel_compiler/gpu/gpu_kernel.h"

namespace mindspore {
namespace kernel {
void GpuDynamicKernel::UpdateArgs() {
  if (!is_input_dynamic_shape_ && is_output_dynamic_shape_ && !have_depends()) {
    return;
  }

  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Update Args: " << cnode->fullname_with_scope();
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto gpu_kernel_mod = dynamic_cast<GpuKernel *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(gpu_kernel_mod);
  gpu_kernel_mod->DestroyResource();
  gpu_kernel_mod->ResetResource();
  gpu_kernel_mod->Init(cnode);
}
}  // namespace kernel
}  // namespace mindspore
