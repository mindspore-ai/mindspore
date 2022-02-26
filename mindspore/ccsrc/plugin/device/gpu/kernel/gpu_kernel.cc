/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/gpu_kernel.h"

namespace mindspore {
namespace kernel {
void NativeGpuKernelMod::InferOp() {
  anf_node_ = kernel_node_.lock();
  if (common::AnfAlgo::IsDynamicShape(kernel_node_.lock())) {
    KernelMod::InferShape();
  }
}

void NativeGpuKernelMod::InitOp() {
  auto cnode = kernel_node_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  KernelMod::GetDepndLists(cnode);
  if (!common::AnfAlgo::GetBooleanAttr(cnode, kAttrInputIsDynamicShape) &&
      common::AnfAlgo::GetBooleanAttr(cnode, kAttrOutputIsDynamicShape) && depend_list_.empty()) {
    return;
  }

  MS_LOG(INFO) << "Update Args: " << cnode->fullname_with_scope();
  DestroyResource();
  ResetResource();
  Init(cnode);
}
}  // namespace kernel
}  // namespace mindspore
