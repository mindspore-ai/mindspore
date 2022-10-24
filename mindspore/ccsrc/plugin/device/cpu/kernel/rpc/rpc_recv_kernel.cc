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

#include "plugin/device/cpu/kernel/rpc/rpc_recv_kernel.h"
#include <map>
#include <utility>
#include <algorithm>

namespace mindspore {
namespace kernel {
bool RpcRecvKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto abstract_base = base_operator->GetPrim()->ToAbstract();
  if (abstract_base) {
    if (abstract_base->isa<abstract::AbstractUMonad>() || abstract_base->isa<abstract::AbstractIOMonad>()) {
      recv_monad_ = true;
    }
  }
  // RpcRecv kernel is similar with Unique, the next op's infer op must be launched after RpcRecv kernel is done.
  is_need_retrieve_output_shape_ = true;
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int RpcRecvKernelMod::Resize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  // Reassign the memory size of recv kernel's inputs.
  for (size_t i = 0; i < inputs.size(); i++) {
    auto int64_shape = inputs[i]->GetShapeVector();
    if (IsDynamic(int64_shape)) {
      // Shape is invalid before recv data.
      MS_LOG(DEBUG) << "The recv kernel's input " << i << " shape inferred is still dynamic:" << int64_shape;
      is_dynamic_shape_ = true;
      return KRET_UNKNOWN_SHAPE;
    }

    int64_t size = 1;
    (void)GetShapeSize(int64_shape, TypeIdToType(inputs[i]->GetDtype()), &size);
    input_size_list_[i] = LongToSize(size);
  }
  // Reassign the memory size of recv kernel's outputs.
  for (size_t i = 0; i < outputs.size(); i++) {
    auto int64_shape = outputs[i]->GetShapeVector();
    if (IsDynamic(int64_shape)) {
      // Shape is invalid before recv data.
      MS_LOG(DEBUG) << "The recv kernel's output " << i << " shape inferred is still dynamic:" << int64_shape;
      is_dynamic_shape_ = true;
      return KRET_UNKNOWN_SHAPE;
    }

    int64_t size = 1;
    (void)GetShapeSize(int64_shape, TypeIdToType(outputs[i]->GetDtype()), &size);
    output_size_list_[i] = LongToSize(size);
  }
  return 0;
}

std::vector<KernelAttr> RpcRecvKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true).AddAllOutInRef(true)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RpcRecv, RpcRecvKernelMod);
}  // namespace kernel
}  // namespace mindspore
