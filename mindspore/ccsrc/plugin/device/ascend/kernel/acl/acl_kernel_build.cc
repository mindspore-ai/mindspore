/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/acl/acl_kernel_build.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {
KernelModPtr AclOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_mod_ptr = std::make_shared<AclKernelMod>(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);

  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(anf_node);
  MS_EXCEPTION_IF_NULL(build_info);
  auto input_formats = build_info->GetAllInputFormats();
  auto input_types = build_info->GetAllInputDeviceTypes();
  auto output_formats = build_info->GetAllOutputFormats();
  auto output_types = build_info->GetAllOutputDeviceTypes();
  kernel_mod_ptr->SetDeviceInfo(input_formats, output_formats, input_types, output_types);

  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode)) {
    return kernel_mod_ptr;
  }

  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  kernel_mod_ptr->SetPrimitive(primitive);
  std::string format = transform::AclHelper::GetFormatFromAttr(primitive);
  for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(cnode); ++i) {
    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, i);
    kernel_mod_ptr->PackageInput(i, format, &shape);
  }
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(cnode); ++i) {
    const auto &shape = common::AnfAlgo::GetOutputInferShape(cnode, i);
    kernel_mod_ptr->PackageOutput(i, shape);
  }
  kernel_mod_ptr->SetNeedConvertHostTensor(true);
  kernel_mod_ptr->CreateAclConverter();
  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
