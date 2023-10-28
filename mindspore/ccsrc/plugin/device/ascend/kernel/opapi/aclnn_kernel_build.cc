/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <string>
#include <utility>

#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_build.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
KernelModPtr AclnnOpBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(DEBUG) << "aclnn op [" << opname << "]";
  auto kernel_ptr = Factory<AclnnKernelMod>::Instance().Create(opname);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "aclnn can't find Kernel[" << opname << "]";
    return nullptr;
  }
  if (!kernel_ptr->Init(anf_node)) {
    MS_LOG(ERROR) << "Kernel initialize failed!";
    return nullptr;
  }

  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(anf_node);
  MS_EXCEPTION_IF_NULL(build_info);
  auto input_types = build_info->GetAllInputDeviceTypes();
  auto output_types = build_info->GetAllOutputDeviceTypes();
  ShapeArray input_shapes;
  ShapeArray output_shapes;
  for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(anf_node); ++i) {
    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(anf_node, i);
    input_shapes.push_back(std::move(shape));
  }
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(anf_node); ++i) {
    auto shape = common::AnfAlgo::GetOutputInferShape(anf_node, i);
    output_shapes.push_back(shape);
  }
  kernel_ptr->SetInputsInfo(input_types, input_shapes);
  kernel_ptr->SetOutputsInfo(output_types, output_shapes);
  return kernel_ptr;
}

bool IsRegisteredAclnnOp(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  return Factory<AclnnKernelMod>::Instance().IsRegistered(opname);
}
}  // namespace kernel
}  // namespace mindspore
