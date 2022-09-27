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
#include "plugin/device/ascend/kernel/acl/acl_kernel_build.h"

#include <vector>
#include <string>
#include "ir/dtype.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_utils.h"

namespace mindspore {
namespace kernel {
namespace {
bool SetIOInputSize(const std::shared_ptr<AnfNode> &anf_node, const size_t &input_num,
                    std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_size_list);
  for (size_t i = 0; i < input_num; i++) {
    auto index = AnfAlgo::GetInputIndexInGraph(anf_node, i);
    auto shape_i = AnfAlgo::GetInputDeviceShape(anf_node, index);
    if (AnfAlgo::GetInputDeviceDataType(anf_node, index) == kObjectTypeString) {
      if (!anf_node->isa<CNode>()) {
        MS_LOG(EXCEPTION) << "anf_node is not CNode.";
      }
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() < (index + 1)) {
        MS_LOG(ERROR) << "cnode inputs size " << cnode->inputs().size() << " is smaller than " << i + 1;
        return false;
      }
      auto input_node = cnode->inputs()[index + 1];
      MS_EXCEPTION_IF_NULL(input_node);
      if (input_node->isa<ValueNode>()) {
        auto value_ptr = GetValueNode(input_node);
        auto value = GetValue<std::string>(value_ptr);
        input_size_list->push_back(value.size());
      }
    } else {
      auto type_ptr = TypeIdToType(AnfAlgo::GetInputDeviceDataType(anf_node, index));
      int64_t size_i = 1;
      if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
        return false;
      }
      input_size_list->push_back(LongToSize(size_i));
    }
  }
  return true;
}

bool SetIOSize(const std::shared_ptr<AnfNode> &anf_node, const AclKernelModPtr &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(anf_node);

  if (!SetIOInputSize(anf_node, input_num, &input_size_list)) {
    return false;
  }
  kernel_mod_ptr->SetInputSizeList(input_size_list);
  if (output_num == 1 && HasAbstractMonad(anf_node)) {
    output_num = 0;
  }
  for (size_t i = 0; i < output_num; i++) {
    auto shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, i));
    int64_t size_i = 1;
    if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
      return false;
    }
    output_size_list.push_back(LongToSize(size_i));
  }
  kernel_mod_ptr->SetOutputSizeList(output_size_list);
  return true;
}

void SetGeInfo(const AnfNodePtr &node, const AclKernelModPtr &kernel_mode_ptr) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_mode_ptr);
  auto op_type = GeOpConvertor::GetOpType(node, true);
  kernel_mode_ptr->SetOpType(op_type);
  const auto &input_desc_list = AclUtils::GetInputTensorDesc(node);
  const auto &output_desc_list = AclUtils::GetOutputTensorDesc(node);
  kernel_mode_ptr->SetInputDescList(input_desc_list);
  kernel_mode_ptr->SetOutputDescList(output_desc_list);
  auto attr_list = GeOpConvertor::GetAttrAndValue(node, true);
  kernel_mode_ptr->SetAttrList(attr_list);
}
}  // namespace

KernelModPtr AclOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto kernel_mod_ptr = std::make_shared<AclKernelMod>(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);

  if (!SetIOSize(anf_node, kernel_mod_ptr)) {
    MS_LOG(EXCEPTION) << "SetIOSize failed for node:" << anf_node->DebugString();
  }

  SetGeInfo(anf_node, kernel_mod_ptr);
  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
