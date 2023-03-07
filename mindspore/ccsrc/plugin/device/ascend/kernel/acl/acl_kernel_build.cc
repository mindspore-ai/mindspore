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
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_mod.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_utils.h"

namespace mindspore {
namespace kernel {
namespace {
static const std::unordered_set<std::string> kAclStaticList = {kTensorMoveOpName, kAddNOpName, kCheckValidOpName};

void GetStringTypeSize(const AnfNodePtr &node, size_t current_idx, size_t real_index,
                       std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(input_size_list);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "This node is not cnode, " << node->fullname_with_scope();
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = common::AnfAlgo::GetInputNode(cnode, current_idx);
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<ValueNode>()) {
    auto value_ptr = GetValueNode(input_node);
    auto value = GetValue<std::string>(value_ptr);
    (*input_size_list)[real_index] = value.size();
  }
}

bool SetIOInputSize(const std::shared_ptr<AnfNode> &anf_node, const size_t &input_num,
                    std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(input_size_list);
  for (size_t i = 0; i < input_num; i++) {
    auto index = AclUtils::GetInputKernelIdxByGraphIdx(anf_node, i);
    if (index < 0) {
      continue;
    }
    if (index >= SizeToInt(input_size_list->size())) {
      MS_LOG(EXCEPTION) << "Invalid index: " << index << ", input vector length: " << input_size_list->size()
                        << ", node: " << anf_node->fullname_with_scope();
    }
    if (AnfAlgo::GetInputDeviceDataType(anf_node, i) == kObjectTypeString) {
      GetStringTypeSize(anf_node, i, index, input_size_list);
    } else {
      auto type_ptr = TypeIdToType(AnfAlgo::GetInputDeviceDataType(anf_node, i));
      int64_t size_i = 1;
      const auto &device_shape = AnfAlgo::GetInputDeviceShape(anf_node, i);
      if (!GetShapeSize(device_shape, type_ptr, &size_i)) {
        MS_LOG(INFO) << "Empty shape or invalid type, shape: " << device_shape << ", type: " << type_ptr
                     << ", index: " << i << ", using default: SIZE_MAX. Node: " << anf_node->fullname_with_scope();
        continue;
      }
      (*input_size_list)[index] = LongToSize(size_i);
    }
  }
  return true;
}

bool SetIOSize(const std::shared_ptr<AnfNode> &anf_node, const AclKernelModPtr &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  const auto &input_names = AclUtils::GetOpInputAnchorNames(anf_node);
  const auto &output_names = AclUtils::GetOpOutputAnchorNames(anf_node);
  std::vector<size_t> input_size_list(input_names.size(), kSizeMax);
  std::vector<size_t> output_size_list(output_names.size(), kSizeMax);

  size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node);
  size_t output_num = AnfAlgo::GetOutputTensorNum(anf_node);

  // Set input size list
  if (!SetIOInputSize(anf_node, input_num, &input_size_list)) {
    return false;
  }
  kernel_mod_ptr->SetInputSizeList(input_size_list);

  // Set output size list
  if (output_num == 1 && HasAbstractMonad(anf_node)) {
    output_num = 0;
  }
  // process output
  for (size_t i = 0; i < output_num; i++) {
    auto idx = AclUtils::GetOutputKernelIdxByGraphIdx(anf_node, i);
    if (idx < 0) {
      continue;
    }
    if (idx >= SizeToInt(output_size_list.size())) {
      MS_LOG(EXCEPTION) << "Invalid index: " << idx << ", output vector length: " << output_size_list.size()
                        << ", node: " << anf_node->fullname_with_scope();
    }
    auto shape_i = AnfAlgo::GetOutputDeviceShape(anf_node, i);
    TypePtr type_ptr = TypeIdToType(AnfAlgo::GetOutputDeviceDataType(anf_node, i));
    int64_t size_i = 1;
    if (!GetShapeSize(shape_i, type_ptr, &size_i)) {
      MS_LOG(INFO) << "Empty output " << i << " with node " << anf_node->DebugString();
      continue;
    }
    output_size_list[idx] = LongToSize(size_i);
  }
  kernel_mod_ptr->SetOutputSizeList(output_size_list);
  return true;
}

void SetGeInfo(const AnfNodePtr &node, const AclKernelModPtr &kernel_mod_ptr) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  auto op_type = GeOpConvertor::GetOpType(node, true);
  kernel_mod_ptr->SetOpType(op_type);
  const auto &input_desc_list = AclUtils::GetInputTensorDesc(node);
  const auto &output_desc_list = AclUtils::GetOutputTensorDesc(node);
  kernel_mod_ptr->SetInputDescList(input_desc_list);
  kernel_mod_ptr->SetOutputDescList(output_desc_list);
  if (kAclStaticList.count(op_type) == 0) {
    kernel_mod_ptr->SetDynamic(true);
  }
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
