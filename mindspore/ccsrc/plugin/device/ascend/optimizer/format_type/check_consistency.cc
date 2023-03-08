/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/format_type/check_consistency.h"

#include <string>
#include <memory>
#include <vector>

#include "include/common/utils/utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace opt {
namespace {
const std::set<std::string> kDefaultCompatibleFormat = {kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
                                                        kOpFormat_NCDHW};

bool CheckFormatForConsistency(const CNodePtr &node, const size_t input_index) {
  MS_EXCEPTION_IF_NULL(node);
  // get prior node's device output format
  auto prev_node = common::AnfAlgo::GetPrevNodeOutput(node, input_index);
  string pre_output_format = AnfAlgo::GetOutputFormat(prev_node.first, prev_node.second);
  string selected_input_format = AnfAlgo::GetInputFormat(node, input_index);
  if (pre_output_format == selected_input_format) {
    if (selected_input_format == kOpFormat_FRAC_Z &&
        (prev_node.first->isa<CNode>() || prev_node.first->isa<Parameter>())) {
      auto pre_groups = common::AnfAlgo::GetAttrGroups(prev_node.first, prev_node.second);
      auto cur_groups = common::AnfAlgo::GetAttrGroups(node, input_index);
      if (pre_groups != cur_groups) {
        MS_LOG(ERROR) << "Found inconsistent format! input format " << input_index << ": " << pre_output_format
                      << " with groups " << pre_groups << ", selected input format: " << selected_input_format
                      << " with groups " << cur_groups;
        return false;
      }
    }
    return true;
  }
  auto input_origin_shape = common::AnfAlgo::GetOutputInferShape(prev_node.first, prev_node.second);
  if (pre_output_format == kOpFormat_DEFAULT || selected_input_format == kOpFormat_DEFAULT) {
    string checking_format = (pre_output_format == kOpFormat_DEFAULT) ? selected_input_format : pre_output_format;
    // when input shape size is 1D, default format and NC1HWC0 are compatible
    if (input_origin_shape.size() == 1 && LongToSize(input_origin_shape[0]) % kCubeSize == 0) {
      return true;
    }
    if (kDefaultCompatibleFormat.find(checking_format) != kDefaultCompatibleFormat.end()) {
      return true;
    }
  }
  if (input_origin_shape.size() == 0) {
    return true;
  }
  MS_LOG(ERROR) << "Found inconsistent format! input format " << input_index << ": " << pre_output_format
                << ", selected input format: " << selected_input_format;
  return false;
}

bool CheckDataTypeForConsistency(const CNodePtr &node, const size_t input_index) {
  MS_EXCEPTION_IF_NULL(node);
  TypeId input_data_type = AnfAlgo::GetPrevNodeOutputDeviceDataType(node, input_index);
  TypeId selected_data_type = AnfAlgo::GetInputDeviceDataType(node, input_index);
  if (input_data_type == selected_data_type) {
    return true;
  }
  MS_LOG(ERROR) << "Found inconsistent dtype! input dtype " << input_index << ": " << TypeIdLabel(input_data_type)
                << ", selected dtype: " << TypeIdLabel(selected_data_type);
  return false;
}
}  // namespace

const BaseRef CheckConsistency::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({X, Xs});
}

const AnfNodePtr CheckConsistency::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }

  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t in_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 0; i < in_num; ++i) {
    if (!CheckFormatForConsistency(cnode, i) || !CheckDataTypeForConsistency(cnode, i)) {
      MS_LOG(EXCEPTION) << "Found inconsistent format or data type! Op: " << common::AnfAlgo::GetCNodeName(cnode) << "["
                        << cnode->DebugString() << "], fullname: " << node->fullname_with_scope();
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
