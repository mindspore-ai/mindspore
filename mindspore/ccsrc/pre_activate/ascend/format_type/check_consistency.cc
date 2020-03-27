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
#include "pre_activate/ascend/format_type/check_consistency.h"

#include <string>
#include <memory>

#include "utils/utils.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckFormatForConsistency(const CNodePtr &node, const size_t input_index) {
  MS_EXCEPTION_IF_NULL(node);
  // get prior node's device output format
  string pre_output_format = AnfAlgo::GetPrevNodeOutputFormat(node, input_index);
  string selected_input_format = AnfAlgo::GetInputFormat(node, input_index);
  if (pre_output_format == selected_input_format) {
    return true;
  }
  auto input_origin_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, input_index);
  if (pre_output_format == kOpFormat_DEFAULT || selected_input_format == kOpFormat_DEFAULT) {
    string checking_format = (pre_output_format == kOpFormat_DEFAULT) ? selected_input_format : pre_output_format;
    // when input shape size is 1D, default format and NC1HWC0 are compatible
    if (input_origin_shape.size() == 1 && checking_format == kOpFormat_NC1HWC0) {
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
  if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsRealKernel(node)) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(cnode); i++) {
    if (!CheckFormatForConsistency(cnode, i) || !CheckDataTypeForConsistency(cnode, i)) {
      MS_LOG(EXCEPTION) << "Found inconsistent format or data type! Op: " << AnfAlgo::GetCNodeName(node) << "["
                        << node->DebugString() << "]";
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
