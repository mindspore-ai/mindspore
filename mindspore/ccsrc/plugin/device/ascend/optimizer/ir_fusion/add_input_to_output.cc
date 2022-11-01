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
#include "plugin/device/ascend/optimizer/ir_fusion/add_input_to_output.h"
#include <vector>
#include <algorithm>
#include "plugin/device/ascend/optimizer/ir_fusion/input_to_output_registry.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/oplib/oplib.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
void GetInputOrOutputNames(const CNodePtr &cnode, const std::string &attr_name, std::vector<std::string> *names_vec) {
  MS_EXCEPTION_IF_NULL(names_vec);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr names_value = primitive->GetAttr(attr_name);
  if (names_value == nullptr) {
    return;
  }
  *names_vec = GetValue<std::vector<std::string>>(names_value);
}

void AddOutputs(const CNodePtr &cnode, const std::vector<size_t> &input_indices) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<std::string> input_names_vec;
  GetInputOrOutputNames(cnode, kAttrInputNames, &input_names_vec);
  std::vector<std::string> output_names_vec;
  GetInputOrOutputNames(cnode, kAttrOutputNames, &output_names_vec);
  AbstractBasePtrList abstract_list;
  auto origin_abstract = cnode->abstract();
  MS_EXCEPTION_IF_NULL(origin_abstract);
  if (origin_abstract->isa<abstract::AbstractTuple>()) {
    auto origin_abstract_tuple = dyn_cast<abstract::AbstractTuple>(origin_abstract);
    MS_EXCEPTION_IF_NULL(origin_abstract_tuple);
    AbstractBasePtrList origin_abstract_list = origin_abstract_tuple->elements();
    (void)std::copy(origin_abstract_list.begin(), origin_abstract_list.end(), std::back_inserter(abstract_list));
  } else {
    (void)abstract_list.emplace_back(origin_abstract);
  }

  for (size_t i = 0; i < input_indices.size(); ++i) {
    size_t index = input_indices[i];
    if (index + 1 >= cnode->inputs().size()) {
      MS_LOG(INFO) << "The input index " << index << " for converting to output is out of range, "
                   << "node: " << cnode->DebugString();
      continue;
    }
    auto node_to_output = cnode->input(index + 1);
    MS_EXCEPTION_IF_NULL(node_to_output);
    (void)abstract_list.emplace_back(node_to_output->abstract());
    if (!input_names_vec.empty() && !output_names_vec.empty() && index < input_names_vec.size()) {
      (void)output_names_vec.emplace_back(input_names_vec[index]);
    }
  }
  if (!output_names_vec.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names_vec), cnode);
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  cnode->set_abstract(abstract_tuple);
}
}  // namespace

const AnfNodePtr AddInputToOutput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::string op_name = common::AnfAlgo::GetCNodeName(cnode);
  InputToOutputRegister reg;
  if (!InputToOutputRegistry::Instance().GetRegisterByOpName(op_name, &reg)) {
    return nullptr;
  }
  int output_num = op_finder_->GetOpRegisteredOutputNum(op_name, cnode);
  // No need add output when it is not a tbe op.
  if (output_num == -1) {
    return nullptr;
  }
  // No need add output if the output num matches the registered output num for tbe.
  if (common::AnfAlgo::GetOutputTensorNum(cnode) >= IntToSize(output_num)) {
    return nullptr;
  }
  bool is_origin_tuple_output = common::AnfAlgo::IsTupleOutput(cnode);
  AddOutputs(cnode, reg.input_indices());
  // No need to create tuple_getitem if the origin output is a tuple because there has already been some tuple_getitems
  // pointed to the outputs.
  if (is_origin_tuple_output) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_outputs;
  auto new_abstract_tuple = dyn_cast<abstract::AbstractTuple>(cnode->abstract());
  MS_EXCEPTION_IF_NULL(new_abstract_tuple);
  CreateMultipleOutputsOfAnfNode(func_graph, cnode, new_abstract_tuple->size(), &new_outputs);
  if (new_outputs.size() != new_abstract_tuple->size()) {
    MS_LOG(EXCEPTION) << "Failed to create outputs of " << cnode->DebugString() << trace::DumpSourceLines(node);
  }
  return new_outputs[0];
}
}  // namespace opt
}  // namespace mindspore
