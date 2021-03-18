/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/gpu/print_reduce_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(CNodePtr node) {
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;

  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; input_index++) {
    inputs_format.push_back(kOpFormat_DEFAULT);
    inputs_type.push_back(AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t output_index = 0; output_index < output_num; output_index++) {
    outputs_format.push_back(kOpFormat_DEFAULT);
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(node, output_index));
  }

  builder.SetInputsFormat(inputs_format);
  builder.SetOutputsFormat(outputs_format);
  builder.SetInputsDeviceType(inputs_type);
  builder.SetOutputsDeviceType(outputs_type);
  return builder.Build();
}

bool GetOptList(const std::vector<AnfNodePtr> &node_list, std::vector<AnfNodePtr> *opt_list,
                std::vector<std::vector<int64_t>> *string_pos_vec,
                std::vector<std::vector<std::string>> *string_value_vec) {
  for (auto &node : node_list) {
    // {prim::kPrimPrint} only print with string will be reduced
    std::vector<int64_t> string_pos;
    std::vector<std::string> string_value;
    if (IsPrimitiveCNode(node, prim::kPrimPrint)) {
      size_t input_num = AnfAlgo::GetInputTensorNum(node);
      for (size_t i = 0; i < input_num; i++) {
        auto current_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), i);
        // not a string
        if (current_node->cast<ValueNodePtr>() == nullptr) {
          continue;
        }
        auto value_node = current_node->cast<ValueNodePtr>()->value();
        // not a string
        if (value_node->type() == nullptr) {
          continue;
        }
        if (value_node->type()->generic_type_id() == kObjectTypeString) {
          auto current_string_value = GetValue<std::string>(value_node);
          string_pos.push_back(i);
          string_value.push_back(std::string(current_string_value));
        } else {
          MS_LOG(EXCEPTION) << "Current value node is not string or tensor";
        }
      }
      if (string_pos.size() != 0) {
        opt_list->push_back(node);
        string_pos_vec->push_back(string_pos);
        string_value_vec->push_back(string_value);
      }
    }
  }
  if (opt_list->size() == 0) {
    return false;
  }
  return true;
}

bool PrintReduceFusion::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<AnfNodePtr> opt_list;
  std::vector<std::vector<int64_t>> string_pos_vec;
  std::vector<std::vector<std::string>> string_value_vec;
  if (!GetOptList(node_list, &opt_list, &string_pos_vec, &string_value_vec)) {
    return false;
  }
  for (size_t idx = 0; idx < opt_list.size(); idx++) {
    auto node = opt_list[idx];
    CNodePtr cnode = utils::cast<CNodePtr>(node);
    size_t input_num = AnfAlgo::GetInputTensorNum(cnode);
    auto prim = std::make_shared<Primitive>("Print");
    std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
    auto string_pos = string_pos_vec[idx];
    std::vector<int64_t> input_flag(input_num);
    for (size_t i = 0; i < string_pos.size(); i++) {
      if (string_pos[i] < 0) {
        MS_LOG(EXCEPTION) << "string_pos cannot be a negative value";
      }
      size_t index = LongToSize(string_pos[i]);
      input_flag[index] = -1;
    }
    for (size_t i = 0; i < input_flag.size(); i++) {
      if (input_flag[i] == -1) {
        continue;
      }
      auto input_tensor = AnfAlgo::GetInputNode(cnode, i);
      MS_EXCEPTION_IF_NULL(input_tensor);
      inputs.push_back(input_tensor);
    }
    // add monad
    auto monad_node = AnfAlgo::GetInputNode(cnode, input_flag.size());
    MS_EXCEPTION_IF_NULL(monad_node);
    inputs.push_back(monad_node);
    auto string_value = string_value_vec[idx];
    // create new cnode
    auto print_fused = graph->NewCNode(inputs);
    // hand over the attrs to new print
    AnfAlgo::SetNodeAttr("string_pos", MakeValue<std::vector<int64_t>>(string_pos), print_fused);
    AnfAlgo::SetNodeAttr("string_value", MakeValue<std::vector<std::string>>(string_value), print_fused);
    // set output type and shape
    std::vector<TypeId> types;
    std::vector<std::vector<size_t>> shapes;
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t i = 0; i < output_num; i++) {
      types.push_back(AnfAlgo::GetOutputInferDataType(cnode, i));
      shapes.push_back(AnfAlgo::GetOutputInferShape(cnode, i));
    }
    AnfAlgo::SetOutputInferTypeAndShape(types, shapes, print_fused.get());
    // add build info
    auto build_info = GenerateKernelBuildInfo(print_fused);
    AnfAlgo::SetSelectKernelBuildInfo(build_info, print_fused.get());
    if (!manager->Replace(cnode, print_fused)) {
      MS_LOG(EXCEPTION) << "manager replace node failed in print reduce fusion.";
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
