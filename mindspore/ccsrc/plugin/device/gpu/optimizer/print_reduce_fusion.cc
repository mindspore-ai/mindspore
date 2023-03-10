/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/print_reduce_fusion.h"

#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
using mindspore::tensor::Tensor;
kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(CNodePtr node) {
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; input_index++) {
    inputs_format.push_back(kOpFormat_DEFAULT);
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t output_index = 0; output_index < output_num; output_index++) {
    outputs_format.push_back(kOpFormat_DEFAULT);
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(node, output_index));
  }

  builder.SetInputsFormat(inputs_format);
  builder.SetOutputsFormat(outputs_format);
  builder.SetInputsDeviceType(inputs_type);
  builder.SetOutputsDeviceType(outputs_type);
  return builder.Build();
}

bool GetOptList(const std::vector<AnfNodePtr> &node_list, std::vector<AnfNodePtr> *opt_list,
                std::vector<std::vector<int64_t>> *string_pos_vec,
                std::vector<std::vector<std::string>> *string_value_vec,
                std::vector<std::vector<std::pair<int64_t, int64_t>>> *not_tensor_pos_vec) {
  MS_EXCEPTION_IF_NULL(opt_list);
  MS_EXCEPTION_IF_NULL(string_pos_vec);
  MS_EXCEPTION_IF_NULL(string_value_vec);

  for (auto &node : node_list) {
    // {prim::kPrimPrint} reduction only applies on print with string, tensor(scalar or tuple)
    MS_EXCEPTION_IF_NULL(node);
    std::vector<int64_t> string_pos;
    std::vector<std::string> string_value;
    std::vector<std::pair<int64_t, int64_t>> value_type;
    if (IsPrimitiveCNode(node, prim::kPrimPrint)) {
      auto prim = common::AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(prim);
      std::vector<int64_t> fake_tensor_pos;
      if (prim->HasAttr(kFakeTensorPos)) {
        auto value_ptr = prim->GetAttr(kFakeTensorPos);
        fake_tensor_pos = GetValue<std::vector<int64_t>>(value_ptr);
      }
      size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
      for (size_t i = 0; i < input_num; i++) {
        auto current_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), i);
        // not tensor(tuple, scalar, string)
        if (current_node->cast<ValueNodePtr>() == nullptr) {
          continue;
        }
        auto value_node = current_node->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        auto shape = value_node->abstract();
        MS_EXCEPTION_IF_NULL(shape);
        auto shape_node = dyn_cast<abstract::Shape>(shape->GetShapeTrack());
        bool is_fake_tensor = (std::find(fake_tensor_pos.begin(), fake_tensor_pos.end(), i) != fake_tensor_pos.end());
        if ((!IsValueNode<Tensor>(value_node) || is_fake_tensor) && shape_node != nullptr) {
          // a scalar or tuple
          auto shape_size = shape_node->shape().size();
          if (shape_size != 0) {
            value_type.push_back(std::make_pair(i, 1));
          } else {
            value_type.push_back(std::make_pair(i, 0));
          }
        }
        auto node_value = value_node->value();
        if (node_value->type() == nullptr) {
          // not a string
          continue;
        }
        auto type = node_value->type();
        MS_EXCEPTION_IF_NULL(type);
        if (type->generic_type_id() == kObjectTypeString) {
          auto current_string_value = GetValue<std::string>(node_value);
          string_pos.push_back(i);
          string_value.push_back(std::string(current_string_value));
        } else {
          MS_LOG(EXCEPTION) << "Current value node is not string or tensor";
        }
      }
      if (string_pos.size() != 0 || value_type.size() != 0) {
        opt_list->push_back(node);
        string_pos_vec->push_back(string_pos);
        string_value_vec->push_back(string_value);
        not_tensor_pos_vec->push_back(value_type);
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
  // first is pos, second is type: 0 is Scalar, 1 is ValueTuple
  std::vector<std::vector<std::pair<int64_t, int64_t>>> not_tensor_pos_vec;
  if (!GetOptList(node_list, &opt_list, &string_pos_vec, &string_value_vec, &not_tensor_pos_vec)) {
    return false;
  }
  for (size_t idx = 0; idx < opt_list.size(); idx++) {
    auto node = opt_list[idx];
    CNodePtr cnode = utils::cast<CNodePtr>(node);
    MS_EXCEPTION_IF_NULL(cnode);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
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
      auto input_tensor = common::AnfAlgo::GetInputNode(cnode, i);
      MS_EXCEPTION_IF_NULL(input_tensor);
      inputs.push_back(input_tensor);
    }
    // add monad
    auto monad_node = common::AnfAlgo::GetInputNode(cnode, input_flag.size());
    MS_EXCEPTION_IF_NULL(monad_node);
    inputs.push_back(monad_node);
    auto string_value = string_value_vec[idx];
    auto value_type_vec = not_tensor_pos_vec[idx];
    // split value type and pos
    std::vector<int64_t> value_type_pos;
    std::vector<int64_t> value_type;
    (void)std::transform(value_type_vec.begin(), value_type_vec.end(), std::back_inserter(value_type_pos),
                         [](const std::pair<int64_t, int64_t> &value) { return value.first; });
    (void)std::transform(value_type_vec.begin(), value_type_vec.end(), std::back_inserter(value_type),
                         [](const std::pair<int64_t, int64_t> &value) { return value.second; });
    // create new cnode
    auto print_fused = graph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(print_fused);
    // hand over the attrs to new print
    common::AnfAlgo::SetNodeAttr("string_pos", MakeValue<std::vector<int64_t>>(string_pos), print_fused);
    common::AnfAlgo::SetNodeAttr("string_value", MakeValue<std::vector<std::string>>(string_value), print_fused);
    common::AnfAlgo::SetNodeAttr("value_type", MakeValue<std::vector<int64_t>>(value_type), print_fused);
    common::AnfAlgo::SetNodeAttr("value_type_pos", MakeValue<std::vector<int64_t>>(value_type_pos), print_fused);
    auto old_prim = GetCNodePrimitive(cnode);
    if (old_prim->HasAttr(kFakeTensorListPos)) {
      auto value_ptr = old_prim->GetAttr(kFakeTensorListPos);
      auto fake_tensor_list_pos = GetValue<std::vector<int64_t>>(value_ptr);
      common::AnfAlgo::SetNodeAttr(kFakeTensorListPos, MakeValue<std::vector<int64_t>>(fake_tensor_list_pos),
                                   print_fused);
    }
    // set output type and shape
    std::vector<TypeId> types;
    std::vector<BaseShapePtr> shapes;
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t i = 0; i < output_num; i++) {
      types.push_back(common::AnfAlgo::GetOutputInferDataType(cnode, i));
      shapes.push_back(AnfAlgo::GetOutputDetailShape(cnode, i));
    }
    common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, print_fused.get());
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
