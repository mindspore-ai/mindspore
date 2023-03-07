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
#include "plugin/device/cpu/optimizer/print_value_type.h"

#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
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
                std::vector<std::vector<std::pair<int64_t, int64_t>>> *not_tensor_pos_vec) {
  MS_EXCEPTION_IF_NULL(opt_list);

  for (auto &node : node_list) {
    // {prim::kPrimPrint} reduction only applies on print with string, tensor(scalar or tuple)
    MS_EXCEPTION_IF_NULL(node);
    std::vector<std::pair<int64_t, int64_t>> value_type;
    if (!IsPrimitiveCNode(node, prim::kPrimPrint)) {
      continue;
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
      if (shape_node != nullptr) {
        // a scalar or tuple
        auto shape_size = shape_node->shape().size();
        if (shape_size != 0) {
          value_type.push_back(std::make_pair(i, 1));
        } else {
          value_type.push_back(std::make_pair(i, 0));
        }
      }
    }
    if (value_type.size() != 0) {
      opt_list->push_back(node);
      not_tensor_pos_vec->push_back(value_type);
    }
  }
  if (opt_list->size() == 0) {
    return false;
  }
  return true;
}

bool PrintValueType::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<AnfNodePtr> opt_list;
  // first is pos, second is type: 0 is Scalar, 1 is ValueTuple
  std::vector<std::vector<std::pair<int64_t, int64_t>>> not_tensor_pos_vec;
  if (!GetOptList(node_list, &opt_list, &not_tensor_pos_vec)) {
    return false;
  }
  for (size_t idx = 0; idx < opt_list.size(); idx++) {
    auto node = opt_list[idx];
    CNodePtr cnode = utils::cast<CNodePtr>(node);
    MS_EXCEPTION_IF_NULL(cnode);
    auto value_type_vec = not_tensor_pos_vec[idx];
    // split value type and pos
    std::vector<int64_t> value_type_pos;
    std::vector<int64_t> value_type;
    (void)std::transform(value_type_vec.begin(), value_type_vec.end(), std::back_inserter(value_type_pos),
                         [](const std::pair<int64_t, int64_t> &value) { return value.first; });
    (void)std::transform(value_type_vec.begin(), value_type_vec.end(), std::back_inserter(value_type),
                         [](const std::pair<int64_t, int64_t> &value) { return value.second; });

    // hand over the attrs to new print
    common::AnfAlgo::SetNodeAttr("value_type", MakeValue<std::vector<int64_t>>(value_type), cnode);
    common::AnfAlgo::SetNodeAttr("value_type_pos", MakeValue<std::vector<int64_t>>(value_type_pos), cnode);
    // set output type and shape
    std::vector<TypeId> types;
    std::vector<BaseShapePtr> shapes;
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t i = 0; i < output_num; i++) {
      types.push_back(common::AnfAlgo::GetOutputInferDataType(cnode, i));
      shapes.push_back(AnfAlgo::GetOutputDetailShape(cnode, i));
    }
    common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, cnode.get());
    // add build info
    auto build_info = GenerateKernelBuildInfo(cnode);
    AnfAlgo::SetSelectKernelBuildInfo(build_info, cnode.get());
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
