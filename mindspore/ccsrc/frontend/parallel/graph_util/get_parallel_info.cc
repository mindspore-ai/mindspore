/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/graph_util/get_parallel_info.h"

#include <memory>
#include <string>
#include <vector>

#include "ir/func_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {
py::dict GetParameterLayout(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  py::dict dict;
  std::vector<AnfNodePtr> graph_params = graph->parameters();

  for (auto para : graph_params) {
    std::string name = std::static_pointer_cast<Parameter>(para)->name();
    auto tensor_layout = para->user_data<parallel::TensorLayout>();
    if (tensor_layout == nullptr) {
      MS_LOG(INFO) << "GetParameterLayout nullptr name = " << name;
    } else {
      auto device_arrangement = tensor_layout->device_arrangement().array();
      auto tensor_map = tensor_layout->tensor_map().array();
      auto slice_shape = tensor_layout->slice_shape().array();
      Shape field_size = {tensor_layout->get_field_size()};
      Shape uniform_split;
      if (tensor_layout->uniform_split()) {
        uniform_split.push_back(1);
      } else {
        uniform_split.push_back(0);
      }

      std::vector<Shape> layout = {device_arrangement, tensor_map, slice_shape, field_size, uniform_split};
      dict[py::str(name)] = layout;
      MS_LOG(INFO) << "GetParameterLayout name = " << name << ", layout " << tensor_layout->ToString();
    }
  }
  return dict;
}

py::dict GetCNodeStrategy(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  py::dict dict;
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto nodes = DeepScopedGraphSearch(ret);

  for (auto node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto distributed_operation_info = cnode->user_data<OperatorInfo>();
      if (distributed_operation_info != nullptr) {
        auto strategyPtr = distributed_operation_info->strategy();
        if (strategyPtr != nullptr) {
          auto strategy = strategyPtr->GetInputDim();
          auto name = cnode->fullname_with_scope();
          dict[py::str(name)] = strategy;
        }
      }
    }
  }
  return dict;
}

py::dict GetAllreduceFusion(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  py::dict dict;
  auto allreduce_prim_list = FindPrimtive(graph, ALL_REDUCE);

  for (auto prim : allreduce_prim_list) {
    auto name_ptr = prim->GetAttr("parameter");
    auto fusion_ptr = prim->GetAttr("fusion");
    if (fusion_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "fusion_ptr is nullptr";
    } else if (name_ptr == nullptr) {
      continue;
    }
    if (!name_ptr->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "name is not StringImm";
    }
    auto name = name_ptr->cast<StringImmPtr>()->value();
    if (!fusion_ptr->isa<Int32Imm>()) {
      MS_LOG(EXCEPTION) << "fusion is not Int32Imm";
    }
    int32_t fusion = fusion_ptr->cast<Int32ImmPtr>()->value();
    dict[py::str(name)] = fusion;
  }
  return dict;
}
}  // namespace parallel
}  // namespace mindspore
