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

#include "parallel/graph_util/get_parallel_info.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/utils.h"
#include "ir/func_graph.h"
#include "parallel/graph_util/graph_info.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {
py::dict GetParameterLayout(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  py::dict dict;
  std::vector<AnfNodePtr> graph_params = graph->parameters();

  for (auto para : graph_params) {
    std::string name = std::static_pointer_cast<Parameter>(para)->name();
    std::shared_ptr<parallel::TensorLayout> tensor_layout = std::static_pointer_cast<Parameter>(para)->tensor_layout();
    if (tensor_layout == nullptr) {
      MS_LOG(INFO) << "GetParameterLayout nullptr name = " << name;
    } else {
      auto device_arrangement = tensor_layout->device_arrangement().array();
      auto tensor_map = tensor_layout->tensor_map().array();
      auto slice_shape = tensor_layout->slice_shape().array();
      std::vector<std::vector<int32_t>> layout = {device_arrangement, tensor_map, slice_shape};
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
      auto distributed_operation_info = cnode->operator_info();
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
