/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/axis_normalizer.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "backend/common/graph_kernel/adapter/callback_impl.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "include/common/utils/anfalgo.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel {
int64_t AxisNormalizer::NormAxis(int64_t x, size_t rank) const { return x >= 0 ? x : x + static_cast<int64_t>(rank); }

bool AxisNormalizer::IsReduce(const AnfNodePtr &node) const {
  std::vector<PrimitivePtr> node_with_axis = {prim::kPrimReduceSum, prim::kPrimReduceMax, prim::kPrimReduceMin,
                                              prim::kPrimArgMax, prim::kPrimArgmin};
  return std::any_of(node_with_axis.begin(), node_with_axis.end(),
                     [&node](const PrimitivePtr &p) { return IsPrimitiveCNode(node, p); });
}

bool AxisNormalizer::AxisProcess(ValuePtr axis, const size_t rank, ShapeVector *axis_vec) const {
  bool diff = false;
  if (axis->isa<Int32Imm>() || axis->isa<Int64Imm>()) {
    auto v1 = AnfUtils::GetIntValue(axis);
    auto v2 = NormAxis(v1, rank);
    axis_vec->push_back(v2);
  } else if (axis->isa<ValueSequence>()) {
    auto vec = axis->cast<ValueSequencePtr>()->value();
    if (vec.empty()) {
      diff = true;
      for (size_t i = 0; i < rank; i++) {
        axis_vec->push_back(i);
      }
    } else if (vec[0]->isa<Int32Imm>() || vec[0]->isa<Int64Imm>()) {
      for (auto v : vec) {
        auto v1 = AnfUtils::GetIntValue(v);
        auto v2 = NormAxis(v1, rank);
        axis_vec->push_back(v2);
        diff = diff || (v1 != v2);
      }
    }
  } else if (axis->isa<tensor::Tensor>()) {
    auto raw_axis_vec = CheckAndConvertUtils::CheckTensorIntValue("axis", axis, "ReduceOp");
    if (raw_axis_vec.empty()) {
      diff = true;
      for (size_t i = 0; i < rank; i++) {
        axis_vec->push_back(i);
      }
    } else {
      for (auto v1 : raw_axis_vec) {
        auto v2 = NormAxis(v1, rank);
        axis_vec->push_back(v2);
      }
      // if tensor shape is empty, create a new 1-d tensor
      auto axis_tensor = axis->cast<tensor::TensorPtr>();
      diff = axis_tensor->shape_c().empty() || raw_axis_vec != *axis_vec;
    }
  }

  return diff;
}

bool AxisNormalizer::Process(const AnfNodePtr &graph_kernel_node) const {
  auto sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(graph_kernel_node);
  auto parameters = sub_func_graph->parameters();
  auto inputs = graph_kernel_node->cast<CNodePtr>()->inputs();
  std::unordered_map<AnfNodePtr, size_t> param_idx_map;
  for (size_t i = 0; i < parameters.size(); ++i) {
    param_idx_map[parameters[i]] = i;
  }
  bool changed = false;
  auto todos = TopoSort(sub_func_graph->get_return());
  for (auto node : todos) {
    if (!IsReduce(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const size_t axis_idx = 2;
    auto axis_node = cnode->input(axis_idx);
    ValuePtr axis = nullptr;
    if (axis_node->isa<ValueNode>()) {
      auto axis_value_node = axis_node->cast<ValueNodePtr>();
      axis = axis_value_node->value();
    } else {  // Parameter
      axis = axis_node->abstract()->BuildValue();
    }
    size_t rank = Callback::Instance()->GetInputShape(node, 0).size();
    if (rank == 0) {
      // scalar tensor
      rank = 1;
    }
    ShapeVector axis_vec;
    auto diff = AxisProcess(axis, rank, &axis_vec);
    if (diff) {
      changed = true;
      std::sort(axis_vec.begin(), axis_vec.end());
      ValuePtr new_axis_value = nullptr;
      new_axis_value = std::make_shared<tensor::Tensor>(axis_vec);
      auto new_axis_node = std::make_shared<ValueNode>(new_axis_value);
      new_axis_node->set_abstract(new_axis_value->ToAbstract());
      if (axis_node->isa<ValueNode>()) {
        cnode->set_input(axis_idx, new_axis_node);
      } else {
        auto idx = param_idx_map[axis_node];
        auto &input_node = inputs[idx + 1];
        auto input_value_node = input_node->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(input_value_node);
        input_value_node->set_abstract(new_axis_node->abstract());
        input_value_node->set_value(new_axis_value);
        axis_node->set_abstract(new_axis_node->abstract());
      }
    }
  }
  return changed;
}

bool AxisNormalizer::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (auto node : todos) {
    if (common::AnfAlgo::IsGraphKernel(node)) {
      changed = Process(node) || changed;
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace mindspore::graphkernel
