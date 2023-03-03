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
#include <vector>
#include "ir/scalar.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::graphkernel {
int64_t AxisNormalizer::NormAxis(int64_t x, size_t rank) const { return x >= 0 ? x : x + static_cast<int64_t>(rank); }

bool AxisNormalizer::IsReduce(const AnfNodePtr &node) const {
  std::vector<PrimitivePtr> node_with_axis = {prim::kPrimReduceSum, prim::kPrimReduceMax, prim::kPrimReduceMin,
                                              prim::kPrimArgMax, prim::kPrimArgmin};
  return std::any_of(node_with_axis.begin(), node_with_axis.end(),
                     [&node](const PrimitivePtr &p) { return IsPrimitiveCNode(node, p); });
}

bool AxisNormalizer::Process(const FuncGraphPtr &func_graph) const {
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (auto node : todos) {
    if (!IsReduce(node)) {
      continue;
    }
    if (auto primitive = GetCNodePrimitive(node); primitive != nullptr && primitive->HasAttr(kAttrAxis)) {
      auto axis = primitive->GetAttr(kAttrAxis);
      size_t rank = AnfAlgo::GetInputDeviceShape(node, 0).size();
      if (rank == 0) {
        // scalar tensor
        rank = 1;
      }
      bool diff = false;
      ShapeVector axis_vec;
      if (axis->isa<Int32Imm>() || axis->isa<Int64Imm>()) {
        auto v1 = GetValue<int64_t>(axis);
        auto v2 = NormAxis(v1, rank);
        axis_vec.push_back(v2);
        diff = true;
      } else if (axis->isa<ValueSequence>()) {
        auto vec = axis->cast<ValueSequencePtr>()->value();
        if (vec.empty()) {
          diff = true;
          for (size_t i = 0; i < rank; i++) {
            axis_vec.push_back(i);
          }
        } else if (vec[0]->isa<Int32Imm>() || vec[0]->isa<Int64Imm>()) {
          for (auto v : vec) {
            auto v1 = GetValue<int64_t>(v);
            auto v2 = NormAxis(v1, rank);
            axis_vec.push_back(v2);
            diff = diff || (v1 != v2);
          }
        }
      }
      if (diff) {
        changed = true;
        std::sort(axis_vec.begin(), axis_vec.end());
        SetNodeAttrSafely(kAttrAxis, MakeValue(axis_vec), node);
      }
    }
  }
  return changed;
}

bool AxisNormalizer::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (auto node : todos) {
    if (common::AnfAlgo::IsGraphKernel(node)) {
      auto sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      changed = Process(sub_func_graph) || changed;
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
