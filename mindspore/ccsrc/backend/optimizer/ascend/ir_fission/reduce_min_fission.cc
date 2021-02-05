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
#include "backend/optimizer/ascend/ir_fission/reduce_min_fission.h"
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr CreateReduceMin(const FuncGraphPtr &graph, const AnfNodePtr &input, const CNodePtr &old_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(old_node);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceMin->name())), input};
  CNodePtr reduce_min = graph->NewCNode(inputs);
  reduce_min->set_scope(old_node->scope());
  AnfAlgo::CopyNodeAttr(kAttrKeepDims, old_node, reduce_min);
  return reduce_min;
}

bool NeedOptimize(const TypeId &dtype, const std::vector<size_t> &shape, const std::vector<int64_t> &axis) {
  if (dtype != kNumberTypeFloat32) {
    MS_LOG(INFO) << "ReduceMin's input Dtype is not float32, no need optimize!";
    return false;
  }
  if (shape.size() == 0 || shape.size() == 1) {
    MS_LOG(INFO) << "ReduceMin's input shape size is " << shape.size() << ", no need optimize!";
    return false;
  }
  if (axis.size() == 1) {
    MS_LOG(INFO) << "ReduceMin axis size is 1, no need optimize!";
    return false;
  }
  int64_t last_dim = SizeToLong(shape.size() - 1);
  if (std::find(axis.begin(), axis.end(), -1) == axis.end() &&
      std::find(axis.begin(), axis.end(), last_dim) == axis.end()) {
    MS_LOG(INFO) << "Attribute of axis does not contain the last axis, not match!";
    return false;
  }
  return true;
}

std::vector<int64_t> CalFirstAxis(const std::vector<size_t> &shape, const std::vector<int64_t> &axis) {
  std::vector<int64_t> axis_fisrt;
  int64_t last_dim = SizeToLong(shape.size() - 1);
  std::copy_if(axis.begin(), axis.end(), std::back_inserter(axis_fisrt),
               [&last_dim](int64_t v) { return v != -1 && v != last_dim; });

  int64_t dim_size = SizeToLong(shape.size());
  if (axis_fisrt.empty()) {
    for (int64_t i = 0; i < dim_size - 1; ++i) {
      axis_fisrt.push_back(i);
    }
  }

  for (size_t i = 0; i < axis_fisrt.size(); ++i) {
    if (axis_fisrt[i] < -dim_size || axis_fisrt[i] > dim_size - 1) {
      MS_LOG(EXCEPTION) << "The axis of ReduceMin verify failed, quit optimizing";
    }
    if (axis_fisrt[i] < 0) {
      axis_fisrt[i] = dim_size + axis_fisrt[i];
    }
  }
  return axis_fisrt;
}

std::vector<size_t> GetInferShape(const std::vector<size_t> &shape, const std::vector<int64_t> &axis_first,
                                  bool keep_dims) {
  std::vector<size_t> shape_first;
  for (size_t item = 0; item < shape.size(); ++item) {
    if (axis_first.end() != std::find(axis_first.begin(), axis_first.end(), item)) {
      if (keep_dims) {
        // If keep_dims is true, current dimension set to 1
        shape_first.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      shape_first.push_back(shape[item]);
    }
  }
  return shape_first;
}
}  // namespace

const BaseRef ReduceMinFission::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimReduceMin, X});
}

const AnfNodePtr ReduceMinFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (graph == nullptr || node == nullptr) {
    return nullptr;
  }
  if (AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  CheckCNodeInputSize(cnode, 1);
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  auto dtype = AnfAlgo::GetPrevNodeOutputInferDataType(cnode, 0);
  auto prim = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasAttr(kAttrAxis) || !prim->HasAttr(kAttrKeepDims)) {
    MS_LOG(INFO) << "ReduceMin has no axis or keep_dims, no need optimize!";
    return nullptr;
  }
  auto axis_value = prim->GetAttr(kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  if (!axis_value->isa<ValueSequeue>()) {
    return nullptr;
  }
  auto axis = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrAxis);
  auto keep_dims = AnfAlgo::GetNodeAttr<bool>(cnode, kAttrKeepDims);

  if (!NeedOptimize(dtype, shape, axis)) {
    MS_LOG(INFO) << "No need optimize for this ReduceMin. " << cnode->DebugString();
    return nullptr;
  }

  // Create reduce_min1
  CNodePtr reduce_min1 = CreateReduceMin(graph, cnode->input(1), cnode);
  std::vector<int64_t> axis_first = CalFirstAxis(shape, axis);
  std::vector<size_t> shape_first = GetInferShape(shape, axis_first, keep_dims);
  AnfAlgo::SetOutputInferTypeAndShape({dtype}, {shape_first}, reduce_min1.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis_first), reduce_min1);

  // Create reduce_min2
  CNodePtr reduce_min2 = CreateReduceMin(graph, reduce_min1, cnode);
  reduce_min2->set_abstract(cnode->abstract());
  std::vector<int64_t> axis_last = {-1};
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis_last), reduce_min2);
  return reduce_min2;
}
}  // namespace opt
}  // namespace mindspore
