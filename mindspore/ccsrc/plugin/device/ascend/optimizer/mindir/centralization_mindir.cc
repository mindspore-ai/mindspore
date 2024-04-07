/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/centralization_mindir.h"
#include <vector>
#include <memory>
#include "ops/math_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/ms_device_shape_transfer.h"

namespace mindspore {
namespace opt {
namespace {
const auto kShapeThreshold = 1000;

bool CheckDeviceShape(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto shape = common::AnfAlgo::GetOutputInferShape(cnode, 0);
  if (shape.size() != kShape4dDims) {
    return true;
  }
  auto type = common::AnfAlgo::GetOutputInferDataType(cnode, 0);
  auto device_shape = trans::TransShapeToDevice(shape, kOpFormat_FRAC_Z, type);
  return device_shape[kDim0] < kShapeThreshold;
}

std::vector<int64_t> GetReduceMeanInferShape(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto axis = cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(axis);
  auto value_node = axis->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto axis_value = GetValue<std::vector<int64_t>>(value);
  auto shape = common::AnfAlgo::GetOutputInferShape(cnode, 0);
  std::vector<int64_t> output_shape = shape;

  if (shape[kDim0] == -1) {
    return std::vector<int64_t>();
  }

  for (size_t i = 0; i < axis_value.size(); i++) {
    if (LongToSize(axis_value[i]) > output_shape.size()) {
      MS_LOG(EXCEPTION) << "Axis " << axis_value[i] << " is out of range [0, " << output_shape.size() << ")"
                        << ", node: " << cnode->fullname_with_scope();
    }
    output_shape[axis_value[i]] = 1;
  }
  return output_shape;
}
}  // namespace
const BaseRef CentralizationMindIR::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  return VectorRef({prim::kPrimCentralization, x1, x2});
}

// compute centralization: y = x - mean(x , axis)
const AnfNodePtr CentralizationMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckDeviceShape(cnode)) {
    return nullptr;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto keep_dims_node = NewValueNode(MakeValue(true));
  kernel_graph->AddValueNodeToGraph(keep_dims_node);

  std::vector<AnfNodePtr> reduce_mean_input = {NewValueNode(std::make_shared<Primitive>(kReduceMeanOpName)),
                                               cnode->input(kIndex1), cnode->input(kIndex2), keep_dims_node};
  auto reduce_mean_node = NewCNode(reduce_mean_input, func_graph);
  MS_EXCEPTION_IF_NULL(reduce_mean_node);
  auto shape = GetReduceMeanInferShape(cnode);
  auto type = common::AnfAlgo::GetOutputInferDataType(cnode, 0);
  common::AnfAlgo::SetOutputInferTypeAndShape({type}, {shape}, reduce_mean_node.get());
  reduce_mean_node->set_scope(node->scope());

  std::vector<AnfNodePtr> sub_input = {NewValueNode(std::make_shared<Primitive>(kSubOpName)), cnode->input(1),
                                       reduce_mean_node};
  auto sub_node = NewCNode(sub_input, func_graph);
  MS_EXCEPTION_IF_NULL(sub_node);
  sub_node->set_abstract(cnode->abstract());
  sub_node->set_scope(cnode->scope());
  MS_LOG(INFO) << "Node has been replaced, node is: " << node->fullname_with_scope();
  return sub_node;
}
}  // namespace opt
}  // namespace mindspore
