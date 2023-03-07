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
#include "plugin/device/cpu/optimizer/softmax_grad_fusion.h"
#include <memory>
#include <vector>
#include <set>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
bool NeedFusion(const AnfNodePtr &reduce_sum, const AnfNodePtr &, const AnfNodePtr &) {
  if (reduce_sum == nullptr || !reduce_sum->isa<CNode>()) {
    MS_LOG(WARNING) << "Matched ReduceSum is not a CNode!";
    return false;
  }
  auto reduce_sum_node = reduce_sum->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(reduce_sum_node);
  if (!GetBoolAttr(reduce_sum_node, kAttrKeepDims)) {
    MS_LOG(INFO) << "ReduceSum's attr keep_dims should be true if do fusion. Otherwise the calculation will be wrong.";
    return false;
  }

  // check axis of ReduceSum should be the last dim
  auto axis_node = common::AnfAlgo::GetInputNode(reduce_sum_node, 1);
  MS_EXCEPTION_IF_NULL(axis_node);
  if (!axis_node->isa<ValueNode>()) {
    MS_LOG(INFO) << "ReduceSum's axis input should be an ValueNode if do fusion.";
    return false;
  }

  int64_t axis = 1;
  auto axis_node_ptr = axis_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(axis_node_ptr);
  auto value_node = GetValueNode(axis_node_ptr);
  MS_EXCEPTION_IF_NULL(value_node);
  if (value_node->isa<tensor::Tensor>()) {
    auto const_tensor = value_node->cast<tensor::TensorPtr>();
    std::vector<int64_t> axis_vector = TensorValueToVector<int64_t>(const_tensor);
    if (axis_vector.size() != 1) {
      MS_LOG(INFO) << "The size of ReduceSum's axis input should be 1 if do fusion, but got " << axis_vector.size();
      return false;
    }
    axis = axis_vector[0];
  } else {
    MS_LOG(INFO) << "ReduceSum's axis input should be a Tensor if do fusion.";
    return false;
  }

  auto dtype = common::AnfAlgo::GetOutputInferDataType(reduce_sum_node, 0);
  if (dtype != kNumberTypeFloat32) {
    MS_LOG(INFO) << kConfusionSoftmaxGradOpName << " cpu kernel only supports float32 currently.";
    return false;
  }

  auto sum_input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(reduce_sum_node, 0);
  auto sum_input_size = SizeToLong(sum_input_shape.size());
  if (sum_input_size == 0) {
    MS_LOG(INFO) << "ReduceSum's input should not be a scalar if do fusion.";
    return false;
  }
  axis = axis < 0 ? axis + sum_input_size : axis;
  if (axis % sum_input_size != sum_input_size - 1) {
    MS_LOG(INFO) << "ReduceSum's attr axis should be the last dim if do fusion.";
    return false;
  }

  return true;
}
}  // namespace

const BaseRef SoftmaxGradFusionCpu::DefinePattern() const {
  // pattern: mul(out, sub(dout, reduce_sum(mul(out, dout), -1)))
  VectorRef mul = VectorRef({prim::kPrimMul, input0_, input1_});
  VectorRef reduce_sum = VectorRef({reduce_sum_, mul, axis_});
  VectorRef sub = VectorRef({prim::kPrimSub, input1_, reduce_sum});
  VectorRef pattern = VectorRef({prim::kPrimMul, input0_, sub});
  return pattern;
}

const AnfNodePtr SoftmaxGradFusionCpu::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  AnfNodePtr input0 = GetAnfNodeByVar(equiv, input0_);
  AnfNodePtr input1 = GetAnfNodeByVar(equiv, input1_);
  AnfNodePtr reduce_sum = GetAnfNodeByVar(equiv, reduce_sum_);
  MS_EXCEPTION_IF_NULL(input0);
  MS_EXCEPTION_IF_NULL(input1);
  if (!NeedFusion(reduce_sum, input0, input1)) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kSoftmaxGradFusionOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), input0, input1};
  auto fused_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(fused_node);
  fused_node->set_abstract(node->abstract());
  fused_node->set_scope(node->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, reduce_sum, fused_node);
  common::AnfAlgo::CopyNodeAttr(kAttrKeepDims, reduce_sum, fused_node);
  return fused_node;
}
}  // namespace opt
}  // namespace mindspore
