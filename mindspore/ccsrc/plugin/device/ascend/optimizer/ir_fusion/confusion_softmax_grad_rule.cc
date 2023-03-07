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
#include "plugin/device/ascend/optimizer/ir_fusion/confusion_softmax_grad_rule.h"

#include <memory>
#include <vector>
#include <set>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
bool NeedFusion(const AnfNodePtr &sum_anf, const AnfNodePtr &input0, const AnfNodePtr &) {
  if (sum_anf == nullptr || !sum_anf->isa<CNode>()) {
    MS_LOG(WARNING) << "Matched ReduceSum is not a CNode!";
    return false;
  }
  auto reduce_sum = sum_anf->cast<CNodePtr>();
  if (!GetBoolAttr(reduce_sum, kAttrKeepDims)) {
    MS_LOG(INFO) << "ReduceSum's attr keep_dims should be true if do fusion. Otherwise the calculation will be wrong.";
    return false;
  }

  // check axis should be last dim
  auto prim = common::AnfAlgo::GetCNodePrimitive(reduce_sum);
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasAttr(kAttrAxis)) {
    MS_LOG(INFO) << "ReduceSum should have attr axis if do fusion.";
    return false;
  }
  auto axis_value = prim->GetAttr(kAttrAxis);
  int64_t axis;
  if (axis_value->isa<Int64Imm>()) {
    axis = GetValue<int64_t>(axis_value);
  } else if (axis_value->isa<ValueTuple>()) {
    auto axis_tuple = GetValue<std::vector<int64_t>>(axis_value);
    if (axis_tuple.size() != 1) {
      MS_LOG(INFO) << "ReduceSum's attr axis size should be 1 if do fusion.";
      return false;
    }
    axis = axis_tuple[0];
  } else {
    return false;
  }
  auto sum_input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(reduce_sum, 0);
  auto sum_input_size = SizeToLong(sum_input_shape.size());
  if (sum_input_size == 0) {
    MS_LOG(INFO) << "ReduceSum's input should not be a scalar if do fusion.";
    return false;
  }
  axis = axis < 0 ? axis + sum_input_size : axis;
  if (axis % sum_input_size != sum_input_size - 1) {
    MS_LOG(INFO) << "ReduceSum's attr axis should be last dim if do fusion.";
    return false;
  }

  const ShapeValueDType last_dim_limit = 30000;
  auto input0_shape = common::AnfAlgo::GetOutputInferShape(input0, 0);
  if (!input0_shape.empty() && input0_shape[input0_shape.size() - 1] > last_dim_limit) {
    MS_LOG(INFO) << "Input shape is too large to optimize, quit fusion, shape: " << input0_shape;
    return false;
  }

  return true;
}
}  // namespace

const BaseRef ConfusionSoftmaxGradRule::DefinePattern() const {
  return VectorRef({prim::kPrimSub, input0_, VectorRef({reduce_sum_, VectorRef({prim::kPrimMul, input1_, input0_})})});
}

const AnfNodePtr ConfusionSoftmaxGradRule::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  AnfNodePtr input0 = GetAnfNodeByVar(equiv, input0_);
  AnfNodePtr input1 = GetAnfNodeByVar(equiv, input1_);
  AnfNodePtr sum_anf = GetAnfNodeByVar(equiv, reduce_sum_);
  if (!NeedFusion(sum_anf, input0, input1)) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kConfusionSoftmaxGradOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), input0, input1};
  auto fusion_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(fusion_node);
  fusion_node->set_abstract(node->abstract());
  fusion_node->set_scope(node->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, sum_anf, fusion_node);
  common::AnfAlgo::CopyNodeAttr(kAttrKeepDims, sum_anf, fusion_node);
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
