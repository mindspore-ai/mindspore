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
#include "backend/optimizer/ascend/ir_fusion/confusion_softmax_grad_rule.h"

#include <memory>
#include <vector>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
bool NeedFusion(const AnfNodePtr &sum_anf, const AnfNodePtr &input0, const AnfNodePtr &input1) {
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
  if (!AnfAlgo::HasNodeAttr(kAttrAxis, reduce_sum)) {
    MS_LOG(INFO) << "ReduceSum should have attr axis if do fusion.";
    return false;
  }
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(reduce_sum, kAttrAxis);
  auto sum_input_shape = AnfAlgo::GetPrevNodeOutputInferShape(reduce_sum, 0);
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
  // check shape black list, wait for tbe op optimization
  std::set<std::vector<size_t>> shape_black_list = {{16, 4, 32, 32000}};
  auto input0_shape = AnfAlgo::GetOutputInferShape(input0, 0);
  auto input1_shape = AnfAlgo::GetOutputInferShape(input1, 0);
  if (input0_shape == input1_shape && shape_black_list.find(input0_shape) != shape_black_list.end()) {
    MS_LOG(INFO) << "Input shape in black list, quit fusion.";
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
  AnfAlgo::CopyNodeAttr(kAttrAxis, sum_anf, fusion_node);
  AnfAlgo::CopyNodeAttr(kAttrKeepDims, sum_anf, fusion_node);
  return fusion_node;
}
}  // namespace opt
}  // namespace mindspore
