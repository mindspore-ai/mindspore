/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/mindir/bn_grad_unify_mindir.h"

#include <vector>
#include <memory>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrUnifyIRPassed = "unifyir_passed";

AnfNodePtr CreateNewBatchNormGrad(const FuncGraphPtr &graph, const CNodePtr &bn_grad_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_grad_node);
  size_t kBNGradInputNum = 6;
  const auto &bn_grad_node_inputs = bn_grad_node->inputs();
  CheckCNodeInputSize(bn_grad_node, kBNGradInputNum);
  std::vector<AnfNodePtr> bn_grad_inputs = {NewValueNode(std::make_shared<Primitive>(kBatchNormGradOpName)),
                                            bn_grad_node_inputs[1],
                                            bn_grad_node_inputs[2],
                                            bn_grad_node_inputs[3],
                                            bn_grad_node_inputs[4],
                                            bn_grad_node_inputs[5]};
  auto new_bn_grad = graph->NewCNode(bn_grad_inputs);
  MS_EXCEPTION_IF_NULL(new_bn_grad);
  new_bn_grad->set_scope(bn_grad_node->scope());
  auto types = {AnfAlgo::GetOutputInferDataType(bn_grad_node, 0), AnfAlgo::GetOutputInferDataType(bn_grad_node, 1),
                AnfAlgo::GetOutputInferDataType(bn_grad_node, 2),
                AnfAlgo::GetPrevNodeOutputInferDataType(bn_grad_node, 3),
                AnfAlgo::GetPrevNodeOutputInferDataType(bn_grad_node, 4)};
  auto shapes = {AnfAlgo::GetOutputInferShape(bn_grad_node, 0), AnfAlgo::GetOutputInferShape(bn_grad_node, 1),
                 AnfAlgo::GetOutputInferShape(bn_grad_node, 2), AnfAlgo::GetPrevNodeOutputInferShape(bn_grad_node, 3),
                 AnfAlgo::GetPrevNodeOutputInferShape(bn_grad_node, 4)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, new_bn_grad.get());
  AnfAlgo::CopyNodeAttrs(bn_grad_node, new_bn_grad);
  AnfAlgo::SetNodeAttr(kAttrUnifyIRPassed, MakeValue(true), new_bn_grad);
  return new_bn_grad;
}
}  // namespace

const BaseRef BatchNormGradUnifyMindIR::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kBatchNormGradOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr BatchNormGradUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::HasNodeAttr(kAttrUnifyIRPassed, cnode)) {
    return nullptr;
  }
  return CreateNewBatchNormGrad(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
