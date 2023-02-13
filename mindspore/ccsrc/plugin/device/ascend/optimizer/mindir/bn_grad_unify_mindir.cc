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
#include "plugin/device/ascend/optimizer/mindir/bn_grad_unify_mindir.h"
#include <vector>
#include <memory>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrUnifyIRPassed = "unifyir_passed";
constexpr auto kX1 = "X1";
constexpr auto kX2 = "X2";
constexpr auto kX3 = "X3";
constexpr auto kX4 = "X4";
constexpr auto kX5 = "X5";
constexpr auto kXs = "Xs";
constexpr auto kMBatchnormGrad = "m_batchnorm_grad";
constexpr auto kRBatchnormGrad = "r_batchnorm_grad";
}  // namespace

AnfNodePtr BuildBatchNormGrad(const PatternMap &m, const AnfNodePtr &new_node) {
  auto node = m.Get(kMBatchnormGrad);
  MS_EXCEPTION_IF_NULL(node);
  auto bn_grad_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(bn_grad_node);
  size_t kBNGradInputNum = 6;
  CheckCNodeInputSize(bn_grad_node, kBNGradInputNum);
  auto new_bn_grad = new_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(new_bn_grad);
  MS_EXCEPTION_IF_NULL(new_bn_grad);
  new_bn_grad->set_scope(bn_grad_node->scope());
  auto types = {common::AnfAlgo::GetOutputInferDataType(bn_grad_node, 0UL),
                common::AnfAlgo::GetOutputInferDataType(bn_grad_node, 1UL),
                common::AnfAlgo::GetOutputInferDataType(bn_grad_node, 2UL),
                common::AnfAlgo::GetPrevNodeOutputInferDataType(bn_grad_node, 3UL),
                common::AnfAlgo::GetPrevNodeOutputInferDataType(bn_grad_node, 4UL)};
  auto shapes = {common::AnfAlgo::GetOutputDetailShape(bn_grad_node, 0UL),
                 common::AnfAlgo::GetOutputDetailShape(bn_grad_node, 1UL),
                 common::AnfAlgo::GetOutputDetailShape(bn_grad_node, 2UL),
                 common::AnfAlgo::GetPrevNodeOutputDetailShape(bn_grad_node, 3UL),
                 common::AnfAlgo::GetPrevNodeOutputDetailShape(bn_grad_node, 4UL)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, new_bn_grad.get());
  common::AnfAlgo::CopyNodeAttrs(bn_grad_node, new_bn_grad);
  common::AnfAlgo::SetNodeAttr(kAttrUnifyIRPassed, MakeValue(true), new_bn_grad);
  return new_bn_grad;
}

bool BatchNormGradUnifyMindIR::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &func_graph,
                                               const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrUnifyIRPassed, cnode) ||
      (func_graph->has_flag(kAttrMutableKernel) && !GetBoolAttr(cnode, kAttrIsTraining))) {
    return false;
  }
  return true;
}

void BatchNormGradUnifyMindIR::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern)
    .AddVar(kX1)
    .AddVar(kX2)
    .AddVar(kX3)
    .AddVar(kX4)
    .AddVar(kX5)
    .AddSeqVar(kXs)
    .AddCNode(kMBatchnormGrad, {std::make_shared<Primitive>(kBatchNormGradOpName), kX1, kX2, kX3, kX4, kX5, kXs});
}

void BatchNormGradUnifyMindIR::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern)
    .AddCNode(kRBatchnormGrad, {std::make_shared<Primitive>(kBatchNormGradOpName), kX1, kX2, kX3, kX4, kX5},
              BuildBatchNormGrad);
}
}  // namespace opt
}  // namespace mindspore
