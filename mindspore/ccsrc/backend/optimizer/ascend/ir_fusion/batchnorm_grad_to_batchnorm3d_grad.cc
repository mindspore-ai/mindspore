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
#include "backend/optimizer/ascend/ir_fusion/batchnorm_grad_to_batchnorm3d_grad.h"
#include <memory>
#include <string>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kBN3DGradInputXIndex = 2;
CNodePtr CreateBatchNorm3DGrad(const FuncGraphPtr &graph, const CNodePtr &batchnorm_grad) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnorm_grad);
  auto prim = std::make_shared<Primitive>(kBatchNorm3DGradOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  for (size_t i = 1; i < batchnorm_grad->size(); ++i) {
    inputs.push_back(batchnorm_grad->input(i));
  }
  auto new_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(batchnorm_grad->scope());
  new_node->set_abstract(batchnorm_grad->abstract());
  AnfAlgo::CopyNodeAttrs(batchnorm_grad, new_node);
  return new_node;
}

bool NeedFusion(const FuncGraphPtr &graph, const CNodePtr &batchnorm_grad) {
  MS_EXCEPTION_IF_NULL(batchnorm_grad);
  if (AnfAlgo::GetInputTensorNum(batchnorm_grad) < kBNGradInputTensorNum) {
    MS_LOG(INFO) << "BatchNormGrad's input less than " << kBNGradInputTensorNum;
    return false;
  }
  auto format = AnfAlgo::GetNodeAttr<std::string>(batchnorm_grad, kAttrFormat);
  const auto &ori_inputs = batchnorm_grad->inputs();
  auto x_shape = AnfAlgo::GetOutputInferShape(ori_inputs[kBN3DGradInputXIndex], 0);
  if (format != kOpFormat_NCDHW || x_shape.size() != 5) {
    MS_LOG(INFO) << "Only format is NCDHW and the input dim of BatchNormGrad is 5, then do fusion. But format is: "
                 << format << ", size of x_shape is: " << x_shape.size();
    return false;
  }
  return true;
}
}  // namespace

const BaseRef BatchNormGrad2BatchNorm3DGRAD::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  VectorRef pattern({prim::kPrimBatchNormGrad, Xs});
  return pattern;
}

const AnfNodePtr BatchNormGrad2BatchNorm3DGRAD::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode_bn_grad = node->cast<CNodePtr>();
  if (!NeedFusion(graph, cnode_bn_grad)) {
    return nullptr;
  }
  auto bn_3d_grad = CreateBatchNorm3DGrad(graph, cnode_bn_grad);
  TransferDepend(cnode_bn_grad, graph, bn_3d_grad);
  return bn_3d_grad;
}
}  // namespace opt
}  // namespace mindspore
