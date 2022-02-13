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
#include "plugin/device/ascend/optimizer/ir_fusion/softmax_dropout_do_mask_v3_fusion.h"
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kSoftmaxInputNum = 1;
constexpr size_t kDropoutV3InputNum = 2;
constexpr size_t kSoftmaxDropoutOutputNum = 2;
constexpr auto kAttrAxes = "axes";
constexpr auto kAttrKeepProb = "keep_prob";
}  // namespace

const BaseRef SoftmaxDropoutDoMaskV3Fusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr mask = std::make_shared<Var>();
  VectorRef softmax({prim::kPrimSoftmax, X});
  VectorRef pattern({prim::kPrimDropoutDoMaskV3, softmax, mask});
  return pattern;
}

const AnfNodePtr SoftmaxDropoutDoMaskV3Fusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                       const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout = CheckAnfNodeIfCNodeAndInputSize(node, kDropoutV3InputNum);
  auto softmax = CheckAnfNodeIfCNodeAndInputSize(dropout->input(1), kSoftmaxInputNum);

  // create SoftmaxV2WithDropoutDoMaskV3D
  std::vector<AnfNodePtr> softmax_dropout_inputs = {
    NewValueNode(std::make_shared<Primitive>(kSoftmaxV2WithDropoutDoMaskV3OpName)), softmax->input(kIndex1),
    dropout->input(kIndex2)};
  auto softmax_dropout = NewCNode(softmax_dropout_inputs, graph);
  MS_EXCEPTION_IF_NULL(softmax_dropout);
  auto types = {AnfAlgo::GetOutputInferDataType(softmax, 0), AnfAlgo::GetOutputInferDataType(dropout, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(softmax, 0), AnfAlgo::GetOutputInferShape(dropout, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, softmax_dropout.get());
  softmax_dropout->set_scope(softmax->scope());
  AnfAlgo::CopyNodeAttr(kAttrAxis, softmax, softmax_dropout);
  AnfAlgo::CopyNodeAttr(kAttrKeepProb, dropout, softmax_dropout);

  // replace softmax's output
  std::vector<AnfNodePtr> softmax_dropout_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_dropout, kSoftmaxDropoutOutputNum, &softmax_dropout_outputs);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(softmax, softmax_dropout_outputs[0]);
  return softmax_dropout_outputs[1];
}
}  // namespace opt
}  // namespace mindspore
