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
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kSoftmaxInputNum = 1;
constexpr size_t kDropoutV3InputNum = 2;
constexpr size_t kSoftmaxDropoutOutputNum = 2;
constexpr size_t kSoftmaxInputShapeSize = 4;
constexpr int64_t kMaxInputH = 512;
constexpr int64_t kBlock = 32;
constexpr auto kAttrAxes = "axes";
constexpr auto kAttrKeepProb = "keep_prob";

bool CheckShapeIsUsePattern(int64_t input_h, int64_t input_w) {
  if (input_h == input_w && input_h <= kMaxInputH && input_h % kBlock == 0) {
    return true;
  }
  return false;
}

bool CheckSoftmax(const CNodePtr &softmax) {
  if (common::AnfAlgo::GetPrevNodeOutputInferDataType(softmax, 0) != kNumberTypeFloat16) {
    MS_LOG(DEBUG) << "Softmax's input type is float16, node: " << softmax->fullname_with_scope();
    return false;
  }

  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(softmax, 0);
  if (input_shape.size() != kSoftmaxInputShapeSize || !CheckShapeIsUsePattern(input_shape[kDim2], input_shape[kDim3])) {
    MS_LOG(DEBUG) << "Softmax's input shape is not supported, node: " << softmax->fullname_with_scope();
    return false;
  }

  return true;
}
}  // namespace

const BaseRef SoftmaxDropoutDoMaskV3Fusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr mask = std::make_shared<Var>();
  VectorRef softmax({prim::kPrimSoftmaxV2, X});
  VectorRef pattern({prim::kPrimDropOutDoMaskV3D, softmax, mask});
  return pattern;
}

const AnfNodePtr SoftmaxDropoutDoMaskV3Fusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                       const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout = CheckAnfNodeIfCNodeAndInputSize(node, kDropoutV3InputNum);
  auto softmax = CheckAnfNodeIfCNodeAndInputSize(dropout->input(1), kSoftmaxInputNum);
  if (!CheckSoftmax(softmax)) {
    return nullptr;
  }

  // create SoftmaxV2WithDropoutDoMaskV3D
  std::vector<AnfNodePtr> softmax_dropout_inputs = {
    NewValueNode(std::make_shared<Primitive>(kSoftmaxV2WithDropOutDoMaskV3DOpName)), softmax->input(kIndex1),
    dropout->input(kIndex2)};
  auto softmax_dropout = NewCNode(softmax_dropout_inputs, graph);
  MS_EXCEPTION_IF_NULL(softmax_dropout);
  auto types = {common::AnfAlgo::GetOutputInferDataType(softmax, 0),
                common::AnfAlgo::GetOutputInferDataType(dropout, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(softmax, 0), AnfAlgo::GetOutputDetailShape(dropout, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, softmax_dropout.get());
  softmax_dropout->set_scope(softmax->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, softmax, softmax_dropout);
  common::AnfAlgo::CopyNodeAttr(kAttrKeepProb, dropout, softmax_dropout);

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
