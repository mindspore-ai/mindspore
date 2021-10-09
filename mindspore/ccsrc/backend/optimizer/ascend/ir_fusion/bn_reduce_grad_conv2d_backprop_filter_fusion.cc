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
#include "backend/optimizer/ascend/ir_fusion/bn_reduce_grad_conv2d_backprop_filter_fusion.h"
#include <memory>
#include <vector>
#include <set>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/ascend/lic_manager.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kConv2DBackpropFilterInputNum = 2;
constexpr size_t kBNTrainingReduceGradInputNum = 7;
constexpr size_t kFusedDbnDwDropoutOutputNum = 2;
constexpr auto kAttrFilterSizes = "filter_sizes";
constexpr auto kAttrPadList = "pad_list";

bool CheckSupported(const CNodePtr &conv_back_filter) {
  auto y_shape = AnfAlgo::GetPrevNodeOutputInferShape(conv_back_filter, 0);
  auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(conv_back_filter, 1);
  auto out_shape = AnfAlgo::GetOutputInferShape(conv_back_filter, 0);
  if (y_shape.size() != kNCHWShapeSize || x_shape.size() != kNCHWShapeSize || out_shape.size() != kNCHWShapeSize) {
    MS_LOG(EXCEPTION) << "The dim of Conv2dBackpropFilter's input and output should be 4";
  }
  const std::set<size_t> kSupportedBatchSize = {32, 256};
  if (kSupportedBatchSize.find(x_shape[0]) == kSupportedBatchSize.end()) {
    return false;
  }

  std::vector<std::vector<size_t>> supported_cases = {
    // c_in, c_out, x_h, x_w, y_h, y_w, k_h, k_w
    {64, 256, 56, 56, 56, 56, 1, 1},  {256, 64, 56, 56, 56, 56, 1, 1},  {3, 64, 224, 224, 112, 112, 7, 7},
    {512, 128, 28, 28, 28, 28, 1, 1}, {64, 64, 56, 56, 56, 56, 3, 3},   {256, 512, 56, 56, 28, 28, 1, 1},
    {128, 512, 28, 28, 28, 28, 1, 1}, {256, 128, 56, 56, 56, 56, 1, 1}, {64, 64, 56, 56, 56, 56, 1, 1},
  };
  return std::any_of(
    supported_cases.begin(), supported_cases.end(), [&x_shape, &y_shape, &out_shape](const std::vector<size_t> &c) {
      return (c[kIndex0] == x_shape[kIndex1] && c[kIndex1] == y_shape[kIndex1] && c[kIndex2] == x_shape[kIndex2] &&
              c[kIndex3] == x_shape[kIndex3] && c[kIndex4] == y_shape[kIndex2] && c[kIndex5] == y_shape[kIndex3] &&
              c[kIndex6] == out_shape[kIndex2] && c[kIndex7] == out_shape[kIndex3]);
    });
}
}  // namespace

const BaseRef BNReduceGradConv2dBackpropFilterFusion::DefinePattern() const {
  VarPtr dbn_inputs = std::make_shared<SeqVar>();
  VarPtr X = std::make_shared<Var>();
  VectorRef bnreducegrad({prim::kPrimBNTrainingReduceGrad, dbn_inputs});
  VectorRef pattern({prim::kPrimConv2DBackpropFilter, bnreducegrad, X});
  return pattern;
}

const AnfNodePtr BNReduceGradConv2dBackpropFilterFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  if (!LicManager::GetInstance().GetPassSwitch(OptPassEnum::Resnet50DbnDwFusionPass)) {
    return nullptr;
  }

  auto conv_back_filter = CheckAnfNodeIfCNodeAndInputSize(node, kConv2DBackpropFilterInputNum);
  auto bnreduce_grad = CheckAnfNodeIfCNodeAndInputSize(conv_back_filter->input(kIndex1), kBNTrainingReduceGradInputNum);
  if (!CheckSupported(conv_back_filter)) {
    return nullptr;
  }

  // create FusedDbnDw
  std::vector<AnfNodePtr> fused_dbn_dw_inputs = {NewValueNode(std::make_shared<Primitive>(kFusedDbnDwOpName)),
                                                 conv_back_filter->input(kIndex2)};
  for (size_t i = 1; i <= kBNTrainingReduceGradInputNum; ++i) {
    fused_dbn_dw_inputs.push_back(bnreduce_grad->input(i));
  }
  auto fused_dbn_dw = graph->NewCNode(fused_dbn_dw_inputs);
  MS_EXCEPTION_IF_NULL(fused_dbn_dw);
  auto types = {AnfAlgo::GetOutputInferDataType(bnreduce_grad, 0),
                AnfAlgo::GetOutputInferDataType(conv_back_filter, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(bnreduce_grad, 0), AnfAlgo::GetOutputInferShape(conv_back_filter, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, fused_dbn_dw.get());
  fused_dbn_dw->set_scope(bnreduce_grad->scope());
  AnfAlgo::CopyNodeAttr(kAttrFilterSizes, conv_back_filter, fused_dbn_dw);
  AnfAlgo::CopyNodeAttr(kAttrStride, conv_back_filter, fused_dbn_dw);
  AnfAlgo::CopyNodeAttr(kAttrPadList, conv_back_filter, fused_dbn_dw);
  AnfAlgo::CopyNodeAttr(kAttrDilation, conv_back_filter, fused_dbn_dw);
  AnfAlgo::CopyNodeAttr(kAttrGroups, conv_back_filter, fused_dbn_dw);
  AnfAlgo::CopyNodeAttr(kAttrFormat, conv_back_filter, fused_dbn_dw);
  AnfAlgo::CopyNodeAttr(kAttrEpsilon, bnreduce_grad, fused_dbn_dw);

  // replace bnreduce_grad's output
  std::vector<AnfNodePtr> fused_dbn_dw_outputs;
  CreateMultipleOutputsOfAnfNode(graph, fused_dbn_dw, kFusedDbnDwDropoutOutputNum, &fused_dbn_dw_outputs);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(bnreduce_grad, fused_dbn_dw_outputs[0]);
  return fused_dbn_dw_outputs[1];
}
}  // namespace opt
}  // namespace mindspore
