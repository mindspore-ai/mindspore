/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/batchnorm_to_bninfer.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kIdxScale = 2;
constexpr size_t kIdxBias = 3;
constexpr size_t kIdxMean = 4;
constexpr size_t kIdxVariance = 5;
constexpr size_t kBatchNormInputNum = 6;

bool CheckBatchNorm(const FuncGraphPtr &graph, const CNodePtr &batchnorm) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnorm);
  if (common::AnfAlgo::GetInputTensorNum(batchnorm) < kBnInputTensorNum) {
    MS_LOG(DEBUG) << "BatchNorm's input number less than " << kBnInputTensorNum;
    return false;
  }
  if (!common::AnfAlgo::HasNodeAttr(kAttrIsTraining, batchnorm)) {
    return false;
  }
  auto is_training = common::AnfAlgo::GetNodeAttr<bool>(batchnorm, kAttrIsTraining);
  if (is_training) {
    MS_LOG(DEBUG) << "Attr 'is_training' is true, no need do fusion";
    return false;
  }
  return true;
}
}  // namespace

CNodePtr BatchNorm2BNInfer::CreateBNInfer(const FuncGraphPtr &graph, const CNodePtr &batchnorm) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnorm);
  auto prim = std::make_shared<Primitive>(kBNInferOpName);

  // Format attr is needed in BatchNormInfer.
  auto origin_prim = common::AnfAlgo::GetCNodePrimitive(batchnorm);
  if (origin_prim->HasAttr(kAttrFormat)) {
    auto format = origin_prim->GetAttr(kAttrFormat);
    prim->AddAttr(kAttrFormat, MakeValue(format));
  }

  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  for (size_t i = 1; i < batchnorm->size(); ++i) {
    inputs.push_back(batchnorm->input(i));
  }
  auto new_node = NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(batchnorm->scope());
  auto old_abs = batchnorm->abstract();
  MS_EXCEPTION_IF_NULL(old_abs);
  auto old_abs_list = old_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(old_abs_list);
  if (old_abs_list->elements().size() == 0) {
    MS_LOG(EXCEPTION) << "BatchNorm's output abstract size is 0";
  }
  new_node->set_abstract(old_abs_list->elements()[0]);
  common::AnfAlgo::CopyNodeAttr(kAttrIsTraining, batchnorm, new_node);
  common::AnfAlgo::CopyNodeAttr(kAttrEpsilon, batchnorm, new_node);
  return new_node;
}

const BaseRef BatchNorm2BNInfer::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Y = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Y);
  return VectorRef({prim::kPrimBatchNorm, Xs});
}

const AnfNodePtr BatchNorm2BNInfer::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!CheckBatchNorm(graph, cnode)) {
    return nullptr;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto bn_infer = CreateBNInfer(graph, cnode);
  TransferDependOrUpdateState(cnode, graph, bn_infer);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->is_from_single_op()) {
    const auto ori_inputs = cnode->inputs();
    if (ori_inputs.size() < kBatchNormInputNum) {
      MS_LOG(EXCEPTION) << "BatchNorm's inputs size is less than 5.";
    }
    auto mean = CreateTensorMoveOp(graph, ori_inputs[kIdxMean]);
    auto variance = CreateTensorMoveOp(graph, ori_inputs[kIdxVariance]);
    auto scale = CreateTensorMoveOp(graph, ori_inputs[kIdxScale]);
    auto bias = CreateTensorMoveOp(graph, ori_inputs[kIdxBias]);
    std::vector<AnfNodePtr> make_tuple_inputs = {
      NewValueNode(prim::kPrimMakeTuple), bn_infer, mean, variance, scale, bias};
    auto make_tuple = graph->NewCNode(make_tuple_inputs);
    return make_tuple;
  } else {
    if (IsUsedByOthers(graph, cnode)) {
      return nullptr;
    } else {
      std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), bn_infer};
      auto make_tuple = graph->NewCNode(make_tuple_inputs);
      return make_tuple;
    }
  }
}
}  // namespace opt
}  // namespace mindspore
