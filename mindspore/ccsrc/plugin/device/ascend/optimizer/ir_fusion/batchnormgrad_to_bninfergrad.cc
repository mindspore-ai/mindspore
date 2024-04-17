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
#include "plugin/device/ascend/optimizer/ir_fusion/batchnormgrad_to_bninfergrad.h"

#include <memory>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "ir/primitive.h"
#include "mindapi/base/types.h"
#include "ops/nn_ops.h"
#include "ops/op_utils.h"
#include "ops/sequence_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kIdxGrads = 1;
constexpr size_t kIdxScale = 3;
constexpr size_t kIdxVariance = 5;
constexpr size_t kIdxIsTraining = 7;
constexpr size_t kIdxEpsilon = 8;

template <typename T>
std::optional<T> GetScalarAnfNodeValue(const AnfNodePtr &anf_node) {
  if (!anf_node->isa<ValueNode>()) {
    return std::nullopt;
  }
  auto value_node = anf_node->cast<ValueNodePtr>();
  auto value_opt = mindspore::ops::GetScalarValue<T>(value_node->value());
  if (!value_opt.has_value()) {
    return std::nullopt;
  }
  return value_opt.value();
}

CNodePtr CreateBNInferGrad(const FuncGraphPtr &graph, const CNodePtr &batchnormgrad, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnormgrad);
  MS_EXCEPTION_IF_NULL(node);
  auto prim = std::make_shared<Primitive>(kBNInferGradOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  inputs.push_back(batchnormgrad->input(kIdxGrads));
  inputs.push_back(batchnormgrad->input(kIdxScale));
  inputs.push_back(batchnormgrad->input(kIdxVariance));
  inputs.push_back(batchnormgrad->input(kIdxEpsilon));
  auto new_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(batchnormgrad->scope());
  new_node->set_abstract(node->abstract());

  auto is_training_opt = GetScalarAnfNodeValue<bool>(batchnormgrad->input(kIdxIsTraining));
  if (is_training_opt.has_value()) {
    auto is_training = is_training_opt.value();
    common::AnfAlgo::SetNodeAttr(kAttrIsTraining, MakeValue(is_training), new_node);
  } else {
    MS_LOG(ERROR) << "For BNInferGrad pass, failed to get attr is_training.";
  }

  auto epsilon_opt = GetScalarAnfNodeValue<pyfloat>(batchnormgrad->input(kIdxEpsilon));
  float epsilon{1e-5};
  if (epsilon_opt.has_value()) {
    epsilon = epsilon_opt.has_value() ? epsilon_opt.value() : 1e-5;
  } else {
    MS_LOG(ERROR) << "For BNInferGrad pass, failed to get attr epsilon, use default epsilon: 1e-5.";
  }
  common::AnfAlgo::SetNodeAttr(kAttrEpsilon, MakeValue(epsilon), new_node);

  return new_node;
}

bool CheckIndex(const AnfNodePtr &index_node) {
  MS_EXCEPTION_IF_NULL(index_node);
  if (!IsValueNode<Int64Imm>(index_node)) {
    return false;
  }
  ValueNodePtr value_node = index_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  int64_t index = GetValue<int64_t>(value_node->value());
  if (index != 0) {
    MS_LOG(DEBUG) << "tuple_getitem must be 0th output of BatchNormGrad";
    return false;
  }
  return true;
}

bool CheckBatchNormGrad(const FuncGraphPtr &graph, const CNodePtr &batchnormgrad) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnormgrad);
  if (common::AnfAlgo::GetInputTensorNum(batchnormgrad) < kBNGradInputTensorNum) {
    MS_LOG(DEBUG) << "BatchNormGrad's input number less than " << kBnInputTensorNum;
    return false;
  }

  auto is_training_opt = GetScalarAnfNodeValue<bool>(batchnormgrad->input(kIdxIsTraining));
  if (!is_training_opt.has_value()) {
    return false;
  }

  auto is_training = is_training_opt.value();
  if (is_training) {
    MS_LOG(DEBUG) << "Attr 'is_training' is true, no need do fusion";
    return false;
  }

  if (IsUsedByOthers(graph, batchnormgrad)) {
    MS_LOG(DEBUG) << "Only the 0th output of BatchNormGrad is used, then do fusion";
    return false;
  }

  return true;
}

bool NeedFusion(const FuncGraphPtr &graph, const AnfNodePtr &node, CNodePtr *batchnorm_grad) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto tuple_getitem = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  CheckCNodeInputSize(tuple_getitem, kTupleGetItemInputTensorNum);
  AnfNodePtr index_node = tuple_getitem->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_node);
  if (!CheckIndex(index_node)) {
    return false;
  }

  AnfNodePtr batchnorm_grad_anf = tuple_getitem->input(kRealInputNodeIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(batchnorm_grad_anf);
  MS_EXCEPTION_IF_NULL(batchnorm_grad);
  *batchnorm_grad = batchnorm_grad_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(*batchnorm_grad);
  return CheckBatchNormGrad(graph, *batchnorm_grad);
}
}  // namespace

std::vector<std::string> BatchNormGrad2BNInferGrad::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimBatchNormGrad->name());
  return ret;
}

const BaseRef BatchNormGrad2BNInferGrad::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Y = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Y);
  VectorRef batchnormgrad({prim::kPrimBatchNormGrad, Xs});
  VectorRef pattern({prim::kPrimTupleGetItem, batchnormgrad, Y});
  return pattern;
}

const AnfNodePtr BatchNormGrad2BNInferGrad::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr batchnorm_grad = nullptr;
  if (!NeedFusion(graph, node, &batchnorm_grad)) {
    return nullptr;
  }
  auto bn_infer_grad = CreateBNInferGrad(graph, batchnorm_grad, node);
  TransferDependOrUpdateState(batchnorm_grad, graph, bn_infer_grad);
  return bn_infer_grad;
}
}  // namespace opt
}  // namespace mindspore
