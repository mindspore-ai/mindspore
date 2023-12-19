/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/scalar_ops_output_unify_mindir.h"
#include <memory>
#include <vector>
#include "ops/array_ops.h"
#include "ops/other_ops.h"
#include "ops/arithmetic_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
const std::vector<PrimitivePtr> scalar_ops = {
  prim::kPrimScalarCast, prim::kPrimScalarTrunc, prim::kPrimScalarAdd, prim::kPrimScalarMul, prim::kPrimScalarSub,
  prim::kPrimScalarDiv,  prim::kPrimScalarPow,   prim::kPrimScalarExp, prim::kPrimScalarGt,  prim::kPrimScalarUadd,
  prim::kPrimScalarUsub, prim::kPrimScalarLt,    prim::kPrimScalarGe,  prim::kPrimScalarLe};
bool IsScalarOp(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    return std::any_of(scalar_ops.begin(), scalar_ops.end(),
                       [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  }
  return false;
}
}  // namespace

const BaseRef ScalarOpsOutputUnifyMindIR::DefinePattern() const {
  VarPtr resize = std::make_shared<CondVar>(IsScalarOp);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({resize, inputs});
}

const AnfNodePtr ScalarOpsOutputUnifyMindIR::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  // attr dtype
  auto data_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  // update abstract
  auto abs = abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector{}), TypeIdToType(data_type));
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for " << node->fullname_with_scope() << " op is " << abs->ToString();
  node->set_abstract(abs);
  return node;
}
}  // namespace opt
}  // namespace mindspore
