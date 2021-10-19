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

#include "tools/optimizer/fusion/reshape_reshape_fusion.h"
#include "ops/op_utils.h"
#include "ops/reshape.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}  // namespace

const BaseRef ReshapeReshapeFusion::DefinePattern() const {
  reshape_input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_input_ != nullptr, {});
  reshape_shape_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(reshape_shape_ != nullptr, {});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto reshape1 = VectorRef({is_reshape1, reshape_input_, is_param});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  return VectorRef({is_reshape2, reshape1, reshape_shape_});
}

const AnfNodePtr ReshapeReshapeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  auto reshape_prim = std::make_shared<ops::Reshape>();
  if (reshape_prim == nullptr) {
    MS_LOG(ERROR) << "Build reshape primitive failed.";
    return nullptr;
  }
  auto value_node = NewValueNode(reshape_prim);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
  auto input = utils::cast<AnfNodePtr>((*equiv)[reshape_input_]);
  auto shape = utils::cast<AnfNodePtr>((*equiv)[reshape_shape_]);
  if (input == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "Cannot find reshape input and weight.";
    return nullptr;
  }
  // create scale op
  auto new_reshape = func_graph->NewCNode({value_node, input, shape});
  return new_reshape;
}
}  // namespace mindspore::opt
