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
#include "tools/optimizer/fusion/conv_tuplegetitem_fusion.h"
#include <memory>
#include "src/param_value_lite.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kTupleGetItemLen = 3;
bool IsTupleGetItemNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem);
  }
  return false;
}
}  // namespace

const BaseRef ConvTupleGetItemFusion::DefinePattern() const {
  auto tuple_var = std::make_shared<CondVar>(IsTupleGetItemNode);
  auto tuple_index = std::make_shared<Var>();
  auto conv_var = std::make_shared<CondVar>(IsConvNode);
  return VectorRef({tuple_var, conv_var, tuple_index});
}

const AnfNodePtr ConvTupleGetItemFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "conv_tuplegetitem_fusion pass";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto tuple_cnode = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(tuple_cnode) != lite::RET_OK ||
      CheckInputSize(tuple_cnode, kTupleGetItemLen) != lite::RET_OK) {
    return nullptr;
  }
  auto idx = GetTupleGetItemOutIndex(tuple_cnode);
  if (idx != 0) {
    MS_LOG(DEBUG) << "TupleGetItem's idx is not 0";
    return nullptr;
  }
  auto conv_node = tuple_cnode->input(1);
  if (CheckIfAnfNodeIsNull(conv_node) != lite::RET_OK) {
    return nullptr;
  }
  auto conv_cnode = conv_node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(conv_cnode) != lite::RET_OK) {
    return nullptr;
  }
  auto abstr = conv_cnode->abstract();
  if (utils::isa<abstract::AbstractTuplePtr>(abstr)) {
    auto elements = utils::cast<abstract::AbstractTuplePtr>(abstr)->elements();
    if (elements.empty()) {
      MS_LOG(ERROR) << "AbstractTuple is empty";
      return nullptr;
    }
    conv_node->set_abstract(elements[0]);
  }
  return conv_node;
}
}  // namespace mindspore::opt
