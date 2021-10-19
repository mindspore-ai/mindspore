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
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
const BaseRef ConvTupleGetItemFusion::DefinePattern() const {
  auto is_tuple_getitem = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>);
  MS_CHECK_TRUE_RET(is_tuple_getitem != nullptr, {});
  auto is_conv = std::make_shared<CondVar>(IsConvNode);
  MS_CHECK_TRUE_RET(is_conv != nullptr, {});
  auto is_var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(is_var != nullptr, {});
  return VectorRef({is_tuple_getitem, is_conv, is_var});
}

const AnfNodePtr ConvTupleGetItemFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  auto tuple_cnode = node->cast<CNodePtr>();
  if (tuple_cnode == nullptr || tuple_cnode->size() != kInputSizeThree) {
    return nullptr;
  }
  if (IsMarkedTrainOp(tuple_cnode)) {
    return nullptr;
  }
  auto idx = GetTupleGetItemOutIndex(tuple_cnode);
  if (idx != 0) {
    MS_LOG(DEBUG) << "TupleGetItem's idx is not 0";
    return nullptr;
  }
  auto conv_node = tuple_cnode->input(1);
  if (conv_node == nullptr) {
    return nullptr;
  }
  auto conv_cnode = conv_node->cast<CNodePtr>();
  if (conv_cnode == nullptr) {
    return nullptr;
  }
  if (IsMarkedTrainOp(conv_cnode)) {
    return nullptr;
  }
  auto abstr = conv_cnode->abstract();
  if (abstr != nullptr && utils::isa<abstract::AbstractTuplePtr>(abstr)) {
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
