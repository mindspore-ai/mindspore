/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/pass/getitem_tuple.h"

#include <memory>
#include "base/core_ops.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
bool IsC(const BaseRef &n) {
  MS_EXCEPTION_IF_NULL(n);
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    return in->isa<ValueNode>();
  } else {
    return false;
  }
}
}  // namespace

const BaseRef GetitemTuple::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr C = std::make_shared<CondVar>(IsC);
  return VectorRef({prim::kPrimTupleGetItem, VectorRef({prim::kPrimMakeTuple, Xs}), C});
}

const AnfNodePtr GetitemTuple::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr tuple_getitem = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  CheckCNodeInputSize(tuple_getitem, kTupleGetItemInputTensorNum);
  AnfNodePtr make_tuple_anf = tuple_getitem->input(kRealInputNodeIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(make_tuple_anf);
  AnfNodePtr index_node = tuple_getitem->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_node);
  if (IsValueNode<Int64Imm>(index_node)) {
    ValueNodePtr value_node = index_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto index = GetValue<int64_t>(value_node->value());
    CNodePtr make_tuple = make_tuple_anf->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    if (make_tuple->inputs().size() > LongToSize(index + 1)) {
      auto ret = make_tuple->input(LongToSize(index + 1));
      MS_EXCEPTION_IF_NULL(ret);
      return ret;
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
