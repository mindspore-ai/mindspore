/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_SLICE_TO_TUPLE_H
#define MINDSPORE_SLICE_TO_TUPLE_H
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <map>

#include "frontend/optimizer/optimizer_caller.h"
#include "primitive/structure_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimSliceGetItem, S, "start"} => {prim::kPrimTupleGetItem, S, 0}
// {prim::kPrimSliceGetItem, S, "stop"} => {prim::kPrimTupleGetItem, S, 1}
// {prim::kPrimSliceGetItem, S, "step"} => {prim::kPrimTupleGetItem, S, 2}
// {prim::kPrimMakeSlice, {X, Y, Z}} => {prim::kPrimMakeTuple, {X, Y, Z}}
class SliceToTuple : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (IsPrimitiveCNode(node, prim::kPrimMakeSlice)) {
      auto make_slice = node->cast<CNodePtr>();
      auto make_tuple_inputs = std::vector<AnfNodePtr>{NewValueNode(prim::kPrimMakeTuple)};
      std::copy(make_slice->inputs().cbegin() + 1, make_slice->inputs().cend(), std::back_inserter(make_tuple_inputs));
      MS_EXCEPTION_IF_NULL(make_slice->func_graph());
      return make_slice->func_graph()->NewCNode(make_tuple_inputs);
    }
    if (IsPrimitiveCNode(node, prim::kPrimSliceGetItem)) {
      auto slice_getitem = node->cast<CNodePtr>();
      auto slice_getitem_slice_input = slice_getitem->input(1);
      auto slice_getitem_item_input = slice_getitem->input(2);
      if (!IsValueNode<StringImm>(slice_getitem_item_input)) {
        return nullptr;
      }
      auto vnode = slice_getitem_item_input->cast<ValueNodePtr>();
      auto slice_attr = GetValue<std::string>(vnode->value());
      static const std::map<std::string, size_t> kSliceAttrToStaticIndex = {
        {kSliceStart, 0}, {kSliceStop, 1}, {kSliceStep, 2}};
      auto iter = kSliceAttrToStaticIndex.find(slice_attr);
      if (iter == kSliceAttrToStaticIndex.end()) {
        MS_EXCEPTION_WITH_NODE(ValueError, node) << "The slice must be [start, stop, step], but got " << slice_attr;
      }
      auto getitem_tuple_inputs =
        std::vector<AnfNodePtr>{NewValueNode(prim::kPrimTupleGetItem), slice_getitem_slice_input,
                                NewValueNode(MakeValue<int64_t>(iter->second))};
      MS_EXCEPTION_IF_NULL(slice_getitem->func_graph());
      return slice_getitem->func_graph()->NewCNode(getitem_tuple_inputs);
    }
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_SLICE_TO_TUPLE_H
