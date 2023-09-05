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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MAKE_LIST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_MAKE_LIST_PASS_H_
#include <set>
#include "include/registry/converter_context.h"
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
class MakeListPass : public Pass {
 public:
  MakeListPass() : Pass("make_list_pass") {}
  ~MakeListPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  STATUS UpdateMakeListAbstracts(const FuncGraphPtr &func_graph);
  AbstractBasePtr ConvertToAbstractTuple(const AbstractBasePtr &abs, size_t depth);
  STATUS MakeListToMakeTuple(const FuncGraphPtr &func_graph);
  AnfNodePtr MakeListNodeRewrite(const AnfNodePtr &node);
  ValuePtr ConvertValueSequenceToValueTuple(const ValuePtr &value, size_t depth, bool *need_convert);
  AnfNodePtr ConvertMakeListValueNode(const ValueNodePtr &value_node, const ValuePtr &value);
  AnfNodePtr ConvertMakeListNode(const AnfNodePtr &node);
  AnfNodePtr ConvertMakeListPrimitiveCNode(const CNodePtr &cnode, const PrimitivePtr &prim);
  AnfNodePtr ConvertListSetItemToTupleSetItem(const CNodePtr &node);
  AnfNodePtr ConvertListGetItemToTupleGetItem(const CNodePtr &node);
  AnfNodePtr ConvertMakeListToMakeTuple(const CNodePtr &node);
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SCALAR_OP_PASS_H_
