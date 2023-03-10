/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_LINK_CUSTOM_OP_H
#define MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_LINK_CUSTOM_OP_H

#include <set>
#include <utility>
#include "ir/anf.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore::opt::dynamic_shape {
using DependPair = std::pair<AnfNodePtr, AnfNodePtr>;
struct DependPairCmp {
  bool operator()(const DependPair &lhs, const DependPair &rhs) const {
    if (lhs.first != rhs.first) {
      return lhs.first > rhs.first;
    }
    return lhs.second > rhs.second;
  }
};

class LinkCustomOp : public Pass {
 public:
  LinkCustomOp() : Pass("link_custom_op") {}
  ~LinkCustomOp() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void InsertDepend(const FuncGraphPtr &g, const AnfNodePtr &prev, const AnfNodePtr &next,
                    AnfNodePtrList *depend_nodes);
  bool LinkInternalOp(const FuncGraphPtr &g, const AnfNodePtr &node, AnfNodePtrList *depend_nodes);
  bool LinkInputOp(const FuncGraphPtr &g, const CNodePtr &cnode, AnfNodePtrList *depend_nodes);
  bool LinkDependSync(const FuncGraphPtr &g, const CNodePtr &cnode, AnfNodePtrList *depend_nodes);
  void AttachDependNodes(const FuncGraphPtr &g, const AnfNodePtrList &depend_nodes) const;

  std::set<DependPair, DependPairCmp> added_set_;
};
}  // namespace mindspore::opt::dynamic_shape
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_LINK_CUSTOM_OP_H
