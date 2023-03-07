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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CORE_SPLIT_SCHEMER_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CORE_SPLIT_SCHEMER_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/hash_map.h"

namespace mindspore::graphkernel {
class SplitSchemer {
 public:
  SplitSchemer() = default;
  virtual ~SplitSchemer() = default;
  virtual bool Split(const FuncGraphPtr &func_graph) = 0;
  virtual bool NeedInline(size_t group_id) const;
  const std::vector<AnfNodePtrList> &split_plan() const { return split_plan_; }

 protected:
  std::vector<AnfNodePtrList> split_plan_;
  std::vector<int> need_inline_;
};
using SplitSchemerPtr = std::shared_ptr<SplitSchemer>;

class CommonSplitSchemer : public SplitSchemer {
 protected:
  // add a new group
  size_t AddGroup(AnfNodePtrList &&nodes, bool need_inline);

  // group the return node and last MakeTuple node (if exists).
  void GroupReturnNode(const FuncGraphPtr &func_graph);

  mindspore::HashMap<AnfNodePtr, size_t> node_group_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CORE_SPLIT_SCHEMER_H_
