/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_GRAPH_UTILS_H_
#define MINDSPORE_CORE_IR_GRAPH_UTILS_H_

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>

#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "utils/label.h"

namespace mindspore {
enum IncludeType { FOLLOW, NOFOLLOW, EXCLUDE };

using IncludeFunc = std::function<IncludeType(const AnfNodePtr &)>;
using FilterFunc = std::function<bool(const AnfNodePtr &)>;
using SuccFunc = std::function<std::vector<AnfNodePtr>(AnfNodePtr)>;
using SearchFunc = std::function<std::vector<AnfNodePtr>(const AnfNodePtr &, const IncludeFunc &)>;
using MatchFunc = std::function<bool(const CNodePtr &)>;

std::vector<AnfNodePtr> DeepScopedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include);
std::vector<AnfNodePtr> DeepUsedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include);
std::vector<AnfNodePtr> DeepLinkedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include);

std::vector<AnfNodePtr> SuccDeeper(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccDeeperSimple(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccIncoming(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccIncludeFV(const FuncGraphPtr &fg, const AnfNodePtr &node);

const std::vector<AnfNodePtr> &GetInputs(const AnfNodePtr &node);

IncludeType AlwaysInclude(const AnfNodePtr &node);
IncludeType IncludeBelongGraph(const FuncGraphPtr &fg, const AnfNodePtr &node);

std::vector<AnfNodePtr> DeepScopedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);
std::vector<AnfNodePtr> DeepUsedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);
std::vector<AnfNodePtr> DeepLinkedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);

std::vector<AnfNodePtr> DeepScopedGraphSearchWithFilter(const AnfNodePtr &root, const IncludeFunc &include,
                                                        const FilterFunc &filter);

class FuncGraphManager;
using FuncGraphManagerPtr = std::shared_ptr<FuncGraphManager>;
std::vector<AnfNodePtr> DeepUsersSearch(const AnfNodePtr &root, const IncludeFunc &include,
                                        const FuncGraphManagerPtr &mng);
std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &root, const SuccFunc &succ = SuccIncoming,
                                 const IncludeFunc &include = AlwaysInclude);

std::vector<CNodePtr> BroadFirstSearchGraphCNodes(const std::vector<CNodePtr> &starts);
std::vector<FuncGraphPtr> BroadFirstSearchGraphUsed(FuncGraphPtr root);

CNodePtr BroadFirstSearchFirstOf(const std::vector<CNodePtr> &starts, const MatchFunc &match_predicate);

class FuncGraphIndex {
 public:
  explicit FuncGraphIndex(const FuncGraphPtr &fg, const SearchFunc &search = DeepScopedGraphSearch,
                          const IncludeFunc &include = AlwaysInclude);
  FuncGraphIndex(const FuncGraphIndex &) = delete;
  FuncGraphIndex &operator=(const FuncGraphIndex &) = delete;

  virtual ~FuncGraphIndex() {}

  std::set<FuncGraphPtr> GetFuncGraphs(const std::string &key);
  std::set<AnfNodePtr> GetNodes(const std::string &key);
  FuncGraphPtr GetFirstFuncGraph(const std::string &key);
  AnfNodePtr GetFirstNode(const std::string &key);

 private:
  void Acquire(const FuncGraphPtr &key);
  void Acquire(const AnfNodePtr &key);

  std::map<std::string, std::set<FuncGraphPtr>> index_func_graph_;
  std::map<std::string, std::set<AnfNodePtr>> index_node_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_GRAPH_UTILS_H_
