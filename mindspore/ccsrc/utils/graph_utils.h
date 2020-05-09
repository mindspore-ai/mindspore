/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_UTILS_GRAPH_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_GRAPH_UTILS_H_

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
#include "ir/meta_tensor.h"
#include "debug/label.h"

namespace mindspore {

enum IncludeType { FOLLOW, NOFOLLOW, EXCLUDE };

using IncludeFunc = std::function<IncludeType(const AnfNodePtr &)>;
using SuccFunc = std::function<std::vector<AnfNodePtr>(AnfNodePtr)>;
using SearchFunc = std::function<std::vector<AnfNodePtr>(const AnfNodePtr &, const IncludeFunc &)>;

std::vector<AnfNodePtr> SuccDeeper(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccDeeperSimple(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccIncoming(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccIncludeFV(const FuncGraphPtr &fg, const AnfNodePtr &node);

IncludeType AlwaysInclude(const AnfNodePtr &node);
IncludeType IncludeBelongGraph(const FuncGraphPtr &fg, const AnfNodePtr &node);

std::vector<AnfNodePtr> DeepScopedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);
std::vector<AnfNodePtr> DeepUsedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);
std::vector<AnfNodePtr> DeepLinkedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);

std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &root, const SuccFunc &succ = SuccIncoming,
                                 const IncludeFunc &include = AlwaysInclude);

std::vector<CNodePtr> BroadFirstSearchGraphCNodes(CNodePtr ret);
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

// Isomorphism

struct PairHasher {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

enum EquivState { kNotEquiv = 0, kEquiv = 1, kPending = 2 };

using FuncGraphPairMapEquiv = std::unordered_map<std::pair<FuncGraphPtr, FuncGraphPtr>, EquivState, PairHasher>;
using NodeMapEquiv = std::unordered_map<AnfNodePtr, AnfNodePtr>;

bool Isomorphic(FuncGraphPtr g1, FuncGraphPtr g2, FuncGraphPairMapEquiv *equiv_func_graph, NodeMapEquiv *equiv_node);

tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_GRAPH_UTILS_H_
