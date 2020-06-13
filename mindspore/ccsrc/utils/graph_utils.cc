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

#include "utils/graph_utils.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <stack>
#include <vector>
#include <list>
#include <string>
#include <fstream>

#include "ir/visitor.h"
#include "ir/manager.h"
#include "utils/log_adapter.h"
#include "common/utils.h"
#include "pipeline/parse/function_block.h"
#include "pipeline/parse/python_adapter.h"

namespace mindspore {
using SymbolicKeyTypePtr = std::shared_ptr<SymbolicKeyType>;

namespace {
class DeepFirstSearcher : public AnfVisitor {
 public:
  explicit DeepFirstSearcher(const IncludeFunc &include) : include_(include) {}
  ~DeepFirstSearcher() override = default;

  std::vector<AnfNodePtr> Search(const AnfNodePtr &root) {
    if (root == nullptr) {
      return res_;
    }
    seen_ = NewSeenGeneration();
    Visit(root);
    return res_;
  }

  void Visit(const AnfNodePtr &node) override {
    MS_EXCEPTION_IF_NULL(node);
    if (node->seen_ == seen_) {
      return;
    }

    node->seen_ = seen_;

    auto incl = include_(node);
    if (incl == EXCLUDE) {
      return;
    }

    res_.push_back(node);
    if (incl == FOLLOW) {
      AnfVisitor::Visit(node);
    }
  }

 private:
  size_t seen_{0};
  IncludeFunc include_;
  std::vector<AnfNodePtr> res_{};
};

class DeepScopedGraphSearcher : public DeepFirstSearcher {
 public:
  explicit DeepScopedGraphSearcher(const IncludeFunc &include) : DeepFirstSearcher(include) {}
  ~DeepScopedGraphSearcher() override = default;

  void Visit(const CNodePtr &cnode) override {
    if (cnode->func_graph() == nullptr) {
      return;
    }

    AnfNodePtr ret = cnode->func_graph()->get_return();
    if (ret != nullptr) {
      DeepFirstSearcher::Visit(ret);
    }

    auto &inputs = cnode->inputs();
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      DeepFirstSearcher::Visit(*iter);
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (!IsValueNode<FuncGraph>(vnode)) {
      return;
    }

    auto graph = GetValueNode<FuncGraphPtr>(vnode);
    AnfNodePtr ret = graph->get_return();
    if (ret != nullptr) {
      DeepFirstSearcher::Visit(ret);
    }
  }

  void Visit(const ParameterPtr &param) override {
    if (param->func_graph() == nullptr) {
      return;
    }

    AnfNodePtr ret = param->func_graph()->get_return();
    if (ret != nullptr) {
      DeepFirstSearcher::Visit(ret);
    }
  }
};

class DeepUsedGraphSearcher : public DeepFirstSearcher {
 public:
  explicit DeepUsedGraphSearcher(const IncludeFunc &include) : DeepFirstSearcher(include) {}
  ~DeepUsedGraphSearcher() override = default;

  void Visit(const CNodePtr &cnode) override {
    auto &inputs = cnode->inputs();
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      DeepFirstSearcher::Visit(*iter);
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (!IsValueNode<FuncGraph>(vnode)) {
      return;
    }

    auto graph = GetValueNode<FuncGraphPtr>(vnode);
    AnfNodePtr ret = graph->get_return();
    if (ret != nullptr) {
      DeepFirstSearcher::Visit(ret);
    }
  }
};

class DeepLinkedGraphSearcher : public DeepFirstSearcher {
 public:
  explicit DeepLinkedGraphSearcher(const IncludeFunc &include) : DeepFirstSearcher(include) {}
  ~DeepLinkedGraphSearcher() override = default;

  void Visit(const CNodePtr &cnode) override {
    auto &inputs = cnode->inputs();
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      DeepFirstSearcher::Visit(*iter);
    }
  }

  void Visit(const ValueNodePtr &) override {}
};

class DeepUsersSearcher : public DeepFirstSearcher {
 public:
  explicit DeepUsersSearcher(const IncludeFunc &include, const FuncGraphManagerPtr &mng)
      : DeepFirstSearcher(include), mng_(mng) {}
  ~DeepUsersSearcher() override = default;

  void Visit(const CNodePtr &cnode) override {
    auto &users = mng_->node_users()[cnode];
    for (auto iter = users.begin(); iter != users.end(); ++iter) {
      DeepFirstSearcher::Visit(iter->first);
    }
  }
  void Visit(const ValueNodePtr &) override {}

 private:
  FuncGraphManagerPtr mng_;
};
}  // namespace

std::vector<AnfNodePtr> DeepScopedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include) {
  return DeepScopedGraphSearcher(include).Search(root);
}

std::vector<AnfNodePtr> DeepUsedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include) {
  return DeepUsedGraphSearcher(include).Search(root);
}

std::vector<AnfNodePtr> DeepLinkedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include) {
  return DeepLinkedGraphSearcher(include).Search(root);
}

std::vector<AnfNodePtr> DeepUsersSearch(const AnfNodePtr &root, const IncludeFunc &include,
                                        const FuncGraphManagerPtr &mng) {
  return DeepUsersSearcher(include, mng).Search(root);
}

std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &root, const SuccFunc &succ, const IncludeFunc &include) {
  size_t seen = NewSeenGeneration();
  std::list<AnfNodePtr> todo(1, root);
  std::unordered_map<AnfNodePtr, size_t> rank;
  std::vector<AnfNodePtr> res;

  while (!todo.empty()) {
    AnfNodePtr node = todo.back();
    if (node == nullptr || node->seen_ == seen) {
      todo.pop_back();
      continue;
    }
    if (rank.find(node) != rank.end() && rank[node] != todo.size()) {
      MS_LOG(EXCEPTION) << "Graph exists cycle, node " << node->DebugString();
    }
    rank[node] = todo.size();
    bool cont = false;
    auto incl = include(node);
    if (incl == FOLLOW) {
      auto succs = succ(node);
      for (const auto i : succs) {
        if ((i != nullptr && i->seen_ != seen)
            // Handle the case for 2 subgraphs calls each other.
            // If the ValueNodeGraph's return is already in the todo list, do not follow it.
            && !((std::find(todo.begin(), todo.end(), i) != todo.end()) && (i->func_graph() != nullptr) &&
                 (i->func_graph()->get_return() == i))) {
          todo.push_back(i);
          cont = true;
        }
      }
    } else if (incl == NOFOLLOW) {
      // do nothing
    } else if (incl == EXCLUDE) {
      node->seen_ = seen;
      todo.pop_back();
      continue;
    } else {
      MS_LOG(EXCEPTION) << "include(node) must return one of: \"follow\", \"nofollow\", \"exclude\"";
    }
    if (cont) {
      continue;
    }
    node->seen_ = seen;
    res.push_back(node);
    todo.pop_back();
  }
  return res;
}

std::vector<AnfNodePtr> SuccDeeper(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  if (IsValueNode<FuncGraph>(node)) {
    auto graph = GetValueNode<FuncGraphPtr>(node);
    auto ret = graph->get_return();
    if (ret != nullptr) {
      vecs.push_back(ret);
    }
    return vecs;
  } else if (node->func_graph() != nullptr) {
    if (node->isa<CNode>()) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
    }
    auto graph = node->func_graph();
    if (graph->get_return() != nullptr) {
      vecs.push_back(graph->get_return());
    }
    return vecs;
  }

  return vecs;
}

std::vector<AnfNodePtr> SuccDeeperSimple(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  if (IsValueNode<FuncGraph>(node)) {
    auto graph = GetValueNode<FuncGraphPtr>(node);
    auto ret = graph->get_return();
    if (ret != nullptr) {
      vecs.push_back(ret);
    }
    return vecs;
  } else {
    if (node->isa<CNode>()) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
    }
    return vecs;
  }
}

std::vector<AnfNodePtr> SuccIncoming(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }

  if (node->isa<CNode>()) {
    auto &inputs = node->cast<CNodePtr>()->inputs();
    (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
  }
  return vecs;
}

std::vector<AnfNodePtr> SuccIncludeFV(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  std::vector<AnfNodePtr> vecs;
  if (node == nullptr) {
    return vecs;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto &inputs = cnode->inputs();
    // Check if free variables used.
    for (const auto &input : inputs) {
      auto input_fg = GetValueNode<FuncGraphPtr>(input);
      if (input_fg) {
        for (auto &fv : input_fg->free_variables_nodes()) {
          if (fv->func_graph() == fg && fg->nodes().contains(fv)) {
            vecs.push_back(fv);
          }
        }
      }
    }
    (void)vecs.insert(vecs.end(), inputs.begin(), inputs.end());
  }
  return vecs;
}

IncludeType AlwaysInclude(const AnfNodePtr &) { return FOLLOW; }

IncludeType IncludeBelongGraph(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  if (node->func_graph() == fg) {
    return FOLLOW;
  } else {
    return EXCLUDE;
  }
}

FuncGraphIndex::FuncGraphIndex(const FuncGraphPtr &fg, const SearchFunc &search, const IncludeFunc &include) {
  MS_EXCEPTION_IF_NULL(fg);
  Acquire(fg);

  auto vec = search(fg->get_return(), include);
  for (auto &node : vec) {
    MS_EXCEPTION_IF_NULL(node);
    Acquire(node);
    if (node->func_graph() != nullptr) {
      Acquire(node->func_graph());
    }
  }
}

std::set<FuncGraphPtr> FuncGraphIndex::GetFuncGraphs(const std::string &key) {
  std::set<FuncGraphPtr> func_graphs;
  if (index_func_graph_.find(key) != index_func_graph_.end()) {
    func_graphs = index_func_graph_[key];
  }
  return func_graphs;
}

std::set<AnfNodePtr> FuncGraphIndex::GetNodes(const std::string &key) {
  if (index_node_.find(key) != index_node_.end()) {
    return index_node_[key];
  }

  return std::set<AnfNodePtr>();
}

FuncGraphPtr FuncGraphIndex::GetFirstFuncGraph(const std::string &key) {
  if (GetFuncGraphs(key).empty()) {
    return nullptr;
  }

  auto fg = *GetFuncGraphs(key).begin();
  return fg;
}

AnfNodePtr FuncGraphIndex::GetFirstNode(const std::string &key) {
  if (GetNodes(key).empty()) {
    return nullptr;
  }

  auto node = *GetNodes(key).begin();
  return node;
}

void FuncGraphIndex::Acquire(const FuncGraphPtr &key) {
  std::string name = label_manage::Label(key->debug_info());
  if (!name.empty()) {
    (void)index_func_graph_[name].insert(key);
  }
}

void FuncGraphIndex::Acquire(const AnfNodePtr &key) {
  std::string name = label_manage::Label(key->debug_info());
  if (!name.empty()) {
    (void)index_node_[name].insert(key);
  }
}

// Isomorphism
static bool SameNodeShallow(const AnfNodePtr &node1, const AnfNodePtr &node2, FuncGraphPairMapEquiv *equiv_func_graph,
                            NodeMapEquiv *const equiv_node) {
  if (equiv_node == nullptr) {
    MS_LOG(ERROR) << "Invalid equiv_node";
    return false;
  }
  if (equiv_node->count(node1) > 0 && (*equiv_node)[node1] == node2) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node1) && IsValueNode<FuncGraph>(node2)) {
    return Isomorphic(GetValueNode<FuncGraphPtr>(node1), GetValueNode<FuncGraphPtr>(node2), equiv_func_graph,
                      equiv_node);
  }
  if (node1->isa<ValueNode>() && node2->isa<ValueNode>()) {
    auto a1 = GetValueNode(node1);
    auto a2 = GetValueNode(node2);
    if (a1->isa<Primitive>() && a2->isa<Primitive>()) {
      return a1->cast<PrimitivePtr>()->name() == a2->cast<PrimitivePtr>()->name();
    } else if (a1->isa<tensor::Tensor>() && a2->isa<tensor::Tensor>()) {
      return a1->cast<tensor::TensorPtr>()->ValueEqual(*(a2->cast<tensor::TensorPtr>()));
    } else {
      return *a1 == *a2;
    }
  }
  if (node1->isa<Parameter>() && node2->isa<Parameter>()) {
    auto para1 = node1->cast<ParameterPtr>();
    auto para2 = node2->cast<ParameterPtr>();
    if (para1->name() == para2->name()) {
      return true;
    }
    MS_LOG(DEBUG) << "two parameters are not equal.";
    return false;
  }
  MS_LOG(ERROR) << "type error";
  return false;
}

static bool SameNode(const AnfNodePtr &node1, const AnfNodePtr &node2, FuncGraphPairMapEquiv *equiv_func_graph,
                     NodeMapEquiv *const equiv_node) {
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  if (node1->isa<CNode>() && node2->isa<CNode>()) {
    auto &inputs1 = node1->cast<CNodePtr>()->inputs();
    auto &inputs2 = node2->cast<CNodePtr>()->inputs();
    for (std::size_t i = 0; i < inputs1.size(); ++i) {
      if (!SameNodeShallow(inputs1[i], inputs2[i], equiv_func_graph, equiv_node)) {
        return false;
      }
    }
    return true;
  }
  return SameNodeShallow(node1, node2, equiv_func_graph, equiv_node);
}

static bool SameSubgraph(AnfNodePtr root1, AnfNodePtr root2, FuncGraphPairMapEquiv *equiv_func_graph,
                         NodeMapEquiv *const equiv_node) {
  std::unordered_set<AnfNodePtr> done;
  std::stack<std::pair<AnfNodePtr, AnfNodePtr>> todo;

  todo.push(std::make_pair(root1, root2));
  while (todo.size() > 0) {
    AnfNodePtr node1 = todo.top().first;
    if (done.count(node1) > 0) {
      todo.pop();
      continue;
    }
    AnfNodePtr node2 = todo.top().second;

    bool condition = false;
    std::vector<AnfNodePtr> s1 = SuccIncoming(node1);
    std::vector<AnfNodePtr> s2 = SuccIncoming(node2);

    if (s1.size() != s2.size()) {
      return false;
    }
    for (std::size_t i = 0; i < s1.size(); ++i) {
      if (done.count(s1[i]) == 0) {
        todo.push(std::make_pair(s1[i], s2[i]));
        condition = true;
      }
    }
    if (condition) {
      continue;
    }
    (void)done.insert(node1);

    auto res = SameNode(node1, node2, equiv_func_graph, equiv_node);
    if (res) {
      (*equiv_node)[node1] = node2;
    } else {
      return false;
    }
    todo.pop();
  }
  return true;
}

bool Isomorphic(FuncGraphPtr fg1, FuncGraphPtr fg2, FuncGraphPairMapEquiv *equiv_func_graph,
                NodeMapEquiv *const equiv_node) {
  auto fg1_fg2 = std::make_pair(fg1, fg2);
  if (equiv_func_graph == nullptr) {
    MS_LOG(ERROR) << "equiv_func_graph not init";
    return false;
  }
  if (equiv_func_graph->find(fg1_fg2) != equiv_func_graph->end()) {
    return (*equiv_func_graph)[fg1_fg2] != kNotEquiv;
  }
  if (fg1 == nullptr || fg2 == nullptr) {
    MS_LOG(ERROR) << "Invalid function graph";
    return false;
  }
  if (fg1->parameters().size() != fg2->parameters().size()) {
    MS_LOG(DEBUG) << "parameters size not match";
    return false;
  }
  if (equiv_node != nullptr) {
    for (std::size_t i = 0; i < fg1->parameters().size(); ++i) {
      (*equiv_node)[fg1->parameters()[i]] = fg2->parameters()[i];
    }
    (*equiv_func_graph)[fg1_fg2] = kPending;
    auto result = SameSubgraph(fg1->get_return(), fg2->get_return(), equiv_func_graph, equiv_node);
    (*equiv_func_graph)[fg1_fg2] = EquivState(result);
    return result;
  }

  MS_LOG(ERROR) << "equiv_node not init";
  return false;
}

tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    MS_EXCEPTION(ArgumentError) << "Nullptr Error!";
  }
  tensor::TensorPtr tensor = nullptr;
  if (scalar->isa<FloatImm>()) {
    tensor = std::make_shared<tensor::Tensor>(py::float_(GetValue<float>(scalar)), kFloat32);
  } else if (scalar->isa<IntergerImm>()) {
    tensor = std::make_shared<tensor::Tensor>(py::int_(GetValue<int>(scalar)), kInt32);
  } else if (scalar->isa<BoolImm>()) {
    tensor = std::make_shared<tensor::Tensor>(py::array(py::bool_(GetValue<bool>(scalar))), kBool);
  } else {
    auto type = scalar->type();
    auto type_str = (type == nullptr) ? "nullptr" : type->ToString();
    MS_LOG(EXCEPTION) << "Invalid scalar type: " << type_str;
  }
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor;
}
}  // namespace mindspore
