/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ir/graph_utils.h"

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
#include "ir/func_graph.h"
#include "utils/label.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace {
class DeepFirstSearcher : public AnfIrVisitor {
 public:
  explicit DeepFirstSearcher(const IncludeFunc &include, const FilterFunc &filter = nullptr)
      : include_(include), filter_(filter) {}
  ~DeepFirstSearcher() override = default;

  std::vector<AnfNodePtr> Search(const AnfNodePtr &root) {
    if (root == nullptr) {
      return std::move(res_);
    }
    seen_ = NewSeenGeneration();
    Visit(root);
    return std::move(res_);
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
    if (filter_ == nullptr || !filter_(node)) {
      res_.push_back(node);
    }
    if (incl == FOLLOW) {
      AnfIrVisitor::Visit(node);
    }
  }

 private:
  size_t seen_{0};
  IncludeFunc include_;
  FilterFunc filter_;
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

// include for if expand the node the search, filter for if put the node to results.
std::vector<AnfNodePtr> DeepScopedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include) {
  return DeepScopedGraphSearcher(include).Search(root);
}

std::vector<AnfNodePtr> DeepScopedGraphSearchWithFilter(const AnfNodePtr &root, const IncludeFunc &include,
                                                        const FilterFunc &filter) {
  return DeepFirstSearcher(include, filter).Search(root);
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
}  // namespace mindspore
