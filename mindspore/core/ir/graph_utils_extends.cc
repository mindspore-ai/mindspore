/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "utils/hash_map.h"
#include "utils/hash_set.h"
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
      : include_(include), filter_(filter) {
    constexpr size_t kVecReserve = 64;
    res_.reserve(kVecReserve);
  }
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
    if (node == nullptr || node->seen_ == seen_) {
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
  SeenNum seen_{0};
  IncludeFunc include_;
  FilterFunc filter_;
  std::vector<AnfNodePtr> res_{};
};

class DeepScopedGraphSearcher : public DeepFirstSearcher {
 public:
  explicit DeepScopedGraphSearcher(const IncludeFunc &include) : DeepFirstSearcher(include) {}
  ~DeepScopedGraphSearcher() override = default;

  void Visit(const CNodePtr &cnode) override {
    auto fg = cnode->func_graph();
    if (fg == nullptr) {
      return;
    }
    AnfNodePtr ret = fg->return_node();
    DeepFirstSearcher::Visit(ret);

    auto &inputs = cnode->inputs();
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      DeepFirstSearcher::Visit(*iter);
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (!IsValueNode<FuncGraph>(vnode)) {
      return;
    }
    auto fg = GetValuePtr<FuncGraph>(vnode);
    const auto &ret = fg->return_node();
    DeepFirstSearcher::Visit(ret);
  }

  void Visit(const ParameterPtr &param) override {
    auto fg = param->func_graph();
    if (fg == nullptr) {
      return;
    }
    AnfNodePtr ret = fg->return_node();
    DeepFirstSearcher::Visit(ret);
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
}  // namespace

// include for if expand the node the search, filter for if put the node to results.
std::vector<AnfNodePtr> DeepScopedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include) {
  return DeepScopedGraphSearcher(include).Search(root);
}

std::vector<AnfNodePtr> DeepScopedGraphSearchWithFilter(const AnfNodePtr &root, const IncludeFunc &include,
                                                        const FilterFunc &filter) {
  return DeepFirstSearcher(include, filter).Search(root);
}

std::vector<AnfNodePtr> DeepLinkedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include) {
  return DeepLinkedGraphSearcher(include).Search(root);
}
}  // namespace mindspore
