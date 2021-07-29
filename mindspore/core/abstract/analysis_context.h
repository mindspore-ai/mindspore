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

#ifndef MINDSPORE_CORE_ABSTRACT_ANALYSIS_CONTEXT_H_
#define MINDSPORE_CORE_ABSTRACT_ANALYSIS_CONTEXT_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <list>

#include "abstract/abstract_value.h"
#include "ir/meta_func_graph.h"

namespace mindspore {
namespace abstract {
class AnalysisContext;
using AnalysisContextWeakPtr = std::weak_ptr<AnalysisContext>;
using ArgsSpecToAnalysisContextMap =
  std::unordered_map<AbstractBasePtrList, AnalysisContextWeakPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>;

// AnalysisContext will be stored in Config in AnalysisCache.
class AnalysisContext {
 public:
  AnalysisContext(const AnalysisContextPtr &parent, const FuncGraphPtr &fg, const AbstractBasePtrList &args_spec_list)
      : parent_(parent), func_graph_(fg), args_spec_list_(args_spec_list) {
    if (parent_ != nullptr) {
      extant_context_cache_ = parent_->extant_context_cache_;
    }
  }
  ~AnalysisContext() = default;

  // Extend this context with values for another graph.
  AnalysisContextPtr NewContext(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list);

  // Return a context restricted to a graph and its parent.
  AnalysisContextPtr FindOwnOrParentContext(const FuncGraphPtr &graph);
  bool operator==(const AnalysisContext &other) const;
  std::size_t hash();
  static AnalysisContextPtr DummyContext();
  bool IsDummyContext();
  FuncGraphPtr func_graph() const { return func_graph_; }
  AnalysisContextPtr parent() const { return parent_; }
  std::string ToString() const;
  AnalysisContextPtr SpecializeKey() const;
  AbstractBasePtrList args_spec_list() { return args_spec_list_; }
  static void ClearContext();
  static AnalysisContextPtr CreateContext(const AnalysisContextPtr &parent, const FuncGraphPtr &fg,
                                          const AbstractBasePtrList &args_spec_list);

 private:
  AnalysisContextPtr parent_;
  FuncGraphPtr func_graph_;
  AbstractBasePtrList args_spec_list_;
  // Record all created context for each func graph.
  // `extant_context_cache_` is copied from its parent context.
  std::unordered_map<FuncGraphPtr, AnalysisContextWeakPtr> extant_context_cache_;
  // Record all created child contexts from this context.
  // Like: key: [func_graph & arguments], value: [child_context]
  std::unordered_map<FuncGraphPtr, ArgsSpecToAnalysisContextMap> children_cache_;

  // There may may be shared_ptr loop like:
  // FuncGraphAbstactClosur->AnalysisContext->children_cache_->ArgsSpec->FuncGraphAbstactClosur.
  // For break the loop, using all_context_ to clear context_.
  static std::list<AnalysisContextPtr> all_context_;
};

struct ContextHasher {
  std::size_t operator()(const AnalysisContextPtr &t) const {
    std::size_t hash = t->hash();
    return hash;
  }
};

struct ContextEqual {
  bool operator()(const AnalysisContextPtr &lhs, const AnalysisContextPtr &rhs) const { return *lhs == *rhs; }
};

extern const AnalysisContextPtr kDummyAnalysisContext;
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_ANALYSIS_CONTEXT_H_
