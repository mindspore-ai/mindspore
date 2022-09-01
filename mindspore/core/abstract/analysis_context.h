/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <vector>
#include <utility>
#include <unordered_map>

#include "utils/macros.h"
#include "utils/hashing.h"
#include "ir/func_graph.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace abstract {
//
// AnalysisContext represents the context that a func graph is called.
// - parent context: determines the free variables used by the func graph;
// - func graph: the func graph be called in current context;
// - argument list: the argument abstracts used to call the func graph.
//
class MS_CORE_API AnalysisContext : public std::enable_shared_from_this<AnalysisContext> {
 public:
  ~AnalysisContext() = default;

  AnalysisContextPtr parent() const { return parent_ == nullptr ? nullptr : parent_->shared_from_this(); }
  const FuncGraphPtr &func_graph() const { return func_graph_; }
  const AbstractBasePtrList &args_spec_list() const { return args_spec_list_; }

  // Extend this context with values for another graph.
  AnalysisContextPtr NewContext(const FuncGraphPtr &fg, const AbstractBasePtrList &args_spec_list);

  // Return a context restricted to a graph and its parent.
  AnalysisContextPtr FindOwnOrParentContext(FuncGraph *fg);

  std::string ToString() const;

  // Return current root dummy context.
  static AnalysisContextPtr DummyContext();

  // Create a new root dummy context.
  static AnalysisContextPtr NewDummyContext();

  // Clear all contexts to release resources.
  static void ClearContext();

 protected:
  // Make constructor protected to prevent creating an isolated context object.
  AnalysisContext(AnalysisContext *parent, const FuncGraphPtr &fg, const AbstractBasePtrList &args_spec_list)
      : parent_(parent), func_graph_(fg), args_spec_list_(args_spec_list) {}

 private:
  // Clear to release resources.
  void Clear();

  // Find context from the given func graph.
  AnalysisContext *FindContext(const FuncGraphPtr &fg);

  // Create a new context instance.
  static AnalysisContextPtr CreateContext(AnalysisContext *parent, const FuncGraphPtr &fg,
                                          const AbstractBasePtrList &args_spec_list);

  using ChildKey = std::pair<FuncGraphPtr, AbstractBasePtrList>;

  struct ChildHash {
    std::size_t operator()(const ChildKey &key) const noexcept;
  };

  struct ChildEqual {
    bool operator()(const ChildKey &a, const ChildKey &b) const noexcept;
  };

  // Parent context, use raw pointer to avoid cycle reference.
  AnalysisContext *parent_;

  // Func graph for current context.
  FuncGraphPtr func_graph_;

  // Func graph arguments in current context.
  AbstractBasePtrList args_spec_list_;

  // Children contexts discriminated by func_graph & arguments.
  std::unordered_map<ChildKey, AnalysisContextPtr, ChildHash, ChildEqual> children_;

  // Root dummy contexts.
  static std::vector<AnalysisContextPtr> dummy_contexts_;
};
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_ANALYSIS_CONTEXT_H_
