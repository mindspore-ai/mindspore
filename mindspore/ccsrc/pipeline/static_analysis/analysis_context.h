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

#ifndef PIPELINE_STATIC_ANALYSIS_ANALYSIS_CONTEXT_H_
#define PIPELINE_STATIC_ANALYSIS_ANALYSIS_CONTEXT_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "pipeline/static_analysis/abstract_value.h"
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
      parent_cache_ = parent_->parent_cache_;
    }
  }

  ~AnalysisContext() = default;

  // Helper function to wrapper constructor to save shared_ptr in parent_cache.
  AnalysisContextPtr NewContext(AnalysisContextPtr parent, FuncGraphPtr fg, const AbstractBasePtrList &args_spec_list);

  // Extend this context with values for another graph.
  AnalysisContextPtr NewFuncGraphContext(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list);

  // Return a context restricted to a graph's dependencies.
  AnalysisContextPtr Filter(const FuncGraphPtr &graph);
  bool operator==(const AnalysisContext &other) const;
  std::size_t hash();
  static AnalysisContextPtr DummyContext();
  bool IsDummyContext();
  FuncGraphPtr func_graph() const { return func_graph_; }
  AnalysisContextPtr parent() const { return parent_; }
  std::string ToString() const;
  AnalysisContextPtr SpecializeKey() const;
  AbstractBasePtrList args_spec_list() { return args_spec_list_; }

 private:
  AnalysisContextPtr parent_;
  FuncGraphPtr func_graph_;
  AbstractBasePtrList args_spec_list_;
  std::unordered_map<FuncGraphPtr, AnalysisContextWeakPtr> parent_cache_;
  std::unordered_map<FuncGraphPtr, ArgsSpecToAnalysisContextMap> children_cache_;
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
#endif  // PIPELINE_STATIC_ANALYSIS_ANALYSIS_CONTEXT_H_
