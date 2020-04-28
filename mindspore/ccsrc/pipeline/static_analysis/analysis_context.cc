/**
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

#include "pipeline/static_analysis/analysis_context.h"

#include <algorithm>

#include "utils/symbolic.h"
#include "debug/trace.h"

namespace mindspore {
namespace abstract {
AnalysisContextPtr AnalysisContext::NewFuncGraphContext(const FuncGraphPtr &func_graph,
                                                        const AbstractBasePtrList &args_spec_list) {
  FuncGraphPtr graph_parent = func_graph->parent();
  auto iter = parent_cache_.find(graph_parent);
  AnalysisContextPtr parent_context = nullptr;
  if (iter != parent_cache_.end()) {
    parent_context = iter->second.lock();
  }
  // if this happen, it will be bug in code. but we raise exception to keep the scene.
  if (parent_context == nullptr) {
    std::ostringstream oss;
    oss << "BUG: cannot found parent_context in current context: " << this->ToString()
        << ", func_graph: " << func_graph->ToString() << ", graph_parent: ";
    if (graph_parent != nullptr) {
      oss << graph_parent->ToString();
    } else {
      oss << "nullptr";
    }
    MS_LOG(EXCEPTION) << oss.str() << " NodeInfo: " << trace::GetDebugInfo(func_graph->debug_info());
  }
  return NewContext(parent_context, func_graph, args_spec_list);
}

AnalysisContextPtr AnalysisContext::Filter(const FuncGraphPtr &func_graph) {
  auto p_iter = parent_cache_.find(func_graph);
  AnalysisContextPtr parent_context = nullptr;
  if (p_iter != parent_cache_.end()) {
    parent_context = p_iter->second.lock();
  } else {
    auto iter_parent = parent_cache_.find(func_graph->parent());
    if (iter_parent != parent_cache_.end()) {
      parent_context = iter_parent->second.lock();
    }
  }
  // if this happen, it will be bug in code. but we raise exception to keep the scene.
  if (parent_context == nullptr) {
    std::ostringstream oss;
    oss << "BUG: Filter graph failed: " << func_graph->ToString() << ", graph_parent: ";
    if (func_graph->parent() != nullptr) {
      oss << func_graph->parent()->ToString();
    } else {
      oss << "nullptr";
    }
    oss << " parent_cache_: {";
    for (auto iter : parent_cache_) {
      if (iter.first == nullptr) {
        oss << " [graph: nullptr";
      } else {
        oss << " [graph: " << iter.first->ToString();
      }
      // iter.second cannot be nullptr even iter.first is nullptr as it will
      // always be a Context() object.
      oss << ", context: " << iter.second.lock()->ToString() << "]";
    }
    oss << "}";
    MS_LOG(EXCEPTION) << oss.str() << " NodeInfo: " << trace::GetDebugInfo(func_graph->debug_info());
  }
  return parent_context;
}

AnalysisContextPtr AnalysisContext::DummyContext() {
  AnalysisContextPtr dummy_context = std::make_shared<AnalysisContext>(nullptr, nullptr, AbstractBasePtrList());
  dummy_context->parent_cache_[nullptr] = std::weak_ptr<AnalysisContext>(dummy_context);
  return dummy_context;
}

const AnalysisContextPtr kDummyAnalysisContext =
  std::make_shared<AnalysisContext>(nullptr, nullptr, AbstractBasePtrList());

bool AnalysisContext::operator==(const AnalysisContext &other) const {
  if (func_graph_ != other.func_graph_) {
    return false;
  }

  if (args_spec_list_.size() != other.args_spec_list_.size()) {
    return false;
  }

  if (((parent_ == nullptr) && (other.parent_ != nullptr)) || ((parent_ != nullptr) && (other.parent_ == nullptr))) {
    return false;
  }
  // Compare parent with content.
  bool is_parent_equal = false;
  if (parent_ == other.parent_) {
    is_parent_equal = true;
  } else if (*parent_ == *other.parent_) {
    is_parent_equal = true;
  } else {
    return false;
  }
  for (std::size_t i = 0; i < args_spec_list_.size(); i++) {
    if (!(*args_spec_list_[i] == *other.args_spec_list_[i])) {
      return false;
    }
  }
  return is_parent_equal;
}

// brief The key which controls the graph cloning in Specialize.
//
// Originally, specialize use context directly as the key for cloning graph. The graph will be cloned multiple times
// for different context, which means the graph is called from different node with different arguments and different
// free values. In order to decrease the number of cloned graphs, we add this `SpecializeKey` method to control what
// graph can be reused.
// The graph called with different SymbolicKey will be reused. The abstract of SymbolicKey parameter will be joined
// and stored in the intermediate_abstract. The joined SymbolicKey would cause Poly Code in infer, thus the reused
// graph with SymbolicKey parameter should be inlined in `opt` pipeline before the next renormalize.
// The graph called with different shape should not be reused, because the combination of `shape` and `Fill` relies
// on correct shape to specialize a tensor constant.
AnalysisContextPtr AnalysisContext::SpecializeKey() const {
  AbstractBasePtrList args_broad_shp;
  (void)std::transform(args_spec_list_.begin(), args_spec_list_.end(), std::back_inserter(args_broad_shp),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         if (arg->isa<AbstractScalar>()) {
                           auto val = arg->GetValueTrack();
                           if (val->isa<SymbolicKeyInstance>()) {
                             auto scalar_spec = dyn_cast<AbstractScalar>(arg);
                             auto ret_spec = scalar_spec->Broaden();
                             ret_spec->set_value(kAnyValue);
                             return ret_spec;
                           }
                         }
                         if (arg->isa<AbstractRef>()) {
                           MS_LOG(DEBUG) << "refkey broaden";
                           auto arg_spec = dyn_cast<AbstractRef>(arg);
                           auto ret_spec = arg_spec->Broaden();
                           return ret_spec;
                         }
                         return arg;
                       });
  AnalysisContextPtr context_new = std::make_shared<AnalysisContext>(nullptr, func_graph_, args_broad_shp);
  context_new->parent_ = parent_;
  return context_new;
}

std::size_t AnalysisContext::hash() {
  std::size_t hash_value = 0;
  // hash() recursion exit condition.
  if (parent_ != nullptr) {
    hash_value = hash_combine(hash_value, parent_->hash());
  }
  if (func_graph_ != nullptr) {
    hash_value = hash_combine(hash_value, func_graph_->hash());
  }
  return hash_value;
}

std::string AnalysisContext::ToString() const {
  std::ostringstream buffer;
  buffer << "{";
  if (func_graph_ != nullptr) {
    buffer << "Func Graph: " << func_graph_->ToString();
  }
  buffer << " Args: ";
  int i = 0;
  for (const auto &arg : args_spec_list_) {
    buffer << "[" << i << "]: " << arg->ToString() << ", ";
    i++;
  }
  if (parent_ != nullptr) {
    buffer << "Parent: " << parent_->ToString();
  }
  buffer << "}";
  return buffer.str();
}
}  // namespace abstract
}  // namespace mindspore
