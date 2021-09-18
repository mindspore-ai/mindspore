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

#include "abstract/analysis_context.h"

#include <algorithm>

#include "utils/symbolic.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace abstract {
std::list<AnalysisContextPtr> AnalysisContext::all_context_;
AnalysisContextPtr AnalysisContext::NewContext(const FuncGraphPtr &func_graph,
                                               const AbstractBasePtrList &args_spec_list) {
  // Find func graph's parent and its parent context firstly.
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphPtr parent_graph = func_graph->parent();
  AnalysisContextPtr parent_context = nullptr;
  auto iter = extant_context_cache_.find(parent_graph);
  if (iter != extant_context_cache_.end()) {
    parent_context = iter->second.lock();
  }
  if (parent_context == nullptr) {  // If parent context is not found, we'll raise exception.
    std::ostringstream oss;
    oss << "BUG: Failed to find parent context in current context: " << this->ToString()
        << ", func_graph: " << func_graph->ToString() << ", parent_graph: ";
    if (parent_graph != nullptr) {
      oss << parent_graph->ToString();
    } else {
      oss << "nullptr";
    }
    MS_LOG(EXCEPTION) << oss.str() << " NodeInfo: " << trace::GetDebugInfo(func_graph->debug_info());
  }

  // Check if we created a context for func graph with the same arguments before.
  auto children_context_map_iter = parent_context->children_cache_.find(func_graph);
  if (children_context_map_iter != parent_context->children_cache_.end()) {
    auto children_context_map = children_context_map_iter->second;
    auto children_context_iter = children_context_map.find(args_spec_list);
    if (children_context_iter != children_context_map.end()) {
      return children_context_iter->second.lock();
    }
  }

  // Create a new context for the func graph and its specific arguments.
  AnalysisContextPtr new_context = CreateContext(parent_context, func_graph, args_spec_list);
  // To avoid cycle-reference, use weak_ptr here.
  auto weak_new_context = std::weak_ptr<AnalysisContext>(new_context);
  new_context->extant_context_cache_[func_graph] = weak_new_context;
  parent_context->children_cache_[func_graph][args_spec_list] = weak_new_context;
  return new_context;
}

AnalysisContextPtr AnalysisContext::FindOwnOrParentContext(const FuncGraphPtr &func_graph) {
  auto p_iter = extant_context_cache_.find(func_graph);
  AnalysisContextPtr extant_context = nullptr;
  if (p_iter != extant_context_cache_.end()) {
    extant_context = p_iter->second.lock();
  } else {
    auto iter_parent = extant_context_cache_.find(func_graph->parent());
    if (iter_parent != extant_context_cache_.end()) {
      extant_context = iter_parent->second.lock();
    }
  }
  // If this happen, it would be a bug in code. But we raise exception to keep the scene.
  if (extant_context == nullptr) {
    std::ostringstream oss;
    oss << "BUG: Failed to find context for: " << func_graph->ToString() << ", parent_graph: ";
    if (func_graph->parent() != nullptr) {
      oss << func_graph->parent()->ToString();
    } else {
      oss << "nullptr";
    }
    oss << " extant context list: {";
    for (const auto &iter : extant_context_cache_) {
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
  return extant_context;
}

AnalysisContextPtr AnalysisContext::DummyContext() {
  AnalysisContextPtr dummy_context = CreateContext(nullptr, nullptr, AbstractBasePtrList());
  dummy_context->extant_context_cache_[nullptr] = std::weak_ptr<AnalysisContext>(dummy_context);
  return dummy_context;
}

bool AnalysisContext::IsDummyContext() {
  return parent_ == nullptr && func_graph_ == nullptr && args_spec_list_.empty();
}

const AnalysisContextPtr kDummyAnalysisContext =
  AnalysisContext::CreateContext(nullptr, nullptr, AbstractBasePtrList());

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
// Originally, specialize use context directly as the key for cloning graph. The graph will be cloned multiple times
// for different context, which means the graph is called from different node with different arguments and different
// free values. In order to decrease the number of cloned graphs, we add this `SpecializeKey` method to control what
// graph can be reused.
// The graph called with different SymbolicKey will be reused. The abstract of SymbolicKey parameter will be joined
// and stored in the intermediate_abstract. The joined SymbolicKey would cause Poly Code in eval, thus the reused
// graph with SymbolicKey parameter should be inlined in `opt` pipeline before the next renormalize.
// The graph called with different shape should not be reused, because the combination of `shape` and `Fill` relies
// on correct shape to specialize a tensor constant.
AnalysisContextPtr AnalysisContext::SpecializeKey() const {
  AbstractBasePtrList args_broad_shp;
  (void)std::transform(args_spec_list_.begin(), args_spec_list_.end(), std::back_inserter(args_broad_shp),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         if (arg->isa<AbstractScalar>()) {
                           auto val = arg->GetValueTrack();
                           MS_EXCEPTION_IF_NULL(val);
                           if (val->isa<SymbolicKeyInstance>()) {
                             auto scalar_spec = dyn_cast<AbstractScalar>(arg);
                             MS_EXCEPTION_IF_NULL(scalar_spec);
                             auto ret_spec = scalar_spec->Broaden();
                             return ret_spec;
                           }
                         }
                         if (arg->isa<AbstractRef>()) {
                           MS_LOG(DEBUG) << "refkey broaden";
                           return arg->Broaden();
                         }
                         return arg;
                       });
  AnalysisContextPtr context_new = CreateContext(nullptr, func_graph_, args_broad_shp);
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
  int64_t i = 0;
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

void AnalysisContext::ClearContext() {
  for (auto &item : all_context_) {
    item->parent_ = nullptr;
    item->func_graph_ = nullptr;
    item->args_spec_list_.clear();
    item->extant_context_cache_.clear();
    item->children_cache_.clear();
  }
  all_context_.clear();
}

AnalysisContextPtr AnalysisContext::CreateContext(const AnalysisContextPtr &parent, const FuncGraphPtr &fg,
                                                  const AbstractBasePtrList &args_spec_list) {
  auto context = std::make_shared<AnalysisContext>(parent, fg, args_spec_list);
  (void)all_context_.emplace_back(context);
  return context;
}
}  // namespace abstract
}  // namespace mindspore
