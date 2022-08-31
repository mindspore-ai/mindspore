/**
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

#include "abstract/analysis_context.h"

#include <vector>
#include <utility>
#include <algorithm>

#include "utils/flags.h"
#include "utils/hashing.h"
#include "utils/ms_utils.h"
#include "utils/trace_base.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"

namespace mindspore {
namespace abstract {
// Sotre all root dummy contexts here.
std::vector<AnalysisContextPtr> AnalysisContext::dummy_contexts_;

// Special equal function for 'while' header:
// Ignore tracking_id when checking equality of FuncGraphAbstractClosure.
static bool IsEqualForWhileHeader(const AbstractBasePtr &a1, const AbstractBasePtr &a2) {
  auto f1 = dyn_cast_ptr<abstract::FuncGraphAbstractClosure>(a1);
  if (f1 != nullptr) {
    auto f2 = dyn_cast_ptr<abstract::FuncGraphAbstractClosure>(a2);
    return f2 != nullptr && f2->IsEqualExceptTrackingId(*f1);
  }
  return common::IsEqual(a1, a2);
}

// Special abstract list equal compare function for 'while' header.
static bool ArgsEqualForWhileHeader(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs) {
  const std::size_t size = lhs.size();
  if (size != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (!IsEqualForWhileHeader(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

// Special abstract list hash for 'while' header:
// Ignore tracking_id when calculate hash for FuncGraphAbstractClosure.
static std::size_t ArgsHashForWhileHeader(const AbstractBasePtrList &args) {
  std::size_t hash_value = args.size();
  for (auto &abs : args) {
    auto fg_abs = dyn_cast_ptr<abstract::FuncGraphAbstractClosure>(abs);
    if (fg_abs != nullptr) {
      hash_value = hash_combine(hash_value, fg_abs->HashWithoutTrackingId());
    } else {
      hash_value = hash_combine(hash_value, abs->hash());
    }
  }
  return hash_value;
}

// Checking children equality.
bool AnalysisContext::ChildEqual::operator()(const ChildKey &a, const ChildKey &b) const noexcept {
  if (a.first != b.first) {
    return false;
  }
  if (a.first != nullptr && a.first->has_flag(GRAPH_FLAG_IS_WHILE_HEADER)) {
    return ArgsEqualForWhileHeader(a.second, b.second);
  }
  return AbstractBasePtrListDeepEqual(a.second, b.second);
}

// Calculate children hash.
std::size_t AnalysisContext::ChildHash::operator()(const ChildKey &key) const noexcept {
  std::size_t hash_value = PointerHash<FuncGraphPtr>{}(key.first);
  if (key.first != nullptr && key.first->has_flag(GRAPH_FLAG_IS_WHILE_HEADER)) {
    return hash_combine(hash_value, ArgsHashForWhileHeader(key.second));
  }
  return hash_combine(hash_value, AbstractBasePtrListHash(key.second));
}

AnalysisContextPtr AnalysisContext::NewContext(const FuncGraphPtr &fg, const AbstractBasePtrList &args_spec_list) {
  // Find func graph's parent and its parent context firstly.
  MS_EXCEPTION_IF_NULL(fg);
  FuncGraphPtr parent_graph = fg->parent();
  auto parent_context = FindContext(parent_graph);
  if (parent_context == nullptr) {
    // If parent context is not found, we'll raise exception.
    MS_LOG(EXCEPTION) << "BUG: Failed to find parent context in current context: " << this->ToString()
                      << ", func_graph: " << fg->ToString()
                      << ", parent_graph: " << (parent_graph == nullptr ? "null" : parent_graph->ToString()) << " "
                      << trace::GetDebugInfo(fg->debug_info());
  }
  // Create or find child context from the parent context.
  auto result = parent_context->children_.emplace(std::make_pair(fg, args_spec_list), nullptr);
  if (result.second) {
    // If exist child not found, create a new context for the func graph with its specific arguments.
    result.first->second = CreateContext(parent_context, fg, args_spec_list);
  }
  return result.first->second;
}

AnalysisContext *AnalysisContext::FindContext(const FuncGraphPtr &fg) {
  if (fg == nullptr) {
    return DummyContext().get();
  }
  if (fg == func_graph_) {
    return this;
  }
  for (auto p = parent_; p != nullptr; p = p->parent_) {
    if (p->func_graph_ == fg) {
      return p;
    }
  }
  return nullptr;
}

AnalysisContextPtr AnalysisContext::FindOwnOrParentContext(FuncGraph *fg) {
  if (fg == nullptr) {
    return DummyContext();
  }
  auto parent_fg = fg->parent();
  if (func_graph_.get() == fg || func_graph_ == parent_fg) {
    return shared_from_this();
  }
  for (auto p = parent_; p != nullptr; p = p->parent_) {
    if (p->func_graph_.get() == fg || p->func_graph_ == parent_fg) {
      return p->shared_from_this();
    }
  }
  // Context not found, it would be a bug in code so we raise exception.
  std::ostringstream oss;
  oss << "BUG: Failed to find context for: " << fg->ToString()
      << ", parent: " << (parent_fg == nullptr ? "null" : parent_fg->ToString()) << " from contexts: [" << ToString();
  for (auto p = parent_; p != nullptr; p = p->parent_) {
    oss << ", " << p->ToString();
  }
  oss << "] " << trace::GetDebugInfo(fg->debug_info());
  MS_LOG(EXCEPTION) << oss.str();
}

AnalysisContextPtr AnalysisContext::DummyContext() {
  if (dummy_contexts_.empty()) {
    (void)NewDummyContext();
  }
  return dummy_contexts_.back();
}

AnalysisContextPtr AnalysisContext::NewDummyContext() {
  auto dummy_context = CreateContext(nullptr, nullptr, AbstractBasePtrList());
  dummy_contexts_.push_back(dummy_context);
  return dummy_context;
}

std::string AnalysisContext::ToString() const {
  std::ostringstream buffer;
  buffer << "{";
  if (func_graph_ != nullptr) {
    buffer << "FuncGraph: " << func_graph_->ToString();
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

void AnalysisContext::Clear() {
  // Recursively clear children.
  for (auto &child : children_) {
    if (child.second != nullptr) {
      child.second->Clear();
    }
  }
  children_.clear();
  parent_ = nullptr;
  func_graph_ = nullptr;
  args_spec_list_.clear();
}

AnalysisContextPtr AnalysisContext::CreateContext(AnalysisContext *parent, const FuncGraphPtr &fg,
                                                  const AbstractBasePtrList &args_spec_list) {
  // This is a hack to solve the problem that std::make_shared can only use public constructor.
  struct MakeSharedEnabler : public AnalysisContext {
    MakeSharedEnabler(AnalysisContext *parent, const FuncGraphPtr &fg, const AbstractBasePtrList &args_spec_list)
        : AnalysisContext(parent, fg, args_spec_list) {}
    ~MakeSharedEnabler() = default;
  };
  return std::make_shared<MakeSharedEnabler>(parent, fg, args_spec_list);
}

void AnalysisContext::ClearContext() {
  // Clear all root dummy contexts and their children.
  for (auto &context : dummy_contexts_) {
    if (context != nullptr) {
      // Children contexts will be cleared recursively.
      context->Clear();
    }
  }
  dummy_contexts_.clear();
}
}  // namespace abstract
}  // namespace mindspore
