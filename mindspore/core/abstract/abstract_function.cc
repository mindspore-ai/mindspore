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

#include "abstract/abstract_function.h"
#include <vector>
#include <utility>
#include <algorithm>
#include "base/base.h"
#include "utils/hashing.h"
#include "utils/hash_set.h"
#include "utils/ms_utils.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace abstract {
class Evaluator;
class AnalysisEngine;

AbstractFunctionPtr AbstractFunction::MakeAbstractFunction(const AbstractFuncAtomPtrList &func_list) {
  if (func_list.size() == 1) {
    return func_list[0];
  }
  return std::make_shared<AbstractFuncUnion>(func_list);
}

AbstractFunctionPtr AbstractFuncAtom::Join(const AbstractFunctionPtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  auto this_func = shared_from_base<AbstractFuncAtom>();
  if (other->isa<AbstractFuncAtom>()) {
    if (*this_func == *other) {
      return this_func;
    }
    return std::make_shared<AbstractFuncUnion>(this_func, other);
  }
  auto other_union = dyn_cast_ptr<AbstractFuncUnion>(other);
  MS_EXCEPTION_IF_NULL(other_union);
  if (other_union->IsSuperSet(this_func)) {
    return other;
  }
  return std::make_shared<AbstractFuncUnion>(this_func, other);
}

void AbstractFuncAtom::Visit(std::function<void(const AbstractFuncAtomPtr &)> visit_func) const {
  visit_func(const_cast<AbstractFuncAtom *>(this)->shared_from_base<AbstractFuncAtom>());
}

bool AbstractFuncAtom::operator==(const AbstractFunction &other) const { return this == &other; }

AbstractFuncUnion::AbstractFuncUnion(const AbstractFuncAtomPtrList &func_list) : func_list_(func_list) {}

AbstractFuncUnion::AbstractFuncUnion(const AbstractFunctionPtr &first, const AbstractFunctionPtr &second) {
  MS_EXCEPTION_IF_NULL(first);
  MS_EXCEPTION_IF_NULL(second);
  AbstractFuncAtomPtrList new_func_list;
  auto build_func_list = [&new_func_list](const AbstractFuncAtomPtr &func) { new_func_list.push_back(func); };
  first->Visit(build_func_list);
  second->Visit(build_func_list);
  func_list_ = std::move(new_func_list);
}

std::string AbstractFuncUnion::ToString() const {
  std::ostringstream buffer;
  buffer << "AbstractFuncUnion({";
  int64_t i = 0;
  for (const auto &func : func_list_) {
    MS_EXCEPTION_IF_NULL(func);
    buffer << "[" << i << "]: " << func->ToString() << ", ";
    i++;
  }
  buffer << "})";
  return buffer.str();
}

std::string AbstractFuncUnion::ToString(bool verbose) const {
  if (verbose) {
    return ToString();
  }
  std::ostringstream buffer;
  buffer << type_name() << "({";
  size_t i = 0;
  for (const auto &func : func_list_) {
    MS_EXCEPTION_IF_NULL(func);
    buffer << func->ToString(false);
    i++;
    if (i < func_list_.size()) {
      buffer << ", ";
    }
  }
  buffer << "})";
  return buffer.str();
}

bool AbstractFuncUnion::IsSuperSet(const AbstractFunctionPtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  bool all_in_list = true;
  other->Visit([this, &all_in_list](const AbstractFuncAtomPtr &func) {
    if (all_in_list) {
      auto iter = std::find(func_list_.begin(), func_list_.end(), func);
      if (iter == func_list_.end()) {
        all_in_list = false;
      }
    }
  });
  return all_in_list;
}

AbstractFunctionPtr AbstractFuncUnion::Join(const AbstractFunctionPtr &other) {
  auto this_func = shared_from_base<AbstractFunction>();
  MS_EXCEPTION_IF_NULL(other);
  if (other->isa<AbstractFuncAtom>()) {
    if (IsSuperSet(other)) {
      return this_func;
    }
    return std::make_shared<AbstractFuncUnion>(this_func, other);
  }
  auto other_union = dyn_cast_ptr<AbstractFuncUnion>(other);
  MS_EXCEPTION_IF_NULL(other_union);
  if (other_union->IsSuperSet(this_func)) {
    return other;
  }
  return std::make_shared<AbstractFuncUnion>(this_func, other);
}

void AbstractFuncUnion::Visit(std::function<void(const AbstractFuncAtomPtr &)> visit_func) const {
  for (const auto &poss : func_list_) {
    visit_func(poss);
  }
}

bool AbstractFuncUnion::operator==(const AbstractFunction &other) const {
  if (!other.isa<AbstractFuncUnion>()) {
    return false;
  }
  const auto &other_union = static_cast<const AbstractFuncUnion &>(other);
  if (func_list_.size() != other_union.func_list_.size()) {
    return false;
  }
  for (size_t i = 0; i < func_list_.size(); ++i) {
    if (!common::IsEqual(func_list_[i], other_union.func_list_[i])) {
      return false;
    }
  }
  return true;
}

std::size_t AbstractFuncUnion::hash() const {
  std::size_t hash_sum = 0;
  for (const auto &f : func_list_) {
    MS_EXCEPTION_IF_NULL(f);
    hash_sum = hash_combine(hash_sum, f->hash());
  }
  return hash_sum;
}

bool PrimitiveAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<PrimitiveAbstractClosure>()) {
    return false;
  }
  const auto &other_abs = static_cast<const PrimitiveAbstractClosure &>(other);
  return (prim_ == other_abs.prim_) && (tracking_id_ == other_abs.tracking_id_);
}

std::size_t PrimitiveAbstractClosure::hash() const {
  auto hash_value = static_cast<std::size_t>(tid());
  hash_value = hash_combine(hash_value, PointerHash<PrimitivePtr>{}(prim_));
  if (tracking_id_ != 0) {
    hash_value = hash_combine(hash_value, static_cast<size_t>(tracking_id_));
  }
  return hash_value;
}

std::string PrimitiveAbstractClosure::ToString(bool verbose) const {
  if (verbose) {
    return ToString();
  }
  return type_name() + " (" + prim_->name() + ")";
}

bool FuncGraphAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<FuncGraphAbstractClosure>()) {
    return false;
  }
  const auto &other_fg = static_cast<const FuncGraphAbstractClosure &>(other);
  return func_graph_ == other_fg.func_graph_ && context_ == other_fg.context_ && tracking_id_ == other_fg.tracking_id_;
}

bool FuncGraphAbstractClosure::IsEqualExceptTrackingId(const FuncGraphAbstractClosure &other) const {
  return (this == &other) || (func_graph_ == other.func_graph_ && context_ == other.context_);
}

std::size_t FuncGraphAbstractClosure::HashWithoutTrackingId() const {
  auto hash_value = hash_combine(tid(), PointerHash<FuncGraphPtr>{}(func_graph_));
  return hash_combine(hash_value, PointerHash<AnalysisContextPtr>{}(context_));
}

std::size_t FuncGraphAbstractClosure::hash() const {
  auto hash_value = hash_combine(tid(), PointerHash<FuncGraphPtr>{}(func_graph_));
  hash_value = hash_combine(hash_value, PointerHash<AnalysisContextPtr>{}(context_));
  if (tracking_id_ != 0) {
    hash_value = hash_combine(hash_value, static_cast<size_t>(tracking_id_));
  }
  return hash_value;
}

std::string FuncGraphAbstractClosure::ToString() const {
  std::stringstream ss;
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_EXCEPTION_IF_NULL(context_);
  ss << "FuncGraphAbstractClosure: "
     << "FuncGraph: " << func_graph_->ToString() << "; Context: " << context_->ToString();
  return ss.str();
}

std::string FuncGraphAbstractClosure::ToString(bool verbose) const {
  if (verbose) {
    return ToString();
  }
  std::stringstream ss;
  MS_EXCEPTION_IF_NULL(func_graph_);
  ss << type_name() << "(" << func_graph_->ToString() << ")";
  return ss.str();
}

bool MetaFuncGraphAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<MetaFuncGraphAbstractClosure>()) {
    return false;
  }
  const auto &other_meta_fg = static_cast<const MetaFuncGraphAbstractClosure &>(other);
  return (meta_func_graph_ == other_meta_fg.meta_func_graph_) && (tracking_id_ == other_meta_fg.tracking_id_);
}

std::size_t MetaFuncGraphAbstractClosure::hash() const {
  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  auto hash_value = hash_combine(tid(), PointerHash<MetaFuncGraphPtr>{}(meta_func_graph_));
  if (tracking_id_ != 0) {
    hash_value = hash_combine(hash_value, static_cast<size_t>(tracking_id_));
  }
  return hash_value;
}

std::string MetaFuncGraphAbstractClosure::ToString() const {
  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  return "MetaFuncGraphAbstractClosure: " + meta_func_graph_->name();
}

namespace {
// Helper class to prevent recursive calls.
class VisitedHistory {
 public:
  explicit VisitedHistory(const void *address) : visited_(!history_.emplace(address).second) { ++deep_; }
  ~VisitedHistory() {
    --deep_;
    // cppcheck-suppress *
    if (deep_ == 0) {  // The result of cppcheck is "Condition (deep_==0) is always true". But it's wrong.
      history_.clear();
    }
  }
  bool IsVisited() const { return visited_; }

 private:
  static inline thread_local mindspore::HashSet<const void *> history_;
  static inline thread_local size_t deep_ = 0;
  bool visited_{false};
};
}  // namespace

bool PartialAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<PartialAbstractClosure>()) {
    return false;
  }
  // Avoid to recursively compare.
  VisitedHistory history(this);
  if (history.IsVisited()) {
    return true;
  }
  const auto &other_partial = static_cast<const PartialAbstractClosure &>(other);
  if (!common::IsEqual(fn_, other_partial.fn_)) {
    return false;
  }
  if (args_abs_list_.size() != other_partial.args_abs_list_.size()) {
    return false;
  }
  for (size_t i = 0; i < args_abs_list_.size(); ++i) {
    const auto &a = args_abs_list_[i];
    const auto &b = other_partial.args_abs_list_[i];
    if (a != nullptr && a->isa<AbstractFunction>()) {
      if (!common::IsEqual(a, b)) {
        return false;
      }
    } else if (a != b) {
      return false;
    }
  }
  return true;
}

std::size_t PartialAbstractClosure::hash() const {
  // Avoid to recursively hashing.
  VisitedHistory history(this);
  if (history.IsVisited()) {
    return 0;
  }
  MS_EXCEPTION_IF_NULL(fn_);
  auto hash_value = hash_combine(tid(), fn_->hash());
  for (const auto &arg : args_abs_list_) {
    if (arg != nullptr && arg->isa<AbstractFunction>()) {
      hash_value = hash_combine(hash_value, arg->hash());
    } else {
      hash_value = hash_combine(hash_value, PointerHash<AbstractBasePtr>{}(arg));
    }
  }
  return hash_value;
}

std::string PartialAbstractClosure::ToString() const {
  // Avoid to recursively ToString.
  VisitedHistory history(this);
  if (history.IsVisited()) {
    return "<recurred>";
  }
  std::ostringstream buffer;
  buffer << "PartialAbstractClosure{" << fn_->ToString() << "(";
  for (const auto &arg : args_abs_list_) {
    buffer << (arg == nullptr ? "<null>" : arg->ToString()) << ", ";
  }
  buffer << ")}";
  return buffer.str();
}

std::string PartialAbstractClosure::ToString(bool verbose) const {
  if (verbose) {
    return ToString();
  }
  std::ostringstream buffer;
  buffer << type_name() << "(" << fn_->ToString(false) << " (argc=" << args_abs_list_.size() << "))";
  return buffer.str();
}

bool JTransformedAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<JTransformedAbstractClosure>()) {
    return false;
  }
  const auto &other_transformed = static_cast<const JTransformedAbstractClosure &>(other);
  return fn_ == other_transformed.fn_;
}

std::size_t JTransformedAbstractClosure::hash() const {
  return hash_combine(tid(), PointerHash<AbstractFuncAtomPtr>{}(fn_));
}

bool TaylorTransformedAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<TaylorTransformedAbstractClosure>()) {
    return false;
  }
  const auto &other_transformed = static_cast<const TaylorTransformedAbstractClosure &>(other);
  return fn_ == other_transformed.fn_;
}

std::size_t TaylorTransformedAbstractClosure::hash() const {
  return hash_combine(tid(), PointerHash<AbstractFuncAtomPtr>{}(fn_));
}

bool ShardTransformedAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<ShardTransformedAbstractClosure>()) {
    return false;
  }
  const auto &other_transformed = static_cast<const ShardTransformedAbstractClosure &>(other);
  return fn_ == other_transformed.fn_;
}

std::size_t ShardTransformedAbstractClosure::hash() const {
  return hash_combine(tid(), PointerHash<AbstractFuncAtomPtr>{}(fn_));
}

bool VmapTransformedAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<VmapTransformedAbstractClosure>()) {
    return false;
  }
  const auto &other_transformed = static_cast<const VmapTransformedAbstractClosure &>(other);
  return fn_ == other_transformed.fn_ && in_axes_ == other_transformed.in_axes_ &&
         out_axes_ == other_transformed.out_axes_;
}

std::size_t VmapTransformedAbstractClosure::hash() const {
  auto hash_value = hash_combine(tid(), PointerHash<AbstractFuncAtomPtr>{}(fn_));
  hash_value = hash_combine(hash_value, PointerHash<ValuePtr>{}(in_axes_));
  hash_value = hash_combine(hash_value, PointerHash<ValuePtr>{}(out_axes_));
  return hash_value;
}

bool VirtualAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<VirtualAbstractClosure>()) {
    return false;
  }
  const auto &other_virtual = static_cast<const VirtualAbstractClosure &>(other);
  if (!common::IsEqual(output_, other_virtual.output_)) {
    return false;
  }
  return AbstractBasePtrListDeepEqual(args_abs_list_, other_virtual.args_abs_list_);
}

std::size_t VirtualAbstractClosure::hash() const {
  MS_EXCEPTION_IF_NULL(output_);
  auto hash_value = hash_combine(tid(), output_->hash());
  return hash_combine(hash_value, AbstractBasePtrListHash(args_abs_list_));
}

std::string VirtualAbstractClosure::ToString() const {
  std::ostringstream buffer;
  buffer << "VirtualAbstractClosure(args: {";
  int64_t i = 0;
  for (const auto &arg : args_abs_list_) {
    MS_EXCEPTION_IF_NULL(arg);
    if (arg->isa<AbstractFuncAtom>()) {
      // If the arg is a subclass of AbstractFuncAtom, a recursive dead loop may occur.
      // So in this case, we use type_name() instead of ToString().
      buffer << "[" << i << "]: " << arg->type_name() << ", ";
    } else {
      buffer << "[" << i << "]: " << arg->ToString() << ", ";
    }
    i++;
  }
  MS_EXCEPTION_IF_NULL(output_);
  buffer << "}, output: " << output_->ToString() << ")";
  return buffer.str();
}

bool TypedPrimitiveAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<TypedPrimitiveAbstractClosure>()) {
    return false;
  }
  // Avoid to recursively compare.
  VisitedHistory history(this);
  if (history.IsVisited()) {
    return true;
  }
  const auto &other_typed = static_cast<const TypedPrimitiveAbstractClosure &>(other);
  if (prim_ != other_typed.prim_) {
    return false;
  }
  if (!common::IsEqual(output_, other_typed.output_)) {
    return false;
  }
  return AbstractBasePtrListDeepEqual(args_abs_list_, other_typed.args_abs_list_);
}

std::size_t TypedPrimitiveAbstractClosure::hash() const {
  // Avoid to recursively hashing.
  VisitedHistory history(this);
  if (history.IsVisited()) {
    return 0;
  }
  auto hash_value = hash_combine(tid(), PointerHash<PrimitivePtr>{}(prim_));
  if (output_ != nullptr) {
    hash_value = hash_combine(hash_value, output_->hash());
  }
  hash_value = hash_combine(hash_value, AbstractBasePtrListHash(args_abs_list_));
  return hash_value;
}

std::string TypedPrimitiveAbstractClosure::ToString() const {
  // Avoid to recursively ToString.
  VisitedHistory history(this);
  if (history.IsVisited()) {
    return "<recurred>";
  }
  std::ostringstream buffer;
  buffer << "TypedPrimitiveAbstractClosure: primitive: " << prim_->name() << "(args: {";
  for (const auto &arg : args_abs_list_) {
    buffer << (arg == nullptr ? "<null>" : arg->ToString()) << ", ";
  }
  MS_EXCEPTION_IF_NULL(output_);
  buffer << "}, output: " << output_->ToString() << ")";
  return buffer.str();
}
}  // namespace abstract
}  // namespace mindspore
