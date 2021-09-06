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

#include "abstract/abstract_function.h"

#include <vector>

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
  auto other_union = dyn_cast<AbstractFuncUnion>(other);
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

AbstractFuncUnion::AbstractFuncUnion(const AbstractFuncAtomPtrList &func_list) { func_list_ = func_list; }

AbstractFuncUnion::AbstractFuncUnion(const AbstractFunctionPtr &first, const AbstractFunctionPtr &second) {
  AbstractFuncAtomPtrList new_func_list;
  auto build_func_list = [&new_func_list](const AbstractFuncAtomPtr &func) { new_func_list.push_back(func); };
  MS_EXCEPTION_IF_NULL(first);
  MS_EXCEPTION_IF_NULL(second);
  first->Visit(build_func_list);
  second->Visit(build_func_list);
  func_list_ = new_func_list;
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

bool AbstractFuncUnion::IsSuperSet(const AbstractFunctionPtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  std::vector<bool> is_in_list;
  auto build_in_list = [this, &is_in_list](const AbstractFuncAtomPtr &func) {
    auto iter = find(func_list_.begin(), func_list_.end(), func);
    if (iter == func_list_.end()) {
      is_in_list.push_back(false);
    }
    return true;
  };
  other->Visit(build_in_list);
  return std::all_of(is_in_list.begin(), is_in_list.end(), [](bool is_in) { return is_in; });
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
  auto other_union = dyn_cast<AbstractFuncUnion>(other);
  MS_EXCEPTION_IF_NULL(other_union);
  if (other_union->IsSuperSet(this_func)) {
    return other;
  }
  return std::make_shared<AbstractFuncUnion>(this_func, other);
}

void AbstractFuncUnion::Visit(std::function<void(const AbstractFuncAtomPtr &)> visit_func) const {
  for (const AbstractFuncAtomPtr &poss : func_list_) {
    visit_func(poss);
  }
}

bool AbstractFuncUnion::operator==(const AbstractFunction &other) const {
  if (!other.isa<AbstractFuncUnion>()) {
    return false;
  }
  auto other_union = static_cast<const AbstractFuncUnion *>(&other);
  if (func_list_.size() != other_union->func_list_.size()) {
    return false;
  }
  return func_list_ == other_union->func_list_;
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
  auto other_prim = static_cast<const PrimitiveAbstractClosure *>(&other);
  MS_EXCEPTION_IF_NULL(prim_);
  return (prim_ == other_prim->prim_ && tracking_id() == other_prim->tracking_id());
}

std::size_t PrimitiveAbstractClosure::hash() const {
  auto hash_value = hash_combine(tid(), prim_->hash());
  // Keep in sync with operator==() which compares the prim_ pointer;
  hash_value = hash_combine(hash_value, std::hash<Primitive *>{}(prim_.get()));
  if (tracking_id() != nullptr) {
    hash_value = hash_combine(hash_value, tracking_id()->hash());
  }
  return hash_value;
}

bool FuncGraphAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<FuncGraphAbstractClosure>()) {
    return false;
  }
  auto other_fg = static_cast<const FuncGraphAbstractClosure *>(&other);
  return func_graph_ == other_fg->func_graph_ && context_ == other_fg->context_ &&
         tracking_id() == other_fg->tracking_id();
}

std::size_t FuncGraphAbstractClosure::hash() const {
  auto hash_value = hash_combine(tid(), func_graph_->hash());
  hash_value = hash_combine(hash_value, context_->hash());
  if (tracking_id() != nullptr) {
    hash_value = hash_combine(hash_value, tracking_id()->hash());
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

bool MetaFuncGraphAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<MetaFuncGraphAbstractClosure>()) {
    return false;
  }
  auto other_meta_fg = static_cast<const MetaFuncGraphAbstractClosure *>(&other);
  return meta_func_graph_ == other_meta_fg->meta_func_graph_ && tracking_id() == other_meta_fg->tracking_id();
}

std::size_t MetaFuncGraphAbstractClosure::hash() const {
  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  auto hash_value = hash_combine(tid(), meta_func_graph_->hash());
  if (tracking_id() != nullptr) {
    hash_value = hash_combine(hash_value, tracking_id()->hash());
  }
  return hash_value;
}

std::string MetaFuncGraphAbstractClosure::ToString() const {
  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  return "MetaFuncGraphAbstractClosure: " + meta_func_graph_->name();
}

bool PartialAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<PartialAbstractClosure>()) {
    return false;
  }
  auto other_partial = static_cast<const PartialAbstractClosure *>(&other);
  if (fn_ != other_partial->fn_) {
    return false;
  }
  if (args_spec_list_.size() != other_partial->args_spec_list_.size()) {
    return false;
  }
  return args_spec_list_ == other_partial->args_spec_list_;
}

std::size_t PartialAbstractClosure::hash() const {
  MS_EXCEPTION_IF_NULL(fn_);
  auto hash_value = hash_combine(tid(), fn_->hash());
  hash_value = hash_combine(hash_value, AbstractBasePtrListHash(args_spec_list_));
  return hash_value;
}

std::string PartialAbstractClosure::ToString() const {
  std::ostringstream buffer;
  buffer << "PartialAbstractClosure(" << fn_->ToString() << "(";
  for (const auto &arg : args_spec_list_) {
    MS_EXCEPTION_IF_NULL(arg);
    buffer << arg->ToString() << ", ";
  }
  buffer << "))";
  return buffer.str();
}

bool JTransformedAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<JTransformedAbstractClosure>()) {
    return false;
  }
  auto other_transformed = static_cast<const JTransformedAbstractClosure *>(&other);
  return fn_ == other_transformed->fn_;
}

std::size_t JTransformedAbstractClosure::hash() const {
  MS_EXCEPTION_IF_NULL(fn_);
  auto hash_value = hash_combine(tid(), fn_->hash());
  return hash_value;
}

bool VirtualAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<VirtualAbstractClosure>()) {
    return false;
  }
  auto other_virtual = static_cast<const VirtualAbstractClosure *>(&other);
  if (output_ != other_virtual->output_) {
    return false;
  }
  if (args_spec_list_.size() != other_virtual->args_spec_list_.size()) {
    return false;
  }
  return args_spec_list_ == other_virtual->args_spec_list_;
}

std::size_t VirtualAbstractClosure::hash() const {
  MS_EXCEPTION_IF_NULL(output_);
  auto hash_value = hash_combine(tid(), output_->hash());
  hash_value = hash_combine(hash_value, AbstractBasePtrListHash(args_spec_list_));
  return hash_value;
}

std::string VirtualAbstractClosure::ToString() const {
  std::ostringstream buffer;
  buffer << "VirtualAbstractClosure(args: {";
  int64_t i = 0;
  for (const auto &arg : args_spec_list_) {
    MS_EXCEPTION_IF_NULL(arg);
    buffer << "[" << i << "]: " << arg->ToString() << ", ";
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
  auto other_typed = static_cast<const TypedPrimitiveAbstractClosure *>(&other);
  if (output_ != other_typed->output_) {
    return false;
  }
  if (prim_ != other_typed->prim_) {
    return false;
  }
  if (args_spec_list_.size() != other_typed->args_spec_list_.size()) {
    return false;
  }
  return args_spec_list_ == other_typed->args_spec_list_;
}

std::size_t TypedPrimitiveAbstractClosure::hash() const {
  auto hash_value = hash_combine(tid(), prim_->hash());
  hash_value = hash_combine(hash_value, AbstractBasePtrListHash(args_spec_list_));
  return hash_value;
}

std::string TypedPrimitiveAbstractClosure::ToString() const {
  std::ostringstream buffer;
  buffer << "TypedPrimitiveAbstractClosure: primitive: " << prim_->name() << "(args: {";
  int64_t i = 0;
  for (const auto &arg : args_spec_list_) {
    MS_EXCEPTION_IF_NULL(arg);
    buffer << "[" << i << "]: " << arg->ToString() << ", ";
    i++;
  }
  MS_EXCEPTION_IF_NULL(output_);
  buffer << "}, output: " << output_->ToString() << ")";
  return buffer.str();
}

bool PyInterpretAbstractClosure::operator==(const AbstractFunction &other) const {
  if (!other.isa<PyInterpretAbstractClosure>()) {
    return false;
  }
  auto other_partial = static_cast<const PyInterpretAbstractClosure *>(&other);
  if (fn_ != other_partial->fn_) {
    return false;
  }
  if (args_spec_list_.size() != other_partial->args_spec_list_.size()) {
    return false;
  }
  return args_spec_list_ == other_partial->args_spec_list_;
}

std::size_t PyInterpretAbstractClosure::hash() const {
  MS_EXCEPTION_IF_NULL(fn_);
  auto hash_value = hash_combine(tid(), fn_->hash());
  hash_value = hash_combine(hash_value, AbstractBasePtrListHash(args_spec_list_));
  return hash_value;
}

std::string PyInterpretAbstractClosure::ToString() const {
  std::ostringstream buffer;
  buffer << "PyInterpretAbstractClosure(" << fn_->ToString() << "(";
  for (const auto &arg : args_spec_list_) {
    MS_EXCEPTION_IF_NULL(arg);
    buffer << arg->ToString() << ", ";
  }
  buffer << "))";
  return buffer.str();
}

bool DummyAbstractClosure::operator==(const AbstractFunction &other) const {
  return !other.isa<DummyAbstractClosure>();
}
}  // namespace abstract
}  // namespace mindspore
