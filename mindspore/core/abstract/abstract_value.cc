/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "abstract/abstract_value.h"

#include <regex>
#include <algorithm>
#include <utility>

#include "ir/value.h"
#include "utils/hash_map.h"
#include "utils/hashing.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "abstract/utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace abstract {
using mindspore::common::IsEqual;
AbstractBase::TraceNodeProvider AbstractBase::trace_node_provider_ = nullptr;

std::string JoinSupplementaryInfo(const AbstractBasePtr &abstract1, const AbstractBasePtr &abstract2) {
  std::ostringstream oss;
  oss << "#dmsg#Framework Error Message:#dmsg#This: " << abstract1->ToString() << ", other: " << abstract2->ToString();
  // Get trace info of node.
  AnfNodePtr node = nullptr;
  if (AbstractBase::trace_node_provider_ != nullptr) {
    AbstractBase::trace_node_provider_(&node);
  }
  if (node != nullptr) {
    oss << ". Please check the node: " << node->DebugString() << trace::DumpSourceLines(node);
  }
  return oss.str();
}

inline void AbstractTypeJoinLogging(const AbstractBasePtr &abstract1, const AbstractBasePtr &abstract2) {
  std::ostringstream oss;
  oss << "Type Join Failed: Abstract type " << abstract1->type_name() << " cannot join with " << abstract2->type_name()
      << ".\nFor more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed\n";
  oss << JoinSupplementaryInfo(abstract1, abstract2);
  MS_EXCEPTION(TypeError) << oss.str();
}

inline void TypeJoinLogging(const TypePtr &type1, const TypePtr &type2, const AbstractBasePtr &abstract1,
                            const AbstractBasePtr &abstract2) {
  std::ostringstream oss;
  oss << "Type Join Failed: dtype1 = " << type1->ToString() << ", dtype2 = " << type2->ToString()
      << ".\nFor more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed\n";
  oss << JoinSupplementaryInfo(abstract1, abstract2);
  MS_EXCEPTION(TypeError) << oss.str();
}

inline void ShapeJoinLogging(const BaseShapePtr &shape1, const BaseShapePtr &shape2, const AbstractBasePtr &abstract1,
                             const AbstractBasePtr &abstract2) {
  std::ostringstream oss;
  oss << "Shape Join Failed: shape1 = " << shape1->ToString() << ", shape2 = " << shape2->ToString()
      << ".\nFor more details, please refer to https://www.mindspore.cn/search?inputValue=Shape%20Join%20Failed\n";
  oss << JoinSupplementaryInfo(abstract1, abstract2);
  MS_EXCEPTION(ValueError) << oss.str();
}

std::string ExtractLoggingInfo(const std::string &info) {
  // Extract log information based on the keyword "Type Join Failed" or "Shape Join Failed"
  std::regex e("(Type Join Failed|Shape Join Failed).*?\n.*?(Type%20Join%20Failed|Shape%20Join%20Failed)");
  std::smatch result;
  bool found = std::regex_search(info, result, e);
  if (found) {
    return result.str();
  }
  return "";
}

static inline bool IsUndeterminedType(const TypePtr &type) {
  return (type != nullptr) && (type->type_id() == kObjectTypeUndeterminedType);
}

bool AbstractBase::operator==(const AbstractBase &other) const {
  if (this == &other) {
    // Same object.
    return true;
  }
  // Check C++ type.
  if (tid() != other.tid()) {
    return false;
  }
  // If both are "undetermined" type, they are considered equal.
  if (IsUndeterminedType(BuildType()) && IsUndeterminedType(other.BuildType())) {
    return true;
  }
  // Check data type, shape and value.
  return IsEqual(type_, other.type_) && IsEqual(shape_, other.shape_) && IsEqual(value_, other.value_);
}

ValuePtr AbstractBase::BuildValue() const {
  if (value_ == nullptr) {
    return RealBuildValue();
  }
  return value_;
}

AbstractBasePtr AbstractBase::Broaden() const {
  AbstractBasePtr clone = Clone();
  MS_EXCEPTION_IF_NULL(clone);
  clone->set_value(kAnyValue);
  return clone;
}

AbstractBasePtr AbstractBase::PartialBroaden() const { return Clone(); }

std::string AbstractBase::ToString() const {
  std::ostringstream buffer;
  std::string value = std::string("value is null");
  if (value_ != nullptr) {
    value = value_->ToString();
  }
  MS_EXCEPTION_IF_NULL(type_);
  MS_EXCEPTION_IF_NULL(shape_);
  buffer << type_name() << "("
         << "Type: " << type_->ToString() << ", Value: " << value << ", Shape: " << shape_->ToString() << ")";
  return buffer.str();
}

std::string AbstractBase::ToString(bool verbose) const {
  if (verbose) {
    return ToString();
  }
  std::ostringstream buffer;
  auto tensor_value = BuildValue();
  auto shape = BuildShape();
  auto type = BuildType();
  if (shape != nullptr && type != nullptr) {
    buffer << type << ", " << shape->ToString();
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      buffer << ", value=...";
    }
  } else if (type != nullptr) {
    buffer << type;
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      buffer << ", value=...";
    }
  }
  return buffer.str();
}

AbstractBasePtr AbstractScalar::Broaden() const {
  if (is_variable_) {
    return AbstractBase::Broaden();
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR)) {
    return AbstractBase::Broaden();
  }
  auto type_id = GetTypeTrack()->type_id();
  if (type_id == kObjectTypeEnvType) {
    return AbstractBase::Broaden();
  }
  return Clone();
}

AbstractBasePtr AbstractScalar::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  if (*this == *other) {
    return shared_from_base<AbstractBase>();
  }
  const auto &type_self = GetTypeTrack();
  const auto &type_other = other->GetTypeTrack();
  TypePtr res_type = TypeJoin(type_self, type_other);
  if (res_type == kAnyType) {
    TypeJoinLogging(type_self, type_other, shared_from_base<AbstractBase>(), other);
  }
  const auto &value_self = GetValueTrack();
  const auto &value_other = other->GetValueTrack();
  ValuePtr res_value = ValueJoin(value_self, value_other);
  if (res_value == value_self) {
    return shared_from_base<AbstractBase>();
  }
  return std::make_shared<AbstractScalar>(res_value, res_type);
}

AbstractBasePtr AbstractType::Clone() const {
  ValuePtr value_self = GetValueTrack();
  if (value_self == nullptr || !value_self->isa<Type>()) {
    return nullptr;
  }
  auto type_self = value_self->cast_ptr<Type>();
  return std::make_shared<AbstractType>(type_self->Clone());
}

bool AbstractType::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  return tid() == other.tid() &&
         IsEqual(dyn_cast_ptr<Type>(GetValueTrack()), dyn_cast_ptr<Type>(other.GetValueTrack()));
}

std::string AbstractType::ToString() const {
  std::ostringstream buffer;
  ValuePtr value_self = GetValueTrack();
  if (value_self == nullptr) {
    buffer << "AbstractType value: nullptr";
    return buffer.str();
  }
  if (!value_self->isa<Type>()) {
    buffer << type_name() << "(Value: nullptr)";
    return buffer.str();
  }
  auto type_self = value_self->cast_ptr<Type>();
  buffer << type_name() << "("
         << "Value: " << type_self->ToString() << ")";
  return buffer.str();
}

std::string AbstractError::ToString() const {
  std::ostringstream buffer;
  auto value_track = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  MS_EXCEPTION_IF_NULL(node_);
  buffer << type_name() << "("
         << "Value: " << value_track->ToString() << ", Node: " << node_->DebugString() << ")";
  return buffer.str();
}

AbstractBasePtr AbstractFunction::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  auto other_func = dyn_cast<AbstractFunction>(other);
  if (other_func == nullptr) {
    AbstractTypeJoinLogging(shared_from_base<AbstractBase>(), other);
  }
  return Join(other_func);
}

bool AbstractFunction::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractFunction>()) {
    return false;
  }
  return *this == static_cast<const AbstractFunction &>(other);
}

namespace {
void CollectSequenceNodes(const AnfNodeWeakPtrList &source_sequence_nodes, AnfNodeWeakPtrList *sequence_nodes_ptr) {
  AnfNodeWeakPtrList &sequence_nodes = *sequence_nodes_ptr;
  auto sequence_nodes_size = source_sequence_nodes.size();
  for (size_t i = 0; i < sequence_nodes_size; ++i) {
    // Lock sequence nodes of this.
    auto &source_weak_node = source_sequence_nodes[i];
    auto source_sequence_node = source_weak_node.lock();
    if (source_sequence_node == nullptr) {
      continue;
    }
    // Check and emplace sequence node for this.
    auto this_iter = std::find_if(
      sequence_nodes.begin(), sequence_nodes.end(),
      [&source_sequence_node](const AnfNodeWeakPtr &weak_node) { return source_sequence_node == weak_node.lock(); });
    if (this_iter == sequence_nodes.end()) {
      (void)sequence_nodes.emplace_back(AnfNodeWeakPtr(source_sequence_node));
    }
  }
}

void SynchronizeSequenceNodesElementsUseFlagsInner(const AnfNodeWeakPtrList &sequence_nodes) {
  // Choose the candidate sequence node, that we use its flags as unique one.
  AnfNodePtr candidate_sequence_node = sequence_nodes[0].lock();
  MS_EXCEPTION_IF_NULL(candidate_sequence_node);
  size_t candidate_index = 0;
  for (size_t i = 1; i < sequence_nodes.size(); ++i) {
    auto current_sequence_node = sequence_nodes[i].lock();
    MS_EXCEPTION_IF_NULL(current_sequence_node);
    if (candidate_sequence_node == current_sequence_node) {
      continue;
    }
    auto candidate_flags = GetSequenceNodeElementsUseFlags(candidate_sequence_node);
    MS_EXCEPTION_IF_NULL(candidate_flags);
    auto current_flags = GetSequenceNodeElementsUseFlags(current_sequence_node);
    MS_EXCEPTION_IF_NULL(current_flags);
    if (candidate_flags == current_flags) {
      continue;
    }

    // Find the sequence node whose flags are most used.
    auto candidate_count = candidate_flags.use_count();
    auto current_count = current_flags.use_count();
    if (candidate_count < current_count) {
      candidate_sequence_node = current_sequence_node;
      candidate_index = i;
    }
  }

  // Synchronize the elements use flags for all sequence nodes with candidate sequence node.
  // We set the same 'elements_use_flags' for them after here.
  auto candidate_flags = GetSequenceNodeElementsUseFlags(candidate_sequence_node);
  MS_LOG(DEBUG) << "Sequence nodes size: " << sequence_nodes.size() << ", candidate node index: " << candidate_index
                << ", candidate node: " << candidate_sequence_node->DebugString() << ", flags: " << candidate_flags;
  for (size_t i = 0; i < sequence_nodes.size(); ++i) {
    auto current_sequence_node = sequence_nodes[i].lock();
    MS_EXCEPTION_IF_NULL(current_sequence_node);
    if (candidate_sequence_node == current_sequence_node) {
      continue;
    }
    auto current_flags = GetSequenceNodeElementsUseFlags(current_sequence_node);
    if (candidate_flags == current_flags) {
      continue;
    }

    // Merge the use flags, set true if either is true.
    for (size_t j = 0; j < candidate_flags->size(); ++j) {
      MS_LOG(DEBUG) << "Check elements_use_flags[" << j << "], this_flag: " << (*candidate_flags)[j]
                    << ", other_flag: " << (*current_flags)[j];
      (*candidate_flags)[j] = ((*candidate_flags)[j] || (*current_flags)[j]);
    }
    // Use the candidate sequence node flags.
    SetSequenceNodeElementsUseFlags(current_sequence_node, candidate_flags);
    MS_LOG(DEBUG) << "Reset flags for sequence node[" << i << "]: " << current_sequence_node->DebugString()
                  << ", flags: " << candidate_flags;
  }
}

void CheckSequenceNodesValid(const AnfNodeWeakPtrList &sequence_nodes) {
  if (!IS_OUTPUT_ON(MsLogLevel::kDebug)) {
    return;
  }
  if (sequence_nodes.size() <= 1) {
    return;
  }
  AnfNodePtr candidate_sequence_node = sequence_nodes[0].lock();
  if (candidate_sequence_node == nullptr) {
    MS_LOG(ERROR) << "candidate_sequence_node is null.";
    return;
  }
  auto candidate_flags = GetSequenceNodeElementsUseFlags(candidate_sequence_node);
  if (candidate_flags == nullptr) {
    MS_LOG(ERROR) << "The candidate_flags is null, sequence_nodes[0]: " << candidate_sequence_node->DebugString();
    return;
  }
  for (size_t i = 0; i < sequence_nodes.size(); ++i) {
    auto current_sequence_node = sequence_nodes[i].lock();
    if (current_sequence_node == nullptr) {
      MS_LOG(ERROR) << "current_sequence_node is null.";
      return;
    }
    MS_LOG(DEBUG) << "sequence_nodes[" << i << "]: " << current_sequence_node << "/"
                  << current_sequence_node->DebugString()
                  << ", flags: " << GetSequenceNodeElementsUseFlags(current_sequence_node);
  }
  for (size_t i = 1; i < sequence_nodes.size(); ++i) {
    auto current_sequence_node = sequence_nodes[i].lock();
    if (current_sequence_node == nullptr) {
      MS_LOG(ERROR) << "current_sequence_node is null.";
      return;
    }
    if (candidate_sequence_node == current_sequence_node) {
      continue;
    }
    candidate_flags = GetSequenceNodeElementsUseFlags(candidate_sequence_node);
    MS_EXCEPTION_IF_NULL(candidate_flags);
    auto current_flags = GetSequenceNodeElementsUseFlags(current_sequence_node);
    MS_EXCEPTION_IF_NULL(current_flags);
    if (candidate_flags == current_flags) {
      continue;
    }
    MS_LOG(ERROR) << "Should use same flags pointer, candidate_node: " << candidate_sequence_node->DebugString()
                  << ", current_node: " << current_sequence_node->DebugString();

    if (candidate_flags->size() != current_flags->size()) {
      MS_LOG(EXCEPTION) << "Flag not same size";
    }
    for (size_t j = 0; j < candidate_flags->size(); ++j) {
      if ((*candidate_flags)[j] != (*current_flags)[j]) {
        MS_LOG(EXCEPTION) << "Not equal elements_use_flags[" << j << "], this_flag: " << (*candidate_flags)[j]
                          << ", other_flag: " << (*current_flags)[j];
      }
    }
  }
}

AnfNodeWeakPtrList SynchronizeSequenceNodesElementsUseFlags(const AnfNodeWeakPtrList &lhs_sequence_nodes,
                                                            const AnfNodeWeakPtrList &rhs_sequence_nodes) {
  // Collect this and other sequence nodes.
  AnfNodeWeakPtrList sequence_nodes;
  CollectSequenceNodes(lhs_sequence_nodes, &sequence_nodes);
  CollectSequenceNodes(rhs_sequence_nodes, &sequence_nodes);
  if (sequence_nodes.size() <= 1) {
    MS_LOG(DEBUG) << "Sequence nodes size should exceed 1.";
    return sequence_nodes;
  }
  // Synchronize the elements use flags for all sequence nodes.
  SynchronizeSequenceNodesElementsUseFlagsInner(sequence_nodes);
  CheckSequenceNodesValid(sequence_nodes);
  return sequence_nodes;
}
}  // namespace

AbstractSequence::AbstractSequence(AbstractBasePtrList &&elements,
                                   const std::shared_ptr<AnfNodeWeakPtrList> &sequence_nodes)
    : elements_(std::move(elements)), sequence_nodes_(sequence_nodes) {
  if (sequence_nodes != nullptr) {
    CheckSequenceNodesValid(*sequence_nodes);
  }
}

AbstractSequence::AbstractSequence(const AbstractBasePtrList &elements,
                                   const std::shared_ptr<AnfNodeWeakPtrList> &sequence_nodes)
    : elements_(elements), sequence_nodes_(sequence_nodes) {
  if (sequence_nodes != nullptr) {
    CheckSequenceNodesValid(*sequence_nodes);
  }
}

const AbstractBasePtr AbstractSequence::operator[](const std::size_t &dim) const {
  if (dynamic_len_) {
    MS_LOG(EXCEPTION) << "Can not get element from dynamic length sequence " << ToString();
  }
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "Index [" << dim << "] Out of the size [" << size() << "] of the list.";
  }
  return elements_[dim];
}

std::string AbstractSequence::ToStringInternal() const {
  std::ostringstream buffer;
  size_t i = 0;
  size_t size = elements_.size();
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    buffer << "element[" << i << "]: " << element->ToString();
    if (i < size - 1) {
      buffer << ", ";
    }
    i++;
  }
  return buffer.str();
}

std::string AbstractSequence::ToString() const {
  std::stringstream ss;
  ss << type_name();
  ss << "{";
  ss << ToStringInternal();
  if (!dynamic_len_ && sequence_nodes() != nullptr && !sequence_nodes()->empty()) {
    ss << ", sequence_nodes: {";
    for (size_t i = 0; i < sequence_nodes()->size(); ++i) {
      auto sequence_node = (*sequence_nodes())[i].lock();
      if (sequence_node == nullptr) {
        ss << "<freed node>";
        continue;
      } else {
        ss << sequence_node->DebugString();
      }
      auto flags = GetSequenceNodeElementsUseFlags(sequence_node);
      if (flags != nullptr) {
        ss << ", elements_use_flags: {ptr: " << flags << ", value: " << (*flags) << "}";
      }
      if (i != sequence_nodes()->size() - 1) {
        ss << ", ";
      }
    }
    ss << "}";
  }
  ss << ", dynamic_len:" << dynamic_len_;
  ss << "}";
  return ss.str();
}

std::string AbstractSequence::ToString(bool verbose) const {
  if (verbose) {
    return ToString();
  }
  std::ostringstream buffer;
  size_t i = 0;
  size_t size = elements_.size();
  buffer << type_name() << " {";
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    buffer << element->ToString(false);
    if (i < size - 1) {
      buffer << ", ";
    }
    i++;
  }
  buffer << "}";
  return buffer.str();
}

AnfNodeWeakPtrList AbstractSequence::SequenceNodesJoin(const AbstractBasePtr &other) {
  AnfNodeWeakPtrList sequence_nodes;
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (!enable_eliminate_unused_element || this->sequence_nodes() == nullptr) {
    return sequence_nodes;
  }
  auto other_sequence = dyn_cast_ptr<AbstractSequence>(other);
  if (other_sequence == nullptr) {
    return sequence_nodes;
  }
  auto this_sequence_nodes_size = (this->sequence_nodes() == nullptr ? 0 : this->sequence_nodes()->size());
  auto other_sequence_nodes_size =
    (other_sequence->sequence_nodes() == nullptr ? 0 : other_sequence->sequence_nodes()->size());
  if (this_sequence_nodes_size == 0 || other_sequence_nodes_size == 0) {
    return sequence_nodes;
  }
  // Collect this and other sequence nodes.
  if (this->sequence_nodes() != nullptr) {
    CollectSequenceNodes(*this->sequence_nodes(), &sequence_nodes);
  }
  if (other_sequence->sequence_nodes() != nullptr) {
    CollectSequenceNodes(*other_sequence->sequence_nodes(), &sequence_nodes);
  }
  if (sequence_nodes.empty()) {
    MS_LOG(INFO) << "Sequence nodes size should not be empty.";
    return sequence_nodes;
  }
  // Synchronize the elements use flags for all sequence nodes.
  SynchronizeSequenceNodesElementsUseFlagsInner(sequence_nodes);

  CheckSequenceNodesValid(sequence_nodes);
  this->InsertSequenceNodes(sequence_nodes);
  other_sequence->InsertSequenceNodes(sequence_nodes);
  return sequence_nodes;
}

void SynchronizeSequenceElementsUseFlagsRecursively(const AbstractSequencePtr &lhs_sequence,
                                                    const AbstractSequencePtr &rhs_sequence) {
  if (lhs_sequence->sequence_nodes() == nullptr || rhs_sequence->sequence_nodes() == nullptr) {
    return;
  }
  auto sequence_nodes =
    SynchronizeSequenceNodesElementsUseFlags(*lhs_sequence->sequence_nodes(), *rhs_sequence->sequence_nodes());
  lhs_sequence->InsertSequenceNodes(sequence_nodes);
  rhs_sequence->InsertSequenceNodes(sequence_nodes);
  if (lhs_sequence->elements().size() != rhs_sequence->elements().size()) {
    MS_LOG(EXCEPTION) << "The elements size should be equal, " << lhs_sequence->ToString() << ", "
                      << rhs_sequence->ToString();
  }
  for (size_t i = 0; i < lhs_sequence->elements().size(); ++i) {
    auto lhs_inner_sequence = dyn_cast<AbstractSequence>(lhs_sequence->elements()[i]);
    if (lhs_inner_sequence == nullptr) {
      continue;
    }
    auto rhs_inner_sequence = dyn_cast<AbstractSequence>(rhs_sequence->elements()[i]);
    if (rhs_inner_sequence == nullptr) {
      continue;
    }
    SynchronizeSequenceElementsUseFlagsRecursively(lhs_inner_sequence, rhs_inner_sequence);
  }
}

void AbstractSequence::InsertSequenceNodes(const AnfNodeWeakPtrList &sequence_nodes) {
  if (dynamic_len_) {
    MS_LOG(EXCEPTION) << "Can not insert sequence nodes for dynamic length sequence " << ToString();
  }
  if (sequence_nodes_ == nullptr) {
    MS_LOG(DEBUG) << "The sequence_nodes is null.";
    sequence_nodes_ = std::make_shared<AnfNodeWeakPtrList>();
  }
  for (auto &weak_node : sequence_nodes) {
    auto sequence_node = weak_node.lock();
    InsertSequenceNode(sequence_node);
  }
}

void AbstractSequence::InsertSequenceNode(const AnfNodePtr &sequence_node) {
  if (dynamic_len_) {
    MS_LOG(EXCEPTION) << "Can not insert sequence node for dynamic length sequence " << ToString();
  }
  if (sequence_nodes_ == nullptr) {
    MS_LOG(DEBUG) << "The sequence_nodes is null.";
    sequence_nodes_ = std::make_shared<AnfNodeWeakPtrList>();
  }
  auto iter =
    std::find_if(sequence_nodes_->begin(), sequence_nodes_->end(),
                 [&sequence_node](const AnfNodeWeakPtr &weak_node) { return sequence_node == weak_node.lock(); });
  if (iter == sequence_nodes_->end()) {
    (void)sequence_nodes_->emplace_back(sequence_node);
    CheckSequenceNodesValid(*sequence_nodes_);
  } else {
    MS_LOG(DEBUG) << "Fail to insert node \'" << sequence_node->DebugString() << "\' into sequence nodes.";
  }
}

void AbstractSequence::UpdateSequenceNode(const AnfNodePtr &old_sequence_node, const AnfNodePtr &new_sequence_node) {
  if (dynamic_len_) {
    MS_LOG(EXCEPTION) << "Can not update sequence node for dynamic length sequence " << ToString();
  }
  if (sequence_nodes_ == nullptr) {
    MS_LOG(DEBUG) << "The sequence_nodes is null.";
    sequence_nodes_ = std::make_shared<AnfNodeWeakPtrList>();
  }
  auto iter = std::find_if(
    sequence_nodes_->begin(), sequence_nodes_->end(),
    [&old_sequence_node](const AnfNodeWeakPtr &weak_node) { return old_sequence_node == weak_node.lock(); });
  if (iter != sequence_nodes_->end()) {
    *iter = new_sequence_node;
    CheckSequenceNodesValid(*sequence_nodes_);
    return;
  }
  MS_LOG(EXCEPTION) << "Not found old node \'" << old_sequence_node->DebugString() << "\' in sequence nodes.";
}

bool AbstractSequence::PurifyElements() {
  if (dynamic_len_ || sequence_nodes_ == nullptr || sequence_nodes_->empty()) {
    return true;
  }
  // Just use any sequence node's elements_use_flags.
  AnfNodePtr not_free_node = nullptr;
  std::shared_ptr<std::vector<bool>> elements_use_flags_ptr = nullptr;
  for (auto &weak_node : *sequence_nodes_) {
    auto sequence_node = weak_node.lock();
    if (sequence_node == nullptr) {
      MS_LOG(DEBUG) << "The node in sequence_nodes is free.";
      continue;
    }
    not_free_node = sequence_node;
    auto flags = GetSequenceNodeElementsUseFlags(sequence_node);
    if (flags != nullptr) {
      elements_use_flags_ptr = flags;
      break;
    }
  }
  if (elements_use_flags_ptr == nullptr) {
    if (not_free_node == nullptr) {
      MS_LOG(ERROR) << "Check if all sequence nodes are released, or none elements use flags in them. nodes size: "
                    << sequence_nodes_->size();
    } else {
      MS_LOG(ERROR) << "Check if none elements use flags in sequence ndoes. one of node: "
                    << not_free_node->DebugString();
    }
    return false;
  }

  // Purify the elements.
  auto &elements_use_flags = *elements_use_flags_ptr;
  if (elements_use_flags.size() < elements_.size()) {
    MS_LOG(EXCEPTION) << "Elements size should not be greater to elements use flags size. " << ToString();
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(elements_[i]);
    if (!elements_use_flags[i]) {
      const auto unuse_node_none = std::make_shared<AbstractScalar>(std::make_shared<Int32Imm>(0));
      if (elements_[i]->isa<AbstractError>()) {
        unuse_node_none->set_type(std::make_shared<Problem>());
      }
      elements_[i] = unuse_node_none;
      MS_LOG(DEBUG) << "Erase elements[" << i << "] abstract as Zero for " << ToString();
    } else {
      MS_LOG(DEBUG) << "Keep elements[" << i << "] abstract as " << elements_[i]->ToString();
    }
  }
  return true;
}

void AbstractSequence::CheckAndConvertToDynamicLenSequence() {
  const size_t input_len = size();
  if (input_len > 1) {
    auto first_element = elements()[0];
    MS_EXCEPTION_IF_NULL(first_element);
    auto first_element_shape = first_element->BuildShape();
    MS_EXCEPTION_IF_NULL(first_element_shape);
    auto first_element_type_id = first_element->BuildType()->generic_type_id();
    for (size_t i = 0; i < input_len; ++i) {
      auto cur_element = elements()[i];
      MS_EXCEPTION_IF_NULL(cur_element);
      auto cur_element_type_id = cur_element->BuildType()->generic_type_id();
      if (first_element_type_id != cur_element_type_id) {
        MS_EXCEPTION(ValueError) << "In graph mode, the element type of dynamic length array must be the same."
                                 << "The element type do not match, can not convert to dynamic length sequence. "
                                 << "The 0th element type is: " << TypeIdToString(first_element_type_id) << ". The "
                                 << i << "th element type is: " << TypeIdToString(cur_element_type_id);
      }
      auto cur_element_shape = cur_element->BuildShape();
      MS_EXCEPTION_IF_NULL(cur_element_shape);
      if (*first_element_shape != *cur_element_shape) {
        MS_EXCEPTION(ValueError) << "In graph mode, the element shape of dynamic length array must be the same."
                                 << "The element shape do not match, can not convert to dynamic length sequence. "
                                 << "The 0th element shape is: " << first_element_shape->ToString() << ". The " << i
                                 << "th element shape is: " << cur_element_shape->ToString();
      }
    }
    set_dynamic_len_element_abs(first_element);
  } else if (input_len == 1) {
    set_dynamic_len_element_abs(elements()[0]);
  }
  set_dynamic_len(true);
}

TypePtrList AbstractSequence::ElementsType() const {
  TypePtrList element_type_list;
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    TypePtr element_type = element->BuildType();
    element_type_list.push_back(element_type);
  }
  return element_type_list;
}

BaseShapePtrList AbstractSequence::ElementsShape() const {
  BaseShapePtrList element_shape_list;
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    BaseShapePtr element_shape = element->BuildShape();
    element_shape_list.push_back(element_shape);
  }
  return element_shape_list;
}

AbstractBasePtrList AbstractSequence::ElementsClone() const {
  AbstractBasePtrList element_list;
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    AbstractBasePtr clone = element->Clone();
    element_list.push_back(clone);
  }
  return element_list;
}

AbstractBasePtrList AbstractSequence::ElementsBroaden() const {
  AbstractBasePtrList element_list;
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    AbstractBasePtr broadend = element->Broaden();
    element_list.push_back(broadend);
  }
  return element_list;
}

AbstractBasePtrList AbstractSequence::ElementsPartialBroaden() const {
  AbstractBasePtrList element_list;
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    AbstractBasePtr broadend = element->PartialBroaden();
    element_list.push_back(broadend);
  }
  return element_list;
}

std::pair<bool, ValuePtr> GetValueFromUserData(const AbstractBasePtr &element_abs) {
  MS_EXCEPTION_IF_NULL(element_abs);
  if (abstract::AbstractBase::pyexecute_user_data_catcher()) {
    return abstract::AbstractBase::pyexecute_user_data_catcher()(element_abs);
  }
  return {false, nullptr};
}

template <typename T>
ValuePtr AbstractSequence::ElementsBuildValue() const {
  std::vector<ValuePtr> element_value_list;
  for (const auto &element : elements_) {
    MS_EXCEPTION_IF_NULL(element);
    auto [has_user_data, element_value] = GetValueFromUserData(element);
    if (has_user_data && element_value != nullptr) {
      element_value_list.push_back(element_value);
      continue;
    }
    element_value = element->BuildValue();
    MS_EXCEPTION_IF_NULL(element_value);
    if (element_value->isa<AnyValue>()) {
      return kAnyValue;
    }
    element_value_list.push_back(element_value);
  }
  return std::make_shared<T>(element_value_list);
}
template MS_CORE_API ValuePtr AbstractSequence::ElementsBuildValue<ValueTuple>() const;
template MS_CORE_API ValuePtr AbstractSequence::ElementsBuildValue<ValueList>() const;

template <typename T>
AbstractBasePtr AbstractSequence::ElementsJoin(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  auto other_sequeue = dyn_cast_ptr<T>(other);
  if (other_sequeue == nullptr) {
    AbstractTypeJoinLogging(shared_from_base<AbstractBase>(), other);
  }
  auto joined_list = AbstractJoin(elements_, other_sequeue->elements_);
  bool changes = false;
  for (std::size_t i = 0; i < elements_.size(); i++) {
    if (elements_[i] != joined_list[i]) {
      changes = true;
      break;
    }
  }
  if (!changes) {
    return shared_from_base<AbstractBase>();
  }
  return std::make_shared<T>(joined_list);
}
template AbstractBasePtr AbstractSequence::ElementsJoin<AbstractList>(const AbstractBasePtr &);
template AbstractBasePtr AbstractSequence::ElementsJoin<AbstractTuple>(const AbstractBasePtr &);

std::size_t AbstractSequence::hash() const {
  if (dynamic_len_) {
    size_t hash_val = hash_combine(tid(), static_cast<size_t>(dynamic_len_));
    if (dynamic_len_element_abs_ != nullptr) {
      return hash_combine(hash_val, static_cast<size_t>(dynamic_len_element_abs_->hash()));
    }
    return hash_val;
  }
  return hash_combine(tid(), AbstractBasePtrListHash(elements_));
}

std::size_t AbstractSequence::size() const {
  if (dynamic_len_) {
    if (dynamic_len_element_abs_ == nullptr) {
      return 0;
    }
    MS_LOG(EXCEPTION) << "Can not get size for dynamic length sequence " << ToString();
  }
  return elements_.size();
}

bool AbstractSequence::empty() const {
  if (dynamic_len_) {
    if (dynamic_len_element_abs_ == nullptr) {
      return true;
    }
    MS_LOG(EXCEPTION) << "Can not call function empty() for dynamic length sequence " << ToString();
  }
  return elements_.empty();
}

void AbstractSequence::set_dynamic_len(bool dynamic_len) {
  if (dynamic_len) {
    sequence_nodes_ = nullptr;
  }
  dynamic_len_ = dynamic_len;
}

void AbstractSequence::set_dynamic_len_element_abs(const AbstractBasePtr &dynamic_len_element_abs) {
  if (dynamic_len_element_abs == nullptr) {
    return;
  }
  if (dynamic_len_element_abs->isa<abstract::AbstractDictionary>()) {
    MS_EXCEPTION(TypeError) << "DynamicSequence does not support dictionary type as element type now.";
  }
  // dynamic_len_element_abs should ignore value.
  dynamic_len_element_abs_ = AbstractBroaden(dynamic_len_element_abs);
}

bool AbstractSequence::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (tid() != other.tid()) {
    return false;
  }
  const auto &other_sequence = dynamic_cast<const AbstractSequence &>(other);
  if (dynamic_len_ != other_sequence.dynamic_len()) {
    // Variable length sequence and constant length sequence can not be the same.
    return false;
  }

  if (dynamic_len_) {
    // If the abstract of element for two variable sequence is the same, these two sequence is the same.
    return IsEqual(dynamic_len_element_abs_, other_sequence.dynamic_len_element_abs());
  }

  if (elements_.size() != other_sequence.elements_.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    if (!IsEqual(elements_[i], other_sequence.elements_[i])) {
      return false;
    }
  }
  return true;
}

TypePtr AbstractTuple::BuildType() const {
  auto ret = std::make_shared<Tuple>(ElementsType());
  if (dynamic_len_) {
    ret->set_dynamic_len(dynamic_len_);
    if (dynamic_len_element_abs_ != nullptr) {
      ret->set_dynamic_element_type(dynamic_len_element_abs_->BuildType());
    }
  }
  return ret;
}

BaseShapePtr AbstractTuple::BuildShape() const {
  if (dynamic_len_) {
    return kDynamicSequenceShape;
  }
  return std::make_shared<TupleShape>(ElementsShape());
}

AbstractBasePtr AbstractTuple::Clone() const {
  auto ret = std::make_shared<AbstractTuple>(ElementsClone(), sequence_nodes());
  ret->set_dynamic_len(dynamic_len_);
  ret->set_dynamic_len_element_abs(dynamic_len_element_abs_);
  return ret;
}

AbstractBasePtr AbstractTuple::Broaden() const {
  auto ret = std::make_shared<AbstractTuple>(ElementsBroaden(), sequence_nodes());
  ret->set_dynamic_len(dynamic_len_);
  ret->set_dynamic_len_element_abs(dynamic_len_element_abs_);
  return ret;
}

AbstractBasePtr AbstractTuple::PartialBroaden() const {
  auto ret = std::make_shared<AbstractTuple>(ElementsPartialBroaden(), sequence_nodes());
  ret->set_dynamic_len(dynamic_len_);
  ret->set_dynamic_len_element_abs(dynamic_len_element_abs_);
  return ret;
}

ValuePtr AbstractTuple::RealBuildValue() const {
  if (dynamic_len_) {
    return kAnyValue;
  }
  return ElementsBuildValue<ValueTuple>();
}

void AbstractTuple::set_shape(const BaseShapePtr &shape) {
  auto tuple_shape = dyn_cast_ptr<TupleShape>(shape);
  MS_EXCEPTION_IF_NULL(tuple_shape);
  if (tuple_shape->shape().size() != elements_.size()) {
    MS_LOG(EXCEPTION) << "Size mismatch: " << tuple_shape->shape().size() << " vs " << elements_.size();
  }

  for (size_t i = 0; i < elements_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(elements_[i]);
    elements_[i]->set_shape(tuple_shape->shape()[i]);
  }
}

bool AbstractTuple::ContainsAllBroadenTensors() const {
  if (dynamic_len_) {
    if (dynamic_len_element_abs_ != nullptr && dynamic_len_element_abs_->isa<AbstractTensor>()) {
      return true;
    }
    return false;
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    if (!(elements_[i]->isa<abstract::AbstractUndetermined>() && elements_[i]->IsBroaden()) &&
        !(elements_[i]->isa<abstract::AbstractTuple>() &&
          elements_[i]->cast_ptr<abstract::AbstractTuple>()->ContainsAllBroadenTensors())) {
      return false;
    }
  }
  return true;
}

bool AbstractTuple::ContainsAllConstants() const {
  for (const auto &element : elements_) {
    auto element_value = element->BuildValue();
    MS_EXCEPTION_IF_NULL(element_value);
    // Check if tuple contains only constants, i.e. string, number, constant tensor and tuple.
    if (!(element_value->isa<StringImm>() || element_value->isa<Scalar>() ||
          (element->isa<abstract::AbstractTensor>() && element_value != kAnyValue) ||
          element->isa<abstract::AbstractTuple>())) {
      return false;
    }
    // Check if inner tuple contains only constants recursively.
    if (element->isa<abstract::AbstractTuple>() &&
        !element->cast_ptr<abstract::AbstractTuple>()->ContainsAllConstants()) {
      return false;
    }
  }
  return true;
}

bool AbstractTuple::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractTuple>()) {
    return false;
  }
  return AbstractSequence::operator==(static_cast<const AbstractSequence &>(other));
}

bool AbstractList::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractList>()) {
    return false;
  }
  return AbstractSequence::operator==(static_cast<const AbstractSequence &>(other));
}

TypePtr AbstractList::BuildType() const {
  auto ret = std::make_shared<List>(ElementsType());
  if (dynamic_len_) {
    ret->set_dynamic_len(dynamic_len_);
    if (dynamic_len_element_abs_ != nullptr) {
      ret->set_dynamic_element_type(dynamic_len_element_abs_->BuildType());
    }
  }
  return ret;
}

BaseShapePtr AbstractList::BuildShape() const {
  if (dynamic_len_) {
    return kDynamicSequenceShape;
  }
  return std::make_shared<ListShape>(ElementsShape());
}

AbstractBasePtr AbstractList::Clone() const {
  auto ret = std::make_shared<AbstractList>(ElementsClone(), sequence_nodes());
  ret->set_dynamic_len(dynamic_len_);
  ret->set_dynamic_len_element_abs(dynamic_len_element_abs_);
  return ret;
}

AbstractBasePtr AbstractList::Broaden() const {
  auto ret = std::make_shared<AbstractList>(ElementsBroaden(), sequence_nodes());
  ret->set_dynamic_len(dynamic_len_);
  ret->set_dynamic_len_element_abs(dynamic_len_element_abs_);
  return ret;
}

AbstractBasePtr AbstractList::PartialBroaden() const {
  auto ret = std::make_shared<AbstractList>(ElementsPartialBroaden(), sequence_nodes());
  ret->set_dynamic_len(dynamic_len_);
  ret->set_dynamic_len_element_abs(dynamic_len_element_abs_);
  return ret;
}

ValuePtr AbstractList::RealBuildValue() const {
  if (dynamic_len_) {
    return kAnyValue;
  }
  return ElementsBuildValue<ValueList>();
}

template <typename T>
AbstractBasePtr AbstractSequence::DynamicLenSequenceJoin(const AbstractBasePtr &other) {
  auto other_seq = dyn_cast_ptr<T>(other);
  if (other_seq == nullptr) {
    MS_LOG(EXCEPTION) << "Can not join AbstractTuple with AbstractList, the first abstract is: " << ToString()
                      << " and the second abstract is: " << other->ToString();
  }
  if (!dynamic_len_ || !other_seq->dynamic_len()) {
    MS_LOG(EXCEPTION) << "Can not join dynamic length sequence with constant length Sequence.";
  }
  auto element_abs1 = dynamic_len_element_abs_;
  auto element_abs2 = other_seq->dynamic_len_element_abs();
  AbstractBasePtr join_element_abs = nullptr;

  // When two element abstracts are not nullptr, join them to get the new element abstract.
  // When one or none of the element abstract is nullptr, the result element abstract is nullptr.
  if (element_abs1 != nullptr && element_abs2 != nullptr) {
    join_element_abs = element_abs1->Join(element_abs2);
  }
  auto ret = Clone();
  ret->cast<AbstractSequencePtr>()->set_dynamic_len_element_abs(join_element_abs);
  return ret;
}

AbstractBasePtr AbstractTuple::Join(const AbstractBasePtr &other) {
  if (dynamic_len_) {
    return DynamicLenSequenceJoin<AbstractTuple>(other);
  }
  auto res = dyn_cast<AbstractSequence>(ElementsJoin<AbstractTuple>(other));
  MS_EXCEPTION_IF_NULL(res);
  res->InsertSequenceNodes(SequenceNodesJoin(other));
  return res;
}

AbstractBasePtr AbstractList::Join(const AbstractBasePtr &other) {
  if (dynamic_len_) {
    return DynamicLenSequenceJoin<AbstractList>(other);
  }
  auto res = dyn_cast<AbstractSequence>(ElementsJoin<AbstractList>(other));
  MS_EXCEPTION_IF_NULL(res);
  res->InsertSequenceNodes(SequenceNodesJoin(other));
  return res;
}

TypePtr AbstractSlice::BuildType() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  TypePtr start = start_->BuildType();
  TypePtr stop = stop_->BuildType();
  TypePtr step = step_->BuildType();
  return std::make_shared<Slice>(start, stop, step);
}

bool AbstractSlice::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractSlice>()) {
    return false;
  }
  const auto &other_slice = dynamic_cast<const AbstractSlice &>(other);
  return IsEqual(start_, other_slice.start_) && IsEqual(stop_, other_slice.stop_) && IsEqual(step_, other_slice.step_);
}

AbstractBasePtr AbstractSlice::Clone() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  AbstractBasePtr start = start_->Clone();
  AbstractBasePtr stop = stop_->Clone();
  AbstractBasePtr step = step_->Clone();
  return std::make_shared<AbstractSlice>(start, stop, step);
}

AbstractBasePtr AbstractSlice::Broaden() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  AbstractBasePtr start = start_->Broaden();
  AbstractBasePtr stop = stop_->Broaden();
  AbstractBasePtr step = step_->Broaden();
  return std::make_shared<AbstractSlice>(start, stop, step);
}

std::string AbstractSlice::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "[";
  MS_EXCEPTION_IF_NULL(start_);
  buffer << start_->ToString() << " : ";
  MS_EXCEPTION_IF_NULL(stop_);
  buffer << stop_->ToString() << " : ";
  MS_EXCEPTION_IF_NULL(step_);
  buffer << step_->ToString();
  buffer << "]";
  return buffer.str();
}

ValuePtr AbstractSlice::RealBuildValue() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  ValuePtr start = start_->BuildValue();
  ValuePtr stop = stop_->BuildValue();
  ValuePtr step = step_->BuildValue();
  if (start->isa<AnyValue>() || stop->isa<AnyValue>() || step->isa<AnyValue>()) {
    return kAnyValue;
  }
  return std::make_shared<ValueSlice>(start, stop, step);
}

std::size_t AbstractSlice::hash() const {
  MS_EXCEPTION_IF_NULL(start_);
  MS_EXCEPTION_IF_NULL(stop_);
  MS_EXCEPTION_IF_NULL(step_);
  return hash_combine({tid(), start_->hash(), stop_->hash(), step_->hash()});
}

ShapePtr AbstractUndetermined::shape() const {
  auto shp = dyn_cast<Shape>(GetShapeTrack());
  if (shp == nullptr) {
    MS_LOG(EXCEPTION) << "Tensor should have a shape.";
  }
  return shp;
}

void AbstractUndetermined::set_shape(const BaseShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<NoShape>()) {
    MS_LOG(EXCEPTION) << "AbstractUndetermined can't set shape as NoShape.";
  }
  AbstractBase::set_shape(shape);
}

TypePtr AbstractTensor::BuildType() const {
  MS_EXCEPTION_IF_NULL(element_);
  TypePtr element_type = element_->BuildType();
  return std::make_shared<TensorType>(element_type);
}

BaseShapePtr AbstractTensor::BuildShape() const {
  auto shape = GetShapeTrack();
  // Guard from using set_shape(nullptr)
  if (shape == nullptr) {
    return kNoShape;
  }
  return shape;
}

AbstractBasePtr AbstractTensor::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  auto other_type = other->BuildType();
  MS_EXCEPTION_IF_NULL(other_type);
  MS_EXCEPTION_IF_NULL(element_);

  // AbstractTensor join with AbstractUndetermined
  if (other_type->type_id() == kObjectTypeUndeterminedType) {
    auto other_undetermined_tensor = dyn_cast_ptr<AbstractUndetermined>(other);
    MS_EXCEPTION_IF_NULL(other_undetermined_tensor);
    // Check shape
    auto res_shape = ShapeJoin(shape(), other_undetermined_tensor->shape());
    if (res_shape == nullptr) {
      ShapeJoinLogging(shape(), other_undetermined_tensor->shape(), shared_from_base<AbstractBase>(), other);
    }
    // Check element
    auto element = element_->Join(other_undetermined_tensor->element());
    MS_EXCEPTION_IF_NULL(element);
    return std::make_shared<AbstractUndetermined>(element, res_shape);
  }

  // AbstractTensor join with AbstractTensor
  auto other_tensor = dyn_cast_ptr<AbstractTensor>(other);
  if (other_tensor == nullptr) {
    AbstractTypeJoinLogging(shared_from_base<AbstractBase>(), other);
  }
  if (*this == *other) {
    return shared_from_base<AbstractBase>();
  }
  // Check shape
  auto res_shape = ShapeJoin(this->shape(), other_tensor->shape());
  if (res_shape == nullptr) {
    ShapeJoinLogging(shape(), other_tensor->shape(), shared_from_base<AbstractBase>(), other);
  }
  // Check element
  auto element = element_->Join(other_tensor->element_);
  MS_EXCEPTION_IF_NULL(element);
  auto ret = std::make_shared<AbstractTensor>(element, res_shape);
  ret->set_is_adapter(is_adapter_);
  return ret;
}

bool AbstractTensor::equal_to(const AbstractTensor &other) const {
  if (this == &other) {
    return true;
  }
  // Check if both Tensor or both AdapterTensor.
  if (is_adapter() != other.is_adapter()) {
    return false;
  }
  const auto &v1 = GetValueTrack();
  const auto &v2 = other.GetValueTrack();
  MS_EXCEPTION_IF_NULL(v1);
  MS_EXCEPTION_IF_NULL(v2);
  // Check if both point to same specific value.
  if (!v1->isa<AnyValue>()) {
    return v1 == v2;
  }
  // Check if both are AnyValue.
  if (!v2->isa<AnyValue>()) {
    return false;
  }
  // Check element type.
  if (!IsEqual(element_, other.element_)) {
    return false;
  }
  // Check shape.
  if (!IsEqual(shape(), other.shape())) {
    return false;
  }
  // Check shape value.
  if (!IsEqual(get_shape_value(), other.get_shape_value())) {
    return false;
  }
  // Check min and max values.
  return IsEqual(get_min_value(), other.get_min_value()) && IsEqual(get_max_value(), other.get_max_value());
}

bool AbstractTensor::operator==(const AbstractTensor &other) const { return equal_to(other); }

bool AbstractTensor::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (tid() != other.tid()) {
    return false;
  }
  return equal_to(static_cast<const AbstractTensor &>(other));
}

AbstractBasePtr AbstractTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(element_);
  auto clone = std::make_shared<AbstractTensor>(element_->Clone());
  ShapePtr shp = shape();
  clone->set_shape(shp->Clone());
  clone->set_value(GetValueTrack());
  clone->set_value_range(get_min_value(), get_max_value());
  clone->set_shape_value(get_shape_value());
  clone->set_is_adapter(is_adapter());
  return clone;
}

AbstractBasePtr AbstractTensor::Broaden() const {
  MS_EXCEPTION_IF_NULL(element_);
  auto broaden = std::make_shared<AbstractTensor>(element_->Broaden());
  auto shp = shape();
  MS_EXCEPTION_IF_NULL(shp);
  broaden->set_shape(shp->Clone());
  broaden->set_value(kAnyValue);
  broaden->set_is_adapter(is_adapter());
  return broaden;
}

AbstractBasePtr AbstractTensor::BroadenWithShape() const {
  MS_EXCEPTION_IF_NULL(element_);
  auto broaden = std::make_shared<AbstractTensor>(element_->Broaden());
  auto shp = shape()->Clone();
  MS_EXCEPTION_IF_NULL(shp);
  shp->Broaden();
  broaden->set_shape(shp);
  broaden->set_value(kAnyValue);
  broaden->set_is_adapter(is_adapter());
  return broaden;
}

AbstractBasePtr AbstractTensor::PartialBroaden() const { return Broaden(); }

std::string AbstractTensor::ToString() const {
  std::ostringstream buffer;
  BaseShapePtr shape_track = GetShapeTrack();
  MS_EXCEPTION_IF_NULL(shape_track);
  MS_EXCEPTION_IF_NULL(element_);
  auto value_track = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  buffer << type_name() << "("
         << "shape: " << shape_track->ToString() << ", element: " << element_->ToString()
         << ", value_ptr: " << value_track << ", value: " << value_track->ToString() << ")";
  return buffer.str();
}

TypePtr AbstractDictionary::BuildType() const {
  std::vector<std::pair<ValuePtr, TypePtr>> key_values;
  for (const auto &item : key_values_) {
    MS_EXCEPTION_IF_NULL(item.first);
    MS_EXCEPTION_IF_NULL(item.second);
    ValuePtr key_type = item.first->BuildValue();
    TypePtr value_type = item.second->BuildType();
    key_values.emplace_back(key_type, value_type);
  }
  return std::make_shared<Dictionary>(key_values);
}

bool AbstractDictionary::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractDictionary>()) {
    return false;
  }
  const auto &other_dict = dynamic_cast<const AbstractDictionary &>(other);
  if (key_values_.size() != other_dict.key_values_.size()) {
    return false;
  }
  for (size_t index = 0; index < key_values_.size(); ++index) {
    auto &kv1 = key_values_[index];
    auto &kv2 = other_dict.key_values_[index];
    if (!IsEqual(kv1.first, kv2.first) || !IsEqual(kv1.second, kv2.second)) {
      return false;
    }
  }
  return true;
}

AbstractBasePtr AbstractDictionary::Clone() const {
  std::vector<AbstractElementPair> kv;
  (void)std::transform(key_values_.cbegin(), key_values_.cend(), std::back_inserter(kv),
                       [](const AbstractElementPair &item) {
                         MS_EXCEPTION_IF_NULL(item.first);
                         MS_EXCEPTION_IF_NULL(item.second);
                         return std::make_pair(item.first->Clone(), item.second->Clone());
                       });
  return std::make_shared<AbstractDictionary>(kv);
}

AbstractBasePtr AbstractDictionary::Broaden() const {
  std::vector<AbstractElementPair> kv;
  (void)std::transform(key_values_.cbegin(), key_values_.cend(), std::back_inserter(kv),
                       [](const AbstractElementPair &item) {
                         MS_EXCEPTION_IF_NULL(item.second);
                         return std::make_pair(item.first, item.second->Broaden());
                       });
  return std::make_shared<AbstractDictionary>(kv);
}

std::string AbstractDictionary::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "{ ";
  for (const auto &kv : key_values_) {
    MS_EXCEPTION_IF_NULL(kv.first);
    MS_EXCEPTION_IF_NULL(kv.second);
    buffer << "(" << kv.first->ToString() << ": " << kv.second->ToString() << ") ";
  }
  buffer << "}";
  return buffer.str();
}

std::size_t AbstractDictionary::hash() const {
  std::size_t hash_sum = std::accumulate(key_values_.cbegin(), key_values_.cend(), tid(),
                                         [](std::size_t hash_sum, const AbstractElementPair &item) {
                                           MS_EXCEPTION_IF_NULL(item.first);
                                           MS_EXCEPTION_IF_NULL(item.second);
                                           hash_sum = hash_combine(hash_sum, item.first->hash());
                                           hash_sum = hash_combine(hash_sum, item.second->hash());
                                           return hash_sum;
                                         });
  return hash_sum;
}

ValuePtr AbstractDictionary::RealBuildValue() const {
  std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
  for (const auto &item : key_values_) {
    MS_EXCEPTION_IF_NULL(item.first);
    MS_EXCEPTION_IF_NULL(item.second);
    auto key_element_value = item.first->BuildValue();
    auto value_element_value = item.second->BuildValue();
    MS_EXCEPTION_IF_NULL(key_element_value);
    MS_EXCEPTION_IF_NULL(value_element_value);
    if (value_element_value->isa<AnyValue>()) {
      return kAnyValue;
    }
    key_values.emplace_back(key_element_value, value_element_value);
  }
  return std::make_shared<ValueDictionary>(key_values);
}

TypePtr AbstractJTagged::BuildType() const {
  MS_EXCEPTION_IF_NULL(element_);
  TypePtr subtype = element_->BuildType();
  return std::make_shared<JTagged>(subtype);
}

AbstractBasePtr AbstractJTagged::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  auto other_jtagged = dyn_cast_ptr<AbstractJTagged>(other);
  if (other_jtagged == nullptr) {
    AbstractTypeJoinLogging(shared_from_base<AbstractBase>(), other);
  }
  MS_EXCEPTION_IF_NULL(element_);
  auto joined_elem = element_->Join(other_jtagged->element_);
  return std::make_shared<AbstractJTagged>(joined_elem);
}

bool AbstractJTagged::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractJTagged>()) {
    return false;
  }
  const auto &other_jtagged = dynamic_cast<const AbstractJTagged &>(other);
  return IsEqual(element_, other_jtagged.element_);
}

std::string AbstractJTagged::ToString() const {
  std::ostringstream buffer;
  MS_EXCEPTION_IF_NULL(element_);
  buffer << type_name() << "("
         << "element: " << element_->ToString() << ")";
  return buffer.str();
}

AbstractRefTensor::AbstractRefTensor(const AbstractTensorPtr &ref_value, const ValuePtr &ref_key_value)
    : AbstractTensor(*ref_value), ref_key_value_(ref_key_value) {
  set_type(std::make_shared<RefType>());
  set_is_adapter(ref_value->is_adapter());
  MS_EXCEPTION_IF_NULL(ref_key_value);
  if (ref_key_value != kAnyValue && !ref_key_value->isa<RefKey>()) {
    MS_LOG(EXCEPTION) << "ref_key_value must be kAnyValue or RefKey, but got:" << ref_key_value->ToString();
  }
}

TypePtr AbstractRefTensor::BuildType() const {
  auto type = AbstractTensor::BuildType();
  auto subtype = dyn_cast_ptr<TensorType>(type);
  MS_EXCEPTION_IF_NULL(subtype);
  return std::make_shared<RefType>(subtype);
}

bool AbstractRefTensor::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractRefTensor>()) {
    return false;
  }
  return AbstractTensor::equal_to(dynamic_cast<const AbstractTensor &>(other));
}

AbstractBasePtr AbstractRefTensor::Join(const std::shared_ptr<AbstractRefTensor> &other) {
  if (*this == *other) {
    return shared_from_base<AbstractRefTensor>();
  }
  // Firstly, join the ref_key_value.
  auto joined_ref_key = ValueJoin(ref_key_value_, other->ref_key_value_);
  // Secondly , join the tensor value.
  auto joined_tensor = AbstractTensor::Join(other)->cast<AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(joined_tensor);
  return std::make_shared<AbstractRefTensor>(joined_tensor, joined_ref_key);
}

AbstractBasePtr AbstractRefTensor::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  // Abstract ref join abstract ref
  if (other->isa<AbstractRefTensor>()) {
    return AbstractRefTensor::Join(other->cast<AbstractRefPtr>());
  }
  // Abstract ref join other abstract are same to AbstractTensor::Join.
  auto joined_tensor = AbstractTensor::Join(other);
  if (!joined_tensor->isa<AbstractTensor>()) {
    MS_LOG(EXCEPTION) << "Expect an AbstractTensor, but got:" << joined_tensor->ToString()
                      << ", other:" << other->ToString();
  }
  return joined_tensor;
}

AbstractBasePtr AbstractRefTensor::Clone() const {
  auto abs_tensor = AbstractTensor::Clone()->cast<AbstractTensorPtr>();
  return std::make_shared<AbstractRefTensor>(abs_tensor, ref_key_value_);
}

AbstractBasePtr AbstractRefTensor::Broaden() const {
  // Always broaden for ref
  auto abs_tensor = AbstractTensor::Broaden()->cast<AbstractTensorPtr>();
  // Broaden the tensor value and keep the ref_key_value.
  auto ret = std::make_shared<AbstractRefTensor>(abs_tensor, ref_key_value_);
  return ret;
}

std::string AbstractRefTensor::ToString() const {
  std::ostringstream buffer;
  MS_EXCEPTION_IF_NULL(ref_key_value_);
  buffer << type_name() << "("
         << "key: " << ref_key_value_->ToString() << " ref_value: " << AbstractTensor::ToString();
  auto value = GetValueTrack();
  if (value != nullptr) {
    buffer << ", value: " << value->ToString();
  }
  buffer << ")";
  return buffer.str();
}

AbstractBasePtr AbstractRefTensor::PartialBroaden() const { return Clone(); }

bool AbstractNone::operator==(const AbstractBase &other) const { return other.isa<AbstractNone>(); }

std::string AbstractNone::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "(Value: None)";
  return buffer.str();
}

ValuePtr AbstractNone::RealBuildValue() const { return kNone; }

bool AbstractNull::operator==(const AbstractBase &other) const { return other.isa<AbstractNull>(); }

std::string AbstractNull::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "(Value: Null)";
  return buffer.str();
}

bool AbstractTimeOut::operator==(const AbstractBase &other) const { return other.isa<AbstractTimeOut>(); }

std::string AbstractTimeOut::ToString() const {
  std::ostringstream buffer;
  buffer << "AbstractTimeOut "
         << "(Value: Null)";
  return buffer.str();
}

bool AbstractEllipsis::operator==(const AbstractBase &other) const { return other.isa<AbstractEllipsis>(); }

std::string AbstractEllipsis::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "(Value: Ellipsis)";
  return buffer.str();
}

TypePtr AbstractKeywordArg::BuildType() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  TypePtr type = arg_value_->BuildType();
  return std::make_shared<Keyword>(arg_name_, type);
}

AbstractBasePtr AbstractKeywordArg::Clone() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  return std::make_shared<AbstractKeywordArg>(arg_name_, arg_value_->Clone());
}

AbstractBasePtr AbstractKeywordArg::Broaden() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  return std::make_shared<AbstractKeywordArg>(arg_name_, arg_value_->Broaden());
}

std::size_t AbstractKeywordArg::hash() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  return hash_combine({tid(), std::hash<std::string>{}(arg_name_), arg_value_->hash()});
}

std::string AbstractKeywordArg::ToString() const {
  std::ostringstream buffer;
  MS_EXCEPTION_IF_NULL(arg_value_);
  buffer << type_name() << "(";
  buffer << "key: " << arg_name_;
  buffer << ", value: " << arg_value_->ToString();
  buffer << ")";
  return buffer.str();
}

bool AbstractKeywordArg::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractKeywordArg>()) {
    return false;
  }
  return *this == static_cast<const AbstractKeywordArg &>(other);
}

bool AbstractKeywordArg::operator==(const AbstractKeywordArg &other) const {
  if (this == &other) {
    return true;
  }
  return other.arg_name_ == arg_name_ && IsEqual(other.arg_value_, arg_value_);
}

ValuePtr AbstractKeywordArg::RealBuildValue() const {
  MS_EXCEPTION_IF_NULL(arg_value_);
  ValuePtr value = arg_value_->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<AnyValue>()) {
    return kAnyValue;
  }
  return std::make_shared<KeywordArg>(arg_name_, value);
}

std::size_t AbstractBasePtrListHash(const AbstractBasePtrList &args_spec_list) {
  // Hash for empty list is zero.
  if (args_spec_list.empty()) {
    return 0;
  }
  // Hashing all elements is costly, we only calculate hash from
  // the first element and last few elements base on some experiments.
  // In some scenarios, this may lead high hash conflicts. Therefore,
  // we should use this hash function in hash tables that can tolerate
  // high hash conflicts, such as std::unordered_map.
  constexpr size_t kMaxLastElements = 4;
  const size_t n_args = args_spec_list.size();
  // Hash from list size and the first element.
  std::size_t hash_value = hash_combine(n_args, args_spec_list[0]->hash());
  // Hash from last few elements.
  const size_t start = ((n_args > kMaxLastElements) ? (n_args - kMaxLastElements) : 1);
  for (size_t i = start; i < n_args; ++i) {
    hash_value = hash_combine(hash_value, args_spec_list[i]->hash());
  }
  return hash_value;
}

bool AbstractBasePtrListDeepEqual(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs) {
  const std::size_t size = lhs.size();
  if (size != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (!IsEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

// SparseTensor
template <typename T>
const T AbstractSparseTensor::GetAbsPtrAt(size_t index) const {
  if (index >= size()) {
    MS_LOG(EXCEPTION) << "Index should be in range of [0, " << size() << "), but got " << index
                      << " for abstract: " << ToString();
  }
  AbstractBasePtr base = elements()[index];
  MS_EXCEPTION_IF_NULL(base);
  return base->cast<T>();
}

BaseShapePtrList AbstractSparseTensor::ElementsShapeTupleRecursive() const {
  BaseShapePtrList element_shape_list;
  for (const auto &element : elements()) {
    MS_EXCEPTION_IF_NULL(element);
    auto abs_tuple = element->cast_ptr<AbstractTuple>();
    if (abs_tuple == nullptr) {
      element_shape_list.push_back(element->BuildShape());
    } else {
      for (const auto &scalar : abs_tuple->elements()) {
        MS_EXCEPTION_IF_NULL(scalar);
        element_shape_list.push_back(scalar->BuildShape());
      }
    }
  }
  return element_shape_list;
}

const TypeId AbstractSparseTensor::GetTensorTypeIdAt(size_t index) const {
  size_t shape_idx = size() - 1;
  if (index >= shape_idx) {
    MS_LOG(EXCEPTION) << "Index must be in range of [0, " << shape_idx << "), but got " << index << " for "
                      << ToString();
  }
  auto abs_tensor = GetAbsPtrAt<abstract::AbstractTensorPtr>(index);
  MS_EXCEPTION_IF_NULL(abs_tensor);
  return abs_tensor->element()->BuildType()->type_id();
}

const TypeId AbstractSparseTensor::GetShapeTypeIdAt(size_t index) const {
  if (index >= shape()->size()) {
    MS_LOG(EXCEPTION) << "Index must be in range of [0, " << shape()->size() << "), but got " << index << " for "
                      << ToString();
  }
  return shape()->elements()[index]->BuildType()->type_id();
}

TypePtr AbstractSparseTensor::BuildType() const { return std::make_shared<SparseTensorType>(); }

const AbstractTuplePtr AbstractSparseTensor::shape() const {
  auto res = GetAbsPtrAt<abstract::AbstractTuplePtr>(size() - 1);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Get shape nullptr in AbstractSparseTensor: " << ToString();
  }
  return res;
}

// RowTensor
TypePtr AbstractRowTensor::BuildType() const {
  MS_EXCEPTION_IF_NULL(element());
  TypePtr element_type = element()->BuildType();
  return std::make_shared<RowTensorType>(element_type);
}

AbstractBasePtr AbstractRowTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(element());
  auto clone = std::make_shared<AbstractRowTensor>(element()->Clone());
  ShapePtr shp = shape();
  MS_EXCEPTION_IF_NULL(shp);
  clone->set_shape(shp->Clone());
  clone->set_value(GetValueTrack());
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  MS_EXCEPTION_IF_NULL(dense_shape_);
  auto indices_clone = indices_->Clone();
  auto value_clone = values_->Clone();
  auto dense_clone = dense_shape_->Clone();
  MS_EXCEPTION_IF_NULL(indices_clone);
  MS_EXCEPTION_IF_NULL(value_clone);
  MS_EXCEPTION_IF_NULL(dense_clone);
  clone->set_indices(indices_clone->cast<AbstractTensorPtr>());
  clone->set_values(value_clone->cast<AbstractTensorPtr>());
  clone->set_dense_shape(dense_clone->cast<AbstractTuplePtr>());
  return clone;
}

AbstractRowTensorPtr AbstractRowTensor::MakeAbstract(const BaseShapePtr &shp) const {
  MS_EXCEPTION_IF_NULL(element());
  auto broaden = std::make_shared<AbstractRowTensor>(element()->Broaden());
  broaden->set_shape(shp);
  broaden->set_value(kAnyValue);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  MS_EXCEPTION_IF_NULL(dense_shape_);
  auto indices_clone = indices_->Clone();
  auto value_clone = values_->Clone();
  auto dense_clone = dense_shape_->Clone();
  MS_EXCEPTION_IF_NULL(indices_clone);
  MS_EXCEPTION_IF_NULL(value_clone);
  MS_EXCEPTION_IF_NULL(dense_clone);
  broaden->set_indices(indices_clone->cast<AbstractTensorPtr>());
  broaden->set_values(value_clone->cast<AbstractTensorPtr>());
  broaden->set_dense_shape(dense_clone->cast<AbstractTuplePtr>());
  return broaden;
}

AbstractBasePtr AbstractRowTensor::Broaden() const {
  auto shp = shape()->Clone();
  MS_EXCEPTION_IF_NULL(shp);
  return MakeAbstract(shp);
}

AbstractBasePtr AbstractRowTensor::BroadenWithShape() const {
  auto shp = shape()->Clone();
  MS_EXCEPTION_IF_NULL(shp);
  shp->Broaden();
  return MakeAbstract(shp);
}

std::string AbstractRowTensor::ToString() const {
  std::ostringstream buffer;
  BaseShapePtr shape_track = GetShapeTrack();
  MS_EXCEPTION_IF_NULL(shape_track);
  MS_EXCEPTION_IF_NULL(element());
  auto value_track = GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  MS_EXCEPTION_IF_NULL(dense_shape_);
  buffer << type_name() << "("
         << "shape: " << shape_track->ToString() << ", element: " << element()->ToString()
         << ", value_ptr: " << value_track << ", value: " << value_track->ToString()
         << ", indices: " << indices_->ToString() << ", values: " << values_->ToString()
         << ", dense_shape: " << dense_shape_->ToString() << ")";
  return buffer.str();
}

// COOTensor
TypePtr AbstractCOOTensor::BuildType() const {
  MS_EXCEPTION_IF_NULL(indices());
  MS_EXCEPTION_IF_NULL(values());
  MS_EXCEPTION_IF_NULL(shape());
  TypePtrList elements{indices()->element()->BuildType(), values()->element()->BuildType()};
  (void)std::transform(shape()->elements().begin(), shape()->elements().end(), std::back_inserter(elements),
                       [](const AbstractBasePtr &p) { return p->BuildType(); });
  return std::make_shared<COOTensorType>(elements);
}

AbstractBasePtr AbstractCOOTensor::Clone() const {
  AbstractBasePtrList element_list;
  for (const auto &element : elements()) {
    MS_EXCEPTION_IF_NULL(element);
    AbstractBasePtr clone = element->Clone();
    element_list.push_back(clone);
  }
  return std::make_shared<abstract::AbstractCOOTensor>(element_list);
}

AbstractBasePtr AbstractCOOTensor::Broaden() const {
  return std::make_shared<abstract::AbstractCOOTensor>(ElementsBroaden());
}

AbstractBasePtr AbstractCOOTensor::PartialBroaden() const { return Broaden(); }

std::string AbstractCOOTensor::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "("
         << "indices: " << indices()->ToString() << ", values" << values()->ToString()
         << ", dense_shape: " << shape()->ToString();
  return buffer.str();
}

const AbstractTensorPtr AbstractCOOTensor::indices() const {
  auto res = GetAbsPtrAt<abstract::AbstractTensorPtr>(kIndicesIdx);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Get indices nullptr in AbstractCOOTensor: " << ToString();
  }
  return res;
}

const AbstractTensorPtr AbstractCOOTensor::values() const {
  auto res = GetAbsPtrAt<abstract::AbstractTensorPtr>(kValuesIdx);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Get values nullptr in AbstractCOOTensor: " << ToString();
  }
  return res;
}

// CSRTensor
TypePtr AbstractCSRTensor::BuildType() const {
  MS_EXCEPTION_IF_NULL(indptr());
  MS_EXCEPTION_IF_NULL(indices());
  MS_EXCEPTION_IF_NULL(values());
  MS_EXCEPTION_IF_NULL(shape());
  TypePtrList elements{indptr()->element()->BuildType(), indices()->element()->BuildType(),
                       values()->element()->BuildType()};
  (void)std::transform(shape()->elements().begin(), shape()->elements().end(), std::back_inserter(elements),
                       [](const AbstractBasePtr &p) { return p->BuildType(); });
  return std::make_shared<CSRTensorType>(elements);
}

AbstractBasePtr AbstractCSRTensor::Clone() const {
  AbstractBasePtrList element_list;
  for (const auto &element : elements()) {
    MS_EXCEPTION_IF_NULL(element);
    AbstractBasePtr clone = element->Clone();
    element_list.push_back(clone);
  }
  return std::make_shared<abstract::AbstractCSRTensor>(element_list);
}

AbstractBasePtr AbstractCSRTensor::Broaden() const {
  return std::make_shared<abstract::AbstractCSRTensor>(ElementsBroaden());
}

AbstractBasePtr AbstractCSRTensor::PartialBroaden() const { return Broaden(); }

std::string AbstractCSRTensor::ToString() const {
  std::ostringstream buffer;
  buffer << type_name() << "("
         << "indptr: " << indptr()->ToString() << ", indices: " << indices()->ToString() << ", values"
         << values()->ToString() << ", dense_shape: " << shape()->ToString();
  return buffer.str();
}

const AbstractTensorPtr AbstractCSRTensor::indptr() const {
  auto res = GetAbsPtrAt<abstract::AbstractTensorPtr>(kIndptrIdx);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Get indptr nullptr in AbstractCSRTensor: " << ToString();
  }
  return res;
}

const AbstractTensorPtr AbstractCSRTensor::indices() const {
  auto res = GetAbsPtrAt<abstract::AbstractTensorPtr>(kIndicesIdx);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Get indices nullptr in AbstractCSRTensor: " << ToString();
  }
  return res;
}

const AbstractTensorPtr AbstractCSRTensor::values() const {
  auto res = GetAbsPtrAt<abstract::AbstractTensorPtr>(kValuesIdx);
  if (res == nullptr) {
    MS_LOG(EXCEPTION) << "Get values nullptr in AbstractCSRTensor: " << ToString();
  }
  return res;
}

AbstractMapTensor::AbstractMapTensor(const MapTensorPtr &map_tensor)
    : AbstractBase(map_tensor, std::make_shared<MapTensorType>(map_tensor->KeyDtype(), map_tensor->ValueDtype()),
                   std::make_shared<Shape>(map_tensor->shape())),
      ref_key_value_(kAnyValue),
      default_value_(map_tensor->default_value()),
      permit_filter_value_(map_tensor->permit_filter_value()),
      evict_filter_value_(map_tensor->evict_filter_value()),
      value_shape_(std::make_shared<Shape>(map_tensor->value_shape())) {}

AbstractMapTensor::AbstractMapTensor(const MapTensorPtr &map_tensor, const ValuePtr &ref_key_value)
    : AbstractBase(kAnyValue, std::make_shared<MapTensorType>(map_tensor->KeyDtype(), map_tensor->ValueDtype()),
                   std::make_shared<Shape>(map_tensor->shape())),
      ref_key_value_(ref_key_value),
      default_value_(map_tensor->default_value()),
      permit_filter_value_(map_tensor->permit_filter_value()),
      evict_filter_value_(map_tensor->evict_filter_value()),
      value_shape_(std::make_shared<Shape>(map_tensor->value_shape())) {}

AbstractMapTensor::AbstractMapTensor(const AbstractMapTensor &other)
    : AbstractBase(other.GetValueTrack(), other.GetTypeTrack(), other.GetShapeTrack()),
      ref_key_value_(other.ref_key_value_),
      default_value_(other.default_value_),
      permit_filter_value_(other.permit_filter_value()),
      evict_filter_value_(other.evict_filter_value()),
      value_shape_(other.value_shape_) {
  set_shape(other.shape());
}

AbstractMapTensor::AbstractMapTensor(const TypePtr &type, const ShapePtr &value_shape, const ValuePtr &value,
                                     const ValuePtr &ref_key_value, const ValuePtr &default_value,
                                     const ValuePtr &permit_filter_value, const ValuePtr &evict_filter_value) {
  set_value(value);
  set_type(type);
  ref_key_value_ = ref_key_value;
  default_value_ = default_value;
  permit_filter_value_ = permit_filter_value;
  evict_filter_value_ = evict_filter_value;
  ShapeVector shape = {abstract::Shape::kShapeDimAny};
  (void)shape.insert(shape.end(), value_shape->shape().begin(), value_shape->shape().end());
  set_shape(std::make_shared<mindspore::abstract::Shape>(shape));
}

AbstractBasePtr AbstractMapTensor::Clone() const { return std::make_shared<AbstractMapTensor>(*this); }

AbstractBasePtr AbstractMapTensor::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  // Same pointer.
  if (this == other.get()) {
    return shared_from_base<AbstractMapTensor>();
  }

  // Check class.
  auto other_abs = dyn_cast<AbstractMapTensor>(other);
  if (other_abs == nullptr) {
    AbstractTypeJoinLogging(shared_from_base<AbstractBase>(), other);
  }

  // Join type.
  auto joined_type = TypeJoin(GetTypeTrack(), other_abs->GetTypeTrack());
  if (joined_type == kAnyType) {
    TypeJoinLogging(GetTypeTrack(), other_abs->GetTypeTrack(), shared_from_base<AbstractBase>(), other);
  }

  // Join shape
  auto joined_shape = ShapeJoin(value_shape(), other_abs->value_shape());
  if (joined_shape == nullptr) {
    ShapeJoinLogging(value_shape(), other_abs->value_shape(), shared_from_base<AbstractBase>(), other);
  }

  // Join value.
  auto joined_value = (GetValueTrack() == other_abs->GetValueTrack() ? GetValueTrack() : kAnyValue);

  // Join the ref_key_value.
  auto joined_ref_key = ValueJoin(ref_key_value_, other_abs->ref_key_value_);

  // Join the default_value.
  auto joined_default_value = ValueJoin(default_value_, other_abs->default_value_);
  if (joined_default_value == kAnyValue) {
    MS_EXCEPTION(ValueError) << "Join default value failed for MapTensor. " << default_value_->ToString()
                             << " != " << other_abs->default_value_->ToString();
  }

  // Join the permit_filter_value.
  auto joined_permit_filter_value = ValueJoin(permit_filter_value_, other_abs->permit_filter_value_);
  if (joined_permit_filter_value == kAnyValue) {
    MS_EXCEPTION(ValueError) << "Join default value failed for MapTensor. " << permit_filter_value_->ToString()
                             << " != " << other_abs->permit_filter_value_->ToString();
  }

  // Join the evict_filter_value.
  auto joined_evict_filter_value = ValueJoin(evict_filter_value_, other_abs->evict_filter_value_);
  if (joined_evict_filter_value == kAnyValue) {
    MS_EXCEPTION(ValueError) << "Join evict_filter_value failed for MapTensor. " << evict_filter_value_->ToString()
                             << " != " << other_abs->evict_filter_value_->ToString();
  }

  return std::make_shared<AbstractMapTensor>(joined_type, joined_shape, joined_value, joined_ref_key,
                                             joined_default_value, joined_permit_filter_value,
                                             joined_evict_filter_value);
}

bool AbstractMapTensor::operator==(const AbstractBase &other) const {
  if (this == &other) {
    return true;
  }
  if (!other.isa<AbstractMapTensor>()) {
    return false;
  }
  const auto &v1 = GetValueTrack();
  const auto &v2 = other.GetValueTrack();
  MS_EXCEPTION_IF_NULL(v1);
  MS_EXCEPTION_IF_NULL(v2);
  // Check if both point to same specific value.
  if (!v1->isa<AnyValue>()) {
    return v1 == v2;
  }
  // Check if both are AnyValue.
  if (!v2->isa<AnyValue>()) {
    return false;
  }
  const auto &other_map_tensor = dynamic_cast<const AbstractMapTensor &>(other);
  return common::IsEqual(GetTypeTrack(), other_map_tensor.GetTypeTrack()) &&
         common::IsEqual(GetShapeTrack(), other_map_tensor.GetShapeTrack()) &&
         common::IsEqual(default_value(), other_map_tensor.default_value());
}

std::size_t AbstractMapTensor::hash() const {
  const auto &type = GetTypeTrack();
  const auto &value_shape = GetShapeTrack();
  MS_EXCEPTION_IF_NULL(type);
  MS_EXCEPTION_IF_NULL(value_shape);
  MS_EXCEPTION_IF_NULL(default_value_);
  std::size_t hash_value = hash_combine(tid(), type->hash());
  hash_value = hash_combine(hash_value, value_shape->hash());
  return hash_combine(hash_value, default_value_->hash());
}

std::string AbstractMapTensor::ToString() const {
  const auto &type = GetTypeTrack();
  const auto &value = GetValueTrack();
  const auto &value_shape = GetShapeTrack();
  MS_EXCEPTION_IF_NULL(type);
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(value_shape);
  return type_name() + "(" + type->ToString() + " " + value_shape->ToString() +
         " key: " + (ref_key_value_ == nullptr ? "<null>" : ref_key_value_->ToString()) +
         " value: " + value->ToString() + ")";
}

AbstractBasePtr AbstractUMonad::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  if (!other->isa<AbstractUMonad>()) {
    auto this_type = GetTypeTrack();
    auto other_type = other->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(this_type);
    MS_EXCEPTION_IF_NULL(other);
    TypeJoinLogging(this_type, other_type, shared_from_base<AbstractBase>(), other);
  }
  return shared_from_base<AbstractBase>();
}

bool AbstractUMonad::operator==(const AbstractBase &other) const { return other.isa<AbstractUMonad>(); }

AbstractBasePtr AbstractIOMonad::Join(const AbstractBasePtr &other) {
  MS_EXCEPTION_IF_NULL(other);
  if (!other->isa<AbstractIOMonad>()) {
    auto this_type = GetTypeTrack();
    auto other_type = other->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(this_type);
    MS_EXCEPTION_IF_NULL(other);
    TypeJoinLogging(this_type, other_type, shared_from_base<AbstractBase>(), other);
  }
  return shared_from_base<AbstractBase>();
}

bool AbstractIOMonad::operator==(const AbstractBase &other) const { return other.isa<AbstractIOMonad>(); }

ValuePtr GetRefKeyValue(const AbstractBasePtr &abs) {
  auto abs_ref = abs->cast_ptr<AbstractRefTensor>();
  if (abs_ref != nullptr) {
    return abs_ref->ref_key_value();
  }
  auto abs_map_tensor = abs->cast_ptr<AbstractMapTensor>();
  if (abs_map_tensor != nullptr) {
    return abs_map_tensor->ref_key_value();
  }
  return nullptr;
}
}  // namespace abstract
}  // namespace mindspore
