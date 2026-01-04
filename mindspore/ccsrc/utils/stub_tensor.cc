/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "include/utils/stub_tensor.h"
#include "utils/ms_exception.h"
#include "include/utils/convert_utils_py.h"
#include "utils/convert_utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "tools/profiler/profiler.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace stub {
namespace {
StubNodePtr MakeStubNode(const TypePtr &type) {
  if (type->isa<TensorType>()) {
    return std::make_shared<TensorNode>();
  } else if (type->isa<Tuple>() || type->isa<List>()) {
    TypePtrList elements;
    if (type->isa<Tuple>()) {
      auto tuple_type = type->cast<TuplePtr>();
      elements = tuple_type->elements();
    } else {
      auto list_type = type->cast<ListPtr>();
      elements = list_type->elements();
    }
    auto node = std::make_shared<SequenceNode>(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      auto elem = MakeStubNode(elements[i]);
      node->SetElement(i, elem);
    }
    return node;
  } else if (type == kTypeAny) {
    return std::make_shared<AnyTypeNode>();
  } else if (type == kTypeNone) {
    return std::make_shared<NoneTypeNode>();
  }
  MS_EXCEPTION(NotSupportError) << "Unsupported stub tensor for type: " << type->ToString();
}

py::object MakeOutput(const StubNodePtr &node) {
  if (node->isa<TensorNode>()) {
    auto tensor = node->cast<std::shared_ptr<TensorNode>>();
    return py::cast(tensor);
  } else if (node->isa<SequenceNode>()) {
    auto seq = node->cast<std::shared_ptr<SequenceNode>>();
    MS_EXCEPTION_IF_NULL(seq);
    auto &elements = seq->Elements();
    if (elements.empty()) {
      return py::cast(seq);
    }
    py::tuple out(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      out[i] = MakeOutput(elements[i]);
    }
    return out;
  } else if (node->isa<AnyTypeNode>()) {
    auto tensor = node->cast<std::shared_ptr<AnyTypeNode>>();
    return py::cast(tensor);
  } else {
    auto tensor = node->cast<std::shared_ptr<NoneTypeNode>>();
    return py::cast(tensor);
  }
}
}  // namespace

bool StubNode::SetAbstract(const AbstractBasePtr &abs) {
  std::unique_lock<std::mutex> lock(mutex_);
  abstract_ = abs;
  cond_var_.notify_all();
  return true;
}

bool StubNode::SetValueSimpleInfo(const ValueSimpleInfoPtr &output_value_simple_info) {
  std::unique_lock<std::mutex> lock(mutex_);
  output_value_simple_info_ = output_value_simple_info;
  cond_var_.notify_all();
  return true;
}
void StubNode::SetValue(const ValuePtr &val) {
  std::unique_lock<std::mutex> lock(mutex_);
  value_ = val;
  cond_var_.notify_all();
}

void StubNode::SetException(const std::exception_ptr &e_ptr) {
  // cppcheck-suppress unreadVariable
  std::unique_lock<std::mutex> lock(mutex_);
  e_ptr_ = e_ptr;
  cond_var_.notify_all();
}

ValuePtr StubNode::WaitValue() {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kWaitPipeline);
  // cppcheck-suppress unreadVariable
  GilReleaseWithCheck gil_release;
  std::unique_lock<std::mutex> lock(mutex_);
  cond_var_.wait(lock, [this] { return value_.get() != nullptr || e_ptr_ != nullptr; });
  if (e_ptr_ != nullptr) {
    // Need to clear exception in the instance.
    MsException::Instance().CheckException();
    std::rethrow_exception(e_ptr_);
  }
  return value_;
}

void StubNode::WaitPipeline() {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kWaitPipeline);
  // cppcheck-suppress unreadVariable
  GilReleaseWithCheck gil_release;
  std::unique_lock<std::mutex> lock(mutex_);
  cond_var_.wait(lock, [this] {
    return abstract_.get() != nullptr || output_value_simple_info_.get() != nullptr || e_ptr_ != nullptr;
  });
  if (e_ptr_ != nullptr) {
    // Need to clear exception in the instance.
    MsException::Instance().CheckException();
    std::rethrow_exception(e_ptr_);
  }
}

AbstractBasePtr StubNode::ToAbstract() {
  WaitPipeline();
  if (output_value_simple_info_ == nullptr) {
    MS_EXCEPTION_IF_NULL(abstract_);
    return abstract_;
  }
  return TransformValueSimpleInfoToAbstract(*output_value_simple_info_);
}

py::object TensorNode::GetValue() {
  auto val = WaitValue();
  return ValueToPyData(val);
}

py::object TensorNode::GetShape() {
  WaitPipeline();
  abstract::ShapePtr shape{nullptr};
  ShapeVector shape_vector;
  if (output_value_simple_info_ == nullptr) {
    MS_EXCEPTION_IF_NULL(abstract_);
    auto base = abstract_->BuildShape();
    shape = base->cast<abstract::ShapePtr>();
    if (shape && !shape->IsDynamic()) {
      shape_vector = shape->shape();
    } else {
      auto val = WaitValue();
      auto tensor = val->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      shape_vector = tensor->shape();
    }
  } else {
    if (output_value_simple_info_->size_ != kIndex1) {
      MS_LOG(EXCEPTION) << "Simple infer size " << output_value_simple_info_->size_ << " is not equal to 1";
    }
    shape_vector = output_value_simple_info_->shape_vector_[kIndex0];
  }
  auto ret = py::tuple(shape_vector.size());
  for (size_t i = 0; i < shape_vector.size(); ++i) {
    ret[i] = shape_vector[i];
  }
  return ret;
}

py::object TensorNode::GetDtype() {
  WaitPipeline();
  TypePtr base = nullptr;
  if (output_value_simple_info_ == nullptr) {
    MS_EXCEPTION_IF_NULL(abstract_);
    base = abstract_->BuildType();
    if (base->isa<TensorType>()) {
      base = base->cast<TensorTypePtr>()->element();
    }
  } else {
    if (output_value_simple_info_->size_ != kIndex1) {
      MS_LOG(EXCEPTION) << "Simple infer size " << output_value_simple_info_->size_ << " is not equal to 1";
    }
    base = output_value_simple_info_->dtype_vector_[kIndex0];
  }
  return py::cast(base);
}

bool TensorNode::SetAbstract(const AbstractBasePtr &abs) {
  if (!abs->isa<abstract::AbstractTensor>() && !abs->isa<abstract::AbstractMapTensor>()) {
    if (!abs->isa<abstract::AbstractScalar>() || abs->BuildValue() != kValueAny) {
      return false;
    }
  }
  return StubNode::SetAbstract(abs);
}

py::object SequenceNode::GetElements() {
  if (!is_elements_build_.load()) {
    (void)WaitPipeline();
  }
  py::tuple out(elements_.size());
  for (size_t i = 0; i < elements_.size(); ++i) {
    out[i] = MakeOutput(elements_[i]);
  }
  return out;
}

bool SequenceNode::SetAbstract(const AbstractBasePtr &abs) {
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  if (seq_abs == nullptr) {
    return false;
  }
  auto children = seq_abs->elements();
  if (!is_elements_build_.load()) {
    for (const auto &child : children) {
      (void)elements_.emplace_back(MakeStubNode(child->BuildType()));
    }
  }
  is_elements_build_ = true;
  if (elements_.size() != children.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    if (!elements_[i]->SetAbstract(children[i])) {
      return false;
    }
  }
  return StubNode::SetAbstract(abs);
}

bool SequenceNode::SetValueSimpleInfo(const mindspore::ValueSimpleInfoPtr &output_value_simple_info) {
  MS_EXCEPTION_IF_NULL(output_value_simple_info);
  if (!is_elements_build_.load()) {
    for (size_t i = 0; i < output_value_simple_info->size_; ++i) {
      (void)elements_.emplace_back(
        MakeStubNode(std::make_shared<TensorType>(output_value_simple_info->dtype_vector_[i])));
    }
  }
  is_elements_build_ = true;
  for (size_t i = 0; i < output_value_simple_info->size_; ++i) {
    auto elem_simple_info = std::make_shared<mindspore::ValueSimpleInfo>();
    elem_simple_info->size_ = kIndex1;
    (void)elem_simple_info->shape_vector_.emplace_back(output_value_simple_info->shape_vector_[i]);
    (void)elem_simple_info->dtype_vector_.emplace_back(output_value_simple_info->dtype_vector_[i]);
    MS_EXCEPTION_IF_NULL(elements_[i]);
    if (!elements_[i]->SetValueSimpleInfo(elem_simple_info)) {
      return false;
    }
  }
  return StubNode::SetValueSimpleInfo(output_value_simple_info);
}

void SequenceNode::SetValue(const ValuePtr &val) {
  auto seq_value = val->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_value);
  auto children = seq_value->value();
  for (size_t i = 0; i < children.size(); ++i) {
    elements_[i]->SetValue(children[i]);
  }
  StubNode::SetValue(val);
}

void SequenceNode::SetException(const std::exception_ptr &e_ptr) {
  for (auto &element : elements_) {
    element->SetException(e_ptr);
  }
  StubNode::SetException(e_ptr);
}

bool StringNode::SetAbstract(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractScalar>()) {
    return false;
  }

  return StubNode::SetAbstract(abs);
}

void StringNode::SetValue(const ValuePtr &val) {
  MS_EXCEPTION_IF_NULL(val);
  if (!val->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a StringImm value for StringNode, but got: " << val->ToString();
  }
  StubNode::SetValue(val);
}

bool ScalarNode::SetAbstract(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractScalar>()) {
    return false;
  }

  return StubNode::SetAbstract(abs);
}

bool AnyTypeNode::SetAbstract(const AbstractBasePtr &abs) {
  real_node_ = MakeStubNode(abs->BuildType());
  auto flag = real_node_->SetAbstract(abs);
  (void)StubNode::SetAbstract(abs);
  return flag;
}

void AnyTypeNode::SetValue(const ValuePtr &val) {
  real_node_->SetValue(val);
  StubNode::SetValue(val);
}

py::object AnyTypeNode::GetRealNode() {
  (void)WaitPipeline();
  return py::cast(real_node_);
}

py::object NoneTypeNode::GetRealValue() {
  auto val = WaitValue();
  return ValueToPyData(val);
}

void AnyTypeNode::SetException(const std::exception_ptr &e_ptr) {
  StubNode::SetException(e_ptr);
  if (real_node_ != nullptr) {
    real_node_->SetException(e_ptr);
  }
}

std::pair<py::object, StubNodePtr> MakeTopNode(const TypePtr &type) {
  auto top = MakeStubNode(type);
  auto ret = MakeOutput(top);
  return std::make_pair(ret, top);
}

std::pair<StubNodePtr, bool> MakeStubNode(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Make stub node for Abstract: " << abs->ToString();
  const auto &type = abs->GetType();
  MS_EXCEPTION_IF_NULL(type);
  const auto &shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(shape);

  if (type->IsSameTypeId(TensorType::kTypeId)) {
    auto node = std::make_shared<TensorNode>();
    if (!shape->IsDynamic()) {
      node->SetAbstract(abs);
    }
    return {node, true};
  } else if (type->IsSameTypeId(Tuple::kTypeId) || type->IsSameTypeId(List::kTypeId)) {
    const auto &seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    if (seq_abs->dynamic_len()) {
      MS_LOG(INFO) << "Can not support dynamic tuple/list type to create a StubNode.";
      return {nullptr, false};
    }
    const auto &elements = seq_abs->elements();
    auto node = std::make_shared<SequenceNode>(elements.size());
    node->StubNode::SetAbstract(seq_abs);
    for (size_t i = 0; i < elements.size(); ++i) {
      auto elem_pair = MakeStubNode(elements[i]);
      if (!elem_pair.second) {
        return {nullptr, false};
      }
      MS_EXCEPTION_IF_NULL(elem_pair.first);
      node->SetElement(i, elem_pair.first);
    }
    return {node, true};
  } else if (type->IsSameTypeId(String::kTypeId)) {
    MS_LOG(DEBUG) << "Make stub string node";
    auto str_node = std::make_shared<StringNode>();
    const auto &string_value = abs->GetValue();
    if (string_value && string_value->isa<StringImm>()) {
      str_node->SetValue(string_value);
      MS_LOG(DEBUG) << "Make stub string node, value: " << string_value->ToString();
    } else {
      MS_LOG(INFO) << "There is no string value in abstract: " << abs->ToString();
      return {nullptr, false};
    }
    if (!str_node->SetAbstract(abs)) {
      MS_LOG(INFO) << "Set abstract for StringNode failed, abstract: " << abs->ToString();
      return {nullptr, false};
    }
    return {str_node, true};
  } else if (type->IsFromTypeId(Number::kTypeId)) {
    MS_LOG(DEBUG) << "Make stub scalar node";
    auto scalar_node = std::make_shared<ScalarNode>();
    const auto &scalar_value = abs->GetValue();
    if (scalar_value && !scalar_value->isa<ValueAny>()) {
      scalar_node->SetValue(scalar_value);
      MS_LOG(DEBUG) << "Make stub scalar node, value: " << scalar_node->ToString();
    }
    if (!scalar_node->SetAbstract(abs)) {
      MS_LOG(INFO) << "Set abstract for ScalarNode failed, abstract: " << abs->ToString();
      return {nullptr, false};
    }
    return {scalar_node, true};
  }
  MS_LOG(INFO) << "Unsupported stub tensor for type: " << type->ToString();
  return {nullptr, false};
}

void FlattenStubNode(const StubNodePtr &node, std::vector<StubNodePtr> *flatten_stub_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(flatten_stub_nodes);
  auto seq = node->cast<std::shared_ptr<SequenceNode>>();
  if (seq) {
    auto &elements = seq->Elements();
    for (size_t i = 0; i < elements.size(); ++i) {
      FlattenStubNode(elements[i], flatten_stub_nodes);
    }
  } else {
    flatten_stub_nodes->push_back(node);
  }
}
}  // namespace stub
}  // namespace mindspore
