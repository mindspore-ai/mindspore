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
#include <condition_variable>
#include <mutex>
#include "utils/ms_exception.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/stub_tensor.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace stub {
namespace {
std::condition_variable stub_cond_var_;
std::mutex stub_mutex_;

StubNodePtr MakeStubNode(const TypePtr &type) {
  if (type->isa<Tuple>() || type->isa<List>()) {
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
      elem->SetTopNode(node);
      node->SetElement(i, elem);
    }
    return node;
  } else {
    if (!type->isa<TensorType>()) {
      MS_LOG(WARNING) << "stub tensor is create for type: " << type->ToString();
    }
    return std::make_shared<TensorNode>();
  }
  return nullptr;
}

py::object MakeOutput(StubNodePtr node) {
  if (node->isa<TensorNode>()) {
    auto tensor = node->cast<std::shared_ptr<TensorNode>>();
    return py::cast(tensor);
  } else {
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
  }
}
}  // namespace

bool StubNode::SetAbstract(const AbstractBasePtr &abs) {
  abstract_ = abs;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
  return true;
}

void StubNode::SetValue(const ValuePtr &val) {
  value_ = val;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
}

void StubNode::SetException(const std::exception_ptr &e_ptr) {
  e_ptr_ = e_ptr;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
}

AbstractBasePtr StubNode::WaitAbstract() {
  GilReleaseWithCheck gil_release;
  if (abstract_.get() == nullptr) {
    auto top = top_node_;
    if (top) {
      top->WaitAbstract();
    } else {
      wait_flag_.store(true);
      std::unique_lock<std::mutex> lock(stub_mutex_);
      stub_cond_var_.wait(lock, [this] { return abstract_.get() != nullptr || e_ptr_ != nullptr; });
      wait_flag_.store(false);
      if (e_ptr_ != nullptr) {
        // Need to clear exception in the instance.
        MsException::Instance().CheckException();
        std::rethrow_exception(e_ptr_);
      }
    }
  }
  return abstract_;
}

ValuePtr StubNode::WaitValue() {
  GilReleaseWithCheck gil_release;
  if (value_.get() == nullptr) {
    auto top = top_node_;
    if (top) {
      top->WaitValue();
    } else {
      wait_flag_.store(true);
      std::unique_lock<std::mutex> lock(stub_mutex_);
      stub_cond_var_.wait(lock, [this] { return value_.get() != nullptr || e_ptr_ != nullptr; });
      wait_flag_.store(false);
      if (e_ptr_ != nullptr) {
        // Need to clear exception in the instance.
        MsException::Instance().CheckException();
        std::rethrow_exception(e_ptr_);
      }
    }
  }
  return value_;
}

py::object TensorNode::GetValue() {
  auto val = WaitValue();
  return ValueToPyData(val);
}

py::object TensorNode::GetShape() {
  auto abs = WaitAbstract();
  auto base = abs->BuildShape();
  auto shape = base->cast<abstract::ShapePtr>();
  ShapeVector shape_vector;
  if (shape && !shape->IsDynamic()) {
    shape_vector = shape->shape();
  } else {
    auto val = WaitValue();
    auto tensor = val->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    shape_vector = tensor->shape();
  }
  auto ret = py::tuple(shape_vector.size());
  for (size_t i = 0; i < shape_vector.size(); ++i) {
    ret[i] = shape_vector[i];
  }
  return ret;
}

py::object TensorNode::GetDtype() {
  auto abs = WaitAbstract();
  auto base = abs->BuildType();
  if (base->isa<TensorType>()) {
    base = base->cast<TensorTypePtr>()->element();
  }
  return py::cast(base);
}

bool TensorNode::SetAbstract(const AbstractBasePtr &abs) {
  if (!abs->isa<abstract::AbstractTensor>() && !abs->isa<abstract::AbstractMapTensor>()) {
    if (!abs->isa<abstract::AbstractScalar>() || abs->BuildValue() != kAnyValue) {
      return false;
    }
  }
  return StubNode::SetAbstract(abs);
}

py::object SequenceNode::GetElements() {
  if (elements_.empty()) {
    (void)WaitAbstract();
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
  if (elements_.empty()) {
    for (auto child : children) {
      elements_.emplace_back(MakeStubNode(child->BuildType()));
    }
  }
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

void SequenceNode::SetValue(const ValuePtr &val) {
  auto seq_value = val->cast<ValueSequencePtr>();
  auto children = seq_value->value();
  for (size_t i = 0; i < children.size(); ++i) {
    elements_[i]->SetValue(children[i]);
    elements_[i]->SetTopNode(nullptr);
  }
  StubNode::SetValue(val);
}

std::pair<py::object, StubNodePtr> MakeTopNode(const TypePtr &type) {
  auto top = MakeStubNode(type);
  auto ret = MakeOutput(top);
  return std::make_pair(ret, top);
}

void RegStubNodes(const py::module *m) {
  (void)py::class_<StubNode, std::shared_ptr<StubNode>>(*m, "StubNode");
  (void)py::class_<TensorNode, StubNode, std::shared_ptr<TensorNode>>(*m, "TensorNode")
    .def("get_value", &TensorNode::GetValue, "get output value of async stub.")
    .def("get_shape", &TensorNode::GetShape, "get output shape of async stub.")
    .def("get_dtype", &TensorNode::GetDtype, "get output dtype of async stub.");
  (void)py::class_<SequenceNode, StubNode, std::shared_ptr<SequenceNode>>(*m, "SequenceNode")
    .def("get_elements", &SequenceNode::GetElements, "get the elements of async stub_seq.");
}
}  // namespace stub
}  // namespace mindspore
