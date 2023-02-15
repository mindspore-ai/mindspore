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
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/stub_tensor.h"

namespace mindspore {
namespace stub {
namespace {
std::condition_variable stub_cond_var_;
std::mutex stub_mutex_;

StubNodePtr MakeStubNode(const TypePtr &type, StubNode *top = nullptr) {
  if (type->isa<Tuple>() || type->isa<List>()) {
    TypePtrList elements;
    if (type->isa<Tuple>()) {
      auto tuple_type = type->cast<TuplePtr>();
      elements = tuple_type->elements();
    } else {
      auto list_type = type->cast<ListPtr>();
      elements = list_type->elements();
    }
    auto node = std::make_shared<SequenceNode>(top, elements.size());
    StubNode *real_top = top ? top : node.get();
    for (size_t i = 0; i < elements.size(); ++i) {
      node->SetElement(i, MakeStubNode(elements[i], real_top));
    }
    return node;
  } else {
    if (!type->isa<TensorType>()) {
      MS_LOG(WARNING) << "stub tensor is create for type: " << type->ToString();
    }
    return std::make_shared<TensorNode>(top);
  }
  return nullptr;
}

std::pair<int, py::object> MakeOutput(StubNodePtr node) {
  if (node->isa<TensorNode>()) {
    auto tensor = node->cast<std::shared_ptr<TensorNode>>();
    return std::make_pair(static_cast<int>(StubNode::TENSOR), py::cast(tensor));
  } else {
    auto seq = node->cast<std::shared_ptr<SequenceNode>>();
    MS_EXCEPTION_IF_NULL(seq);
    auto &elements = seq->Elements();
    py::tuple out(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      auto ret = MakeOutput(elements[i]);
      out[i] = ret.second;
    }
    return std::make_pair(static_cast<int>(StubNode::TUPLE), out);
  }
}
}  // namespace

void StubNode::SetAbstract(const AbstractBasePtr &abs) {
  abstract_ = abs;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
}

void StubNode::SetValue(const ValuePtr &val) {
  value_ = val;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
}

AbstractBasePtr StubNode::WaitAbstract() {
  if (abstract_.get() == nullptr) {
    if (top_node_) {
      top_node_->WaitAbstract();
    } else {
      wait_flag_.store(true);
      std::unique_lock<std::mutex> lock(stub_mutex_);
      stub_cond_var_.wait(lock, [this] { return abstract_.get() == nullptr; });
      wait_flag_.store(false);
    }
  }
  return abstract_;
}

ValuePtr StubNode::WaitValue() {
  if (value_.get() == nullptr) {
    if (top_node_) {
      top_node_->WaitValue();
    } else {
      wait_flag_.store(true);
      std::unique_lock<std::mutex> lock(stub_mutex_);
      stub_cond_var_.wait(lock, [this] { return value_.get() == nullptr; });
      wait_flag_.store(false);
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
  MS_EXCEPTION_IF_NULL(shape);
  auto &shape_vector = shape->shape();
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

void SequenceNode::SetAbstract(const AbstractBasePtr &abs) {
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  auto children = seq_abs->elements();
  if (elements_.empty()) {
    for (auto child : children) {
      elements_.emplace_back(MakeStubNode(child->BuildType()));
    }
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    elements_[i]->SetAbstract(children[i]);
  }
  StubNode::SetAbstract(abs);
}

void SequenceNode::SetValue(const ValuePtr &val) {
  auto seq_value = val->cast<ValueSequencePtr>();
  auto children = seq_value->value();
  for (size_t i = 0; i < elements_.size(); ++i) {
    elements_[i]->SetValue(children[i]);
  }
  StubNode::SetValue(val);
}

std::pair<py::object, StubNodePtr> MakeTopNode(const TypePtr &type) {
  auto top = MakeStubNode(type, nullptr);
  auto ret = MakeOutput(top);
  return std::make_pair(py::make_tuple(ret.first, ret.second), top);
}

void RegStubNodes(const py::module *m) {
  (void)py::class_<StubNode, std::shared_ptr<StubNode>>(*m, "StubNode");
  (void)py::class_<TensorNode, StubNode, std::shared_ptr<TensorNode>>(*m, "TensorNode")
    .def("get_value", &TensorNode::GetValue, "get output value of async stub.")
    .def("get_shape", &TensorNode::GetShape, "get output shape of async stub.")
    .def("get_dtype", &TensorNode::GetDtype, "get output dtype of async stub.");
}
}  // namespace stub
}  // namespace mindspore
