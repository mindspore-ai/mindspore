/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "pybind_api/utils/stub_tensor_py.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/pynative/pynative_utils.h"

namespace mindspore {
namespace py = pybind11;

void RegPyNativeAsyncStub(const py::module *m) {
  (void)py::class_<StubNode, std::shared_ptr<StubNode>>(*m, "StubNode")
    .def("get_value", &StubNode::GetValue, "get output value of async stub.")
    .def("get_shape", &StubNode::GetShape, "get output shape of async stub.")
    .def("get_dtype", &StubNode::GetDtype, "get output dtype of async stub.");
}

ShapeVector StubNode::GetShapeVector() {
  auto base = abs->BuildShape();
  auto shape = base->cast<abstract::ShapePtr>();
  if (!shape) {
    MS_LOG(EXCEPTION) << "Only Tensor shape is supported by Stub now: " << base->ToString();
  }
  return shape->shape();
}

TypePtr StubNode::GetTypePtr() {
  auto base = abs->BuildType();
  if (base->isa<TensorType>()) {
    return base->cast<TensorTypePtr>()->element();
  }
  return base;
}

py::object StubNode::GetValue() { return pynative::PyNativeAlgo::DataConvert::ValueToPyObj(value); }

py::object StubNode::GetShape() {
  auto shape_vector = GetShapeVector();
  auto ret = py::tuple(shape_vector.size());
  for (size_t i = 0; i < shape_vector.size(); ++i) {
    ret[i] = shape_vector[i];
  }
  return ret;
}

py::object StubNode::GetDtype() { return py::cast(GetTypePtr()); }

py::object StubOutConverter::Convert(const abstract::AbstractBasePtr &abs, const ValuePtr &value) {
  py::object result;
  if (abs->isa<abstract::AbstractTensor>() || (value && value->isa<tensor::Tensor>())) {
    // In `TensorArray` case, abstract is AbstractScalar and value is Tensor.
    result = ConvertTensor(abs, value);
    root_type_ = static_cast<int>(StubNode::TENSOR);
  } else if (abs->isa<abstract::AbstractTuple>()) {
    result = ConvertTuple(abs->cast<abstract::AbstractTuplePtr>(), value);
    root_type_ = static_cast<int>(StubNode::TUPLE);
  } else if (abs->isa<abstract::AbstractList>()) {
    result = ConvertList(abs->cast<abstract::AbstractListPtr>(), value);
    // Should we create StubNode::LIST? Otherwise, this list output will be cast to tuple in python.
    root_type_ = static_cast<int>(StubNode::TUPLE);
  } else if (abs->isa<abstract::AbstractScalar>() || abs->isa<abstract::AbstractType>() ||
             abs->isa<abstract::AbstractSlice>()) {
    // Here are some types that `output_get_by_infer_value == true`
    result = ConvertScalar(abs);
    root_type_ = static_cast<int>(StubNode::SCALAR);
  } else {
    MS_LOG(EXCEPTION) << "StubOutConverter cannot handle this type of abstract: " << abs->ToString();
  }
  return result;
}

py::object StubOutConverter::ConvertTensor(const abstract::AbstractBasePtr &tensor_abs, const ValuePtr &value) {
  auto stub = std::make_shared<StubNode>();
  stub->value = value;
  stub->abs = tensor_abs;
  return py::cast(stub);
}

py::object StubOutConverter::ConvertTuple(const abstract::AbstractTuplePtr &seq_abs, const ValuePtr &value) {
  auto elements = seq_abs->elements();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "value and abs not match: value " << value->ToString() << " vs abstract "
                      << seq_abs->ToString();
  }
  auto seq_value = value->cast<ValueTuplePtr>();
  if (seq_value->size() > seq_abs->size()) {
    MS_LOG(EXCEPTION) << "Cannot convert, abstract size must greater or equal to value size: " << seq_value->size()
                      << " vs " << seq_abs->size();
  }
  py::tuple out(seq_value->size());
  for (size_t i = 0; i < seq_value->size(); ++i) {
    out[i] = Convert(elements[i], seq_value->value()[i]);
  }
  return out;
}

py::object StubOutConverter::ConvertList(const abstract::AbstractListPtr &seq_abs, const ValuePtr &value) {
  auto elements = seq_abs->elements();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueList>()) {
    MS_LOG(EXCEPTION) << "value and abs not match: value " << value->ToString() << " vs abstract "
                      << seq_abs->ToString();
  }
  auto seq_value = value->cast<ValueListPtr>();
  if (seq_value->size() > seq_abs->size()) {
    MS_LOG(EXCEPTION) << "Cannot convert, abstract size must greater or equal to value size: " << seq_value->size()
                      << " vs " << seq_abs->size();
  }
  py::list out(seq_value->size());
  for (size_t i = 0; i < seq_value->size(); ++i) {
    out[i] = Convert(elements[i], seq_value->value()[i]);
  }
  return out;
}

py::object StubOutConverter::ConvertScalar(const abstract::AbstractBasePtr &abs) {
  return pynative::PyNativeAlgo::DataConvert::ValueToPyObj(abs->BuildValue());
}
}  // namespace mindspore
