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

#ifndef MINDSPORE_CCSRC_UTILS_STUB_TENSOR_PY_H_
#define MINDSPORE_CCSRC_UTILS_STUB_TENSOR_PY_H_
#include <string>
#include <functional>
#include <memory>

#include "pybind11/pybind11.h"
#include "base/base.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace py = pybind11;

namespace stub {
constexpr auto PY_ATTR_STUB = "stub";
}  // namespace stub

void RegPyNativeAsyncStub(const py::module *m);

class StubNode {
 public:
  enum { TENSOR = 0, CSR_TENSOR, COO_TENSOR, ROW_TENSOR, TUPLE, SCALAR, NOT_SUPPORT };
  StubNode() {}
  ~StubNode() {}
  abstract::AbstractBasePtr abs;
  ValuePtr value;

  // Api for python StubTensor object
  py::object GetValue();
  py::object GetShape();
  py::object GetDtype();

 private:
  ShapeVector GetShapeVector();
  TypePtr GetTypePtr();
};
using StubNodePtr = std::shared_ptr<StubNode>;

class StubOutConverter {
 public:
  StubOutConverter() {}
  ~StubOutConverter() {}
  py::object Convert(const abstract::AbstractBasePtr &abs, const ValuePtr &value = nullptr);
  int GetRootType() { return root_type_; }

 private:
  py::object ConvertTensor(const abstract::AbstractBasePtr &tensor_abs, const ValuePtr &value);
  py::object ConvertScalar(const abstract::AbstractBasePtr &scalar_abs);
  py::object ConvertTuple(const abstract::AbstractTuplePtr &seq_abs, const ValuePtr &value);
  py::object ConvertList(const abstract::AbstractListPtr &seq_abs, const ValuePtr &value);
  int root_type_{static_cast<int>(StubNode::NOT_SUPPORT)};
};

using StubOutConverterPtr = std::shared_ptr<StubOutConverter>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_STUB_TENSOR_PY_H_
