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
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_execute.h"
#include "include/common/pybind_api/api_register.h"
#include "ops/op_def.h"
namespace mindspore {
namespace pynative {
class Parser {
 public:
  explicit Parser(const ops::OpDef &op_def);
  void Parse(py::list args);
  ValuePtr ToTensor(size_t i);
  template <typename T>
  ValueTuplePtr ToTensorList(size_t i);
  Int64ImmPtr ToInt(size_t i);
  std::optional<Int64ImmPtr> ToIntOptional(size_t i);
  template <typename T>
  ValueTuplePtr ToIntList(size_t i);
  BoolImmPtr ToBool(size_t i);
  template <typename T>
  ValueTuplePtr ToBoolList(size_t i);
  FP32ImmPtr ToFloat(size_t i);
  template <typename T>
  ValueTuplePtr ToFloatList(size_t i);
  ScalarPtr ToScalar(size_t i);
  TypePtr ToDtype(size_t i);
  py::object Wrap(const TensorPtr &tensor);

 private:
  void ThrowException(size_t i);
  ops::OpDef op_def_;
  py::list *python_args_;
};
}  // namespace pynative
}  // namespace mindspore
