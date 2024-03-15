/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CONVERTER_H
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CONVERTER_H
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_execute.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/utils/primfunc_utils.h"
#include "ops/op_def.h"

namespace mindspore {
namespace pynative {
class Converter {
 public:
  explicit Converter(ops::OpDef *op_def);
  void Parse(const py::list &python_args);
  ValuePtr ToTensor(const py::list &python_args, size_t i);
  std::optional<ValuePtr> ToTensorOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToTensorList(const py::list &python_args, size_t i);
  Int64ImmPtr ToInt(const py::list &python_args, size_t i);
  std::optional<Int64ImmPtr> ToIntOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToIntList(const py::list &python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToIntListOptional(const py::list &python_args, size_t i);
  BoolImmPtr ToBool(const py::list &python_args, size_t i);
  std::optional<BoolImmPtr> ToBoolOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToBoolList(const py::list &python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToBoolListOptional(const py::list &python_args, size_t i);
  FP32ImmPtr ToFloat(const py::list &python_args, size_t i);
  std::optional<FP32ImmPtr> ToFloatOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToFloatList(const py::list &python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToFloatListOptional(const py::list &python_args, size_t i);
  ScalarPtr ToScalar(const py::list &python_args, size_t i);
  std::optional<ScalarPtr> ToScalarOptional(const py::list &python_args, size_t i);
  StringImmPtr ToString(const py::list &python_args, size_t i);
  std::optional<StringImmPtr> ToStringOptional(const py::list &python_args, size_t i);
  Int64ImmPtr ToDtype(const py::list &python_args, size_t i);
  std::optional<Int64ImmPtr> ToDtypeOptional(const py::list &python_args, size_t i);
  ValuePtr ConvertByCastDtype(const py::object &input, const ops::OpInputArg &op_arg, size_t i);
  ValueTuplePtr ConvertValueTupleByCastDtype(const py::list &python_args, const ops::OpInputArg &op_arg, size_t index);
  const std::vector<ops::OP_DTYPE> &source_type() const { return source_type_; }

 private:
  ops::OpDefPtr op_def_;
  // If op not type cast, source_type is default type: DT_BEGIN, if op type cast, source_type is origin type.
  std::vector<ops::OP_DTYPE> source_type_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CONVERTER_H
