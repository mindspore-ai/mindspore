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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_

#include <memory>
#include <string>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace pynative {
class PyNativeExecutor;

namespace PyNativeAlgo {
// Common function
struct Common {
  static std::string GetIdByValue(const ValuePtr &v);
  static TypePtr GetTypeFromAbstract(const abstract::AbstractBasePtr &abs);
  static bool IsDynamicShape(const FrontendOpRunInfoPtr &op_run_info);
  static bool ValueHasDynamicShape(const ValuePtr &value);
  static std::shared_ptr<PyNativeExecutor> GetPyNativeExecutor();
};

// Parser python
struct PyParser {
  static std::string GetPyObjId(const py::handle &obj);
  static std::string GetIdByPyObj(const py::object &obj);
  static size_t GetTupleSize(const py::tuple &args);
  static py::list FilterTensorArgs(const py::args &args, bool has_sens = false);
  static void SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg);
  static void ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs);
};

// Data convert
struct DataConvert {
  static ValuePtr PyObjToValue(const py::object &obj);
  static ValuePtr BaseRefToValue(const BaseRef &value);
  static ValuePtr VectorRefToValue(const VectorRef &vec_ref);
  static void ConvertTupleArg(py::tuple *res, size_t *const index, const py::tuple &arg);
  static py::tuple ConvertArgs(const py::tuple &args);
  static void GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target);
  static void ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                           const tensor::CSRTensorPtr &csr_tensor, const PrimitivePtr &op_prim);
  static void ConvertValueTupleToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq);
  static void PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                       const PrimitivePtr &op_prim, size_t index);
  static void ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                        const PrimitivePtr &op_prim, size_t index);
  static void ConvertValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                                   const PrimitivePtr &op_prim);
  static bool NeedConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target,
                                          mindspore::HashSet<size_t> *input_to_attr_ptr);
  static bool RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                           size_t input_index, const PrimitivePtr &op_prim,
                                           const mindspore::HashSet<size_t> &input_attrs);
};
};  // namespace PyNativeAlgo
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_UTILS_H_
