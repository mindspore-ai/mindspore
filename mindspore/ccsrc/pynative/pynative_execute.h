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

#ifndef MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_
#define MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>

#include "pybind11/pybind11.h"

#include "transform/convert.h"
#include "transform/graph_runner.h"
#include "transform/types.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace pynative {

using MeTensor = mindspore::tensor::Tensor;
using MeTensorPtr = mindspore::tensor::TensorPtr;
using GeTensor = ge::Tensor;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using GeGraph = ge::Graph;
using GeGraphPtr = std::shared_ptr<GeGraph>;
using GeOperator = ge::Operator;
using GeOperatorPtr = std::shared_ptr<GeOperator>;

namespace py = pybind11;

enum PynativeStatusCode {
  PYNATIVE_SUCCESS = 0,
  PYNATIVE_OP_NOT_IMPLEMENTED_ERR = 1,
  PYNATIVE_OP_INPUTS_ERR = 2,
  PYNATIVE_OP_PARAMS_ERR = 3,
  PYNATIVE_OP_ATTRS_ERR = 4,
  PYNATIVE_GRAPH_MANAGER_ERR = 5,
  PYNATIVE_GRAPH_GE_BUILD_ERR = 6,
  PYNATIVE_GRAPH_GE_RUN_ERR = 7,
  PYNATIVE_UNKNOWN_STATE = 0XFF
};

enum RunOpArgsEnum { PY_PRIM = 0, PY_NAME, PY_INPUTS, PY_INPUT_MASK, PY_ARGS_NUM };

struct OpExecInfo {
  PrimitivePyPtr py_primitive;
  std::string op_name;
  AbstractBasePtr abstract;

  py::tuple op_inputs;
  py::tuple inputs_mask;
  py::dict op_attrs;
};
using OpExecInfoPtr = std::shared_ptr<OpExecInfo>;
OpExecInfoPtr GenerateOpExecInfo(const py::args& args);
bool BuildSingleOpGraph(const OpExecInfoPtr& op_exec_info, const std::vector<GeTensorPtr>& inputs,
                        const std::unordered_map<std::string, ValuePtr>& attrs, const GeGraphPtr& graph);

py::object RunOpInGE(const OpExecInfoPtr& op_exec_info, PynativeStatusCode* status);

py::object RunOpInVM(const OpExecInfoPtr& op_exec_info, PynativeStatusCode* status);

py::tuple RunOp(const py::args& args);
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_PYNATIVE_EXECUTE_H_
