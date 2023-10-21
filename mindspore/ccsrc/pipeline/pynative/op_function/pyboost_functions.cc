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
#include "include/common/pybind_api/api_register.h"
#include "pipeline/pynative/forward/forward.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/op_function/python_arg_parser.h"
#include "kernel/pyboost/op/baddbmm.h"
#include "kernel/pyboost/op/bias_add.h"
#include "kernel/pyboost/op/square.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore::ops {
extern OpDef gBaddbmm;
}
namespace mindspore::pynative {
py::object Pyboost_Baddbmm(const py::args &args) {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunOp);
  auto op_run_info = PyNativeAlgo::PyBoost::Init(args);
  static Parser parser(&ops::gBaddbmm);
  py::list input_args = args[kIndex1];
  parser.Parse(input_args);
  auto input = parser.ToTensor(kIndex0);
  auto batch1 = parser.ToTensor(kIndex1);
  auto batch2 = parser.ToTensor(kIndex2);
  auto beta = parser.ToScalar(kIndex3);
  auto alpha = parser.ToScalar(kIndex4);
  auto op = CREATE_PYBOOST_OP(Baddbmm, op_run_info->base_op_run_info.device_target);
  if (op_run_info->requires_grad) {
    op->set_grad_func([op_run_info](const std::vector<ValuePtr> &inputs, const std::vector<TensorPtr> &output,
                                    const std::vector<abstract::AbstractBasePtr> &input_abs,
                                    const AbstractBasePtr &output_abs) {
      PyNativeAlgo::PyBoost::DoGrad(op_run_info, inputs, output, input_abs, output_abs);
    });
  }
  op->set_primitive(op_run_info->op_grad_info->op_prim);
  auto output = op->Call(input, batch1, batch2, beta, alpha);
  MS_LOG(DEBUG) << "Run Pyboost_Baddbmm end";
  return parser.Wrap(output);
}

py::object Pyboost_BiasAdd(const py::args &args) {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunOp);
  if (args.size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Three args are needed by RunOp";
  }
  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &grad_executor = pynative_executor->grad_executor();
  forward_executor->Init();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  // Used for async run
  op_run_info->requires_grad = grad_executor->RequiresGrad();
  if (op_run_info->requires_grad) {
    op_run_info->base_op_run_info.use_dynamic_shape_process = grad_executor->use_dynamic_shape_process();
  } else {
    op_run_info->base_op_run_info.use_dynamic_shape_process =
      grad_executor->forward_use_dynamic_shape_process() || grad_executor->use_dynamic_shape_process();
  }
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[0]);
  op_run_info->base_op_run_info.device_target =
    forward_executor->GetCurrentDeviceTarget(op_run_info->op_grad_info->op_prim);
  op_run_info->cell_obj_id = forward_executor->GetCurrentCellObjId();
  // static Parser parser(&ops::gBaddbmm);
  //
  // parser.Parse(input_args);
  // auto input_x = parser.ToTensor(kIndex0);
  // auto bias = parser.ToTensor(kIndex1);

  auto convert_tensor = [](const py::object &obj) -> tensor::TensorPtr {
    if (!py::isinstance<tensor::Tensor>(obj)) {
      return nullptr;
    }
    return obj.cast<tensor::TensorPtr>();
  };

  py::list input_args = args[1];
  auto input_x = convert_tensor(input_args[kIndex0]);
  auto bias = convert_tensor(input_args[kIndex1]);

  auto op = CREATE_PYBOOST_OP(BiasAdd, op_run_info->base_op_run_info.device_target);
  op->set_primitive(std::make_shared<Primitive>(*op_run_info->op_grad_info->op_prim));
  auto output = op->Call(input_x, bias);
  // return parser.Wrap(output);
  py::tuple v(1);
  v[0] = output;
  return v[0];
}

py::object Pyboost_Square(const py::args &args) {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunOp);
  if (args.size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Three args are needed by RunOp";
  }
  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &grad_executor = pynative_executor->grad_executor();
  forward_executor->Init();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  // Used for async run
  op_run_info->requires_grad = grad_executor->RequiresGrad();
  if (op_run_info->requires_grad) {
    op_run_info->base_op_run_info.use_dynamic_shape_process = grad_executor->use_dynamic_shape_process();
  } else {
    op_run_info->base_op_run_info.use_dynamic_shape_process =
      grad_executor->forward_use_dynamic_shape_process() || grad_executor->use_dynamic_shape_process();
  }
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[0]);
  op_run_info->base_op_run_info.device_target =
    forward_executor->GetCurrentDeviceTarget(op_run_info->op_grad_info->op_prim);
  op_run_info->cell_obj_id = forward_executor->GetCurrentCellObjId();
  // static Parser parser(&ops::gBaddbmm);
  //
  // parser.Parse(input_args);
  // auto input_x = parser.ToTensor(kIndex0);
  // auto bias = parser.ToTensor(kIndex1);

  auto convert_tensor = [](const py::object &obj) -> tensor::TensorPtr {
    if (!py::isinstance<tensor::Tensor>(obj)) {
      return nullptr;
    }
    return obj.cast<tensor::TensorPtr>();
  };

  py::list input_args = args[1];
  auto input = convert_tensor(input_args[kIndex0]);

  auto op = CREATE_PYBOOST_OP(Square, op_run_info->base_op_run_info.device_target);
  op->set_primitive(std::make_shared<Primitive>(*op_run_info->op_grad_info->op_prim));
  auto output = op->Call(input);
  // return parser.Wrap(output);
  py::tuple v(1);
  v[0] = output;
  return v[0];
}

void RegisterPyBoostFunction(py::module *m) {
  m->def("pyboost_baddbmm", &mindspore::pynative::Pyboost_Baddbmm, "Encrypt the data.");
  m->def("pyboost_bias_add", &mindspore::pynative::Pyboost_BiasAdd, "Encrypt the data.");
  m->def("pyboost_square", &mindspore::pynative::Pyboost_Square, "Encrypt the data.");
}
}  // namespace mindspore::pynative
