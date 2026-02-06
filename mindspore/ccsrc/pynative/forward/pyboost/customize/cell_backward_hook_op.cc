/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "pynative/backward/grad_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/pynative/forward/pyboost/converter.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pynative/utils/predict_out_type_map.h"
#include "include/pynative/forward/pyboost/forward_task.h"
#include "include/pynative/forward/pyboost/fallback.h"
#include "primitive/auto_generate/gen_ops_def.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/functions/base.h"
#include "include/pynative/utils/pyboost/auto_generate/cell_backward_hook.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "pybind_api/pynative/pynative_api.h"

namespace mindspore::pynative {
py::object PyboostCellBackwardHookBase(const PrimitivePtr &prim, const py::list &args) {
#ifndef ENABLE_TEST
  MS_LOG(DEBUG) << "Run Pyboost_CellBackwardHook start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  static pynative::Converter converter(&ops::gCellBackwardHook);
  converter.Parse(args.ptr());
  auto tensors = converter.ToTensorList<pynative::CPythonTuple>(args.ptr(), kIndex0);
  MS_EXCEPTION_IF_NULL(tensors);
  const auto &values = tensors->value();
  if (fallback_enabled() && std::any_of(values.begin(), values.end(), [](const ValuePtr &value) {
        MS_EXCEPTION_IF_NULL(value);
        return value->isa<Tensor>() && value->cast<TensorPtr>()->has_fallback();
      })) {
    auto op_call = std::make_shared<OpCall>("CellBackwardHook", [prim](const py::args &args, const py::kwargs &kwargs) {
      auto list_args = py::list(args);
      return PyboostCellBackwardHookBase(prim, list_args);
    });
    return py::reinterpret_steal<py::object>(pynative::HandleFallback(args.ptr(), py::cast(op_call)));
  }

  static auto op_type = kernel::pyboost::GetOpTypeFromOpdef(ops::gCellBackwardHook);
  op_run_info->source_type = converter.source_type();

  {
    kernel::pyboost::OpRunStatus::Get().set_run_info(
      kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision, op_run_info->device_target));
    GilReleaseWithCheck no_gil;
    runtime::Pipeline::Get().frontend_stage()->Wait();
  }

  MS_LOG(DEBUG) << "Run frontend task Pyboost_CellBackwardHook start";

  // stub tensor to tensor.
  auto tensors_tensor_list =
    PyNativeAlgo::Common::ConvertStubNodeToValueTuple(tensors, true, op_run_info->requires_grad);

  // Create op
  auto op = CREATE_PYBOOST_OP(CellBackwardHook, op_run_info->device_target);
  op->set_primitive(prim);
  // Run op
  {
    kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);
    (void)op->Call(tensors_tensor_list);
  }
  // Create output value
  auto real_out = AutoGradUtil::MakeMultiOutput(op_run_info->requires_grad, op);
  // Do auto grad
  if (op_run_info->requires_grad) {
    auto op_grad_info =
      std::make_shared<OpGradInfo>(op_type, op->primitive(), std::vector<ValuePtr>({tensors_tensor_list}), real_out);
    AutoGradUtil::SetInferMultiOutputToGrad(op_grad_info, op);
    PyNativeAlgo::PyBoost::DoGrad(op, op_grad_info, op_run_info->async_status);
  }

  MS_LOG(DEBUG) << "Run Pyboost_CellBackwardHook end";
  return py::reinterpret_steal<py::object>(tensor::Wrap(real_out));
#else
  return PyNativeAlgo::PyBoost::RunPyFunction(prim, args);
#endif
}

py::object Pyboost_CellBackwardHook(const py::args &args) {
  if (args.size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Two args are needed by RunOp"
                      << ", but got " << args.size();
  }
  const auto &prim = PyNativeAlgo::PyBoost::ConvertPrimitive(args[0]);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, prim->name(),
                                     false, true);
  return PyboostCellBackwardHookBase(prim, args[1]);
}

void RegisterCellBackwardHookFunction(py::module *m) {
  m->def("pyboost_cell_backward_hook", &mindspore::pynative::Pyboost_CellBackwardHook, "CellBackwardHook Ops");
}
}  // namespace mindspore::pynative
