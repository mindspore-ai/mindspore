/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "primitive/auto_generate/gen_ops_def.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/pynative/utils/pyboost/functions/base.h"
#include "include/pynative/utils/pyboost/auto_generate/custom_ext.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "pybind_api/pynative/pynative_api.h"

namespace mindspore::pynative {
py::object PyboostCustomExtBase(const PrimitivePtr &prim, const py::list &args) {
#ifndef ENABLE_TEST
  MS_LOG(DEBUG) << "Run Pyboost_CustomExt start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  static pynative::Converter converter(&ops::gCustomExt);
  converter.Parse(args.ptr());
  auto tensors = converter.ToTensorList<pynative::CPythonTuple>(args.ptr(), kIndex0);

  static auto op_type = kernel::pyboost::GetOpTypeFromOpdef(ops::gCustomExt);
  op_run_info->source_type = converter.source_type();

  {
    GilReleaseWithCheck no_gil;
    runtime::Pipeline::Get().frontend_stage()->Wait();
  }

  MS_LOG(DEBUG) << "Run frontend task Pyboost_CustomExt start";

  // stub tensor to tensor.
  auto tensors_tensor_list =
    PyNativeAlgo::Common::ConvertStubNodeToValueTuple(tensors, true, op_run_info->requires_grad);

  // Do mixed precision and implicit cast
  static const std::vector<std::vector<size_t>> same_type_table{};
  auto [cast_tensors_tensor_list] =
    PyNativeAlgo::PyBoost::SetPyBoostCastForInputs<0>(op_run_info, "CustomExt", same_type_table, tensors_tensor_list);

  // Create op
  auto op = CREATE_PYBOOST_OP(CustomExt, op_run_info->device_target);
  op->set_primitive(prim);
  // Run op
  (void)op->Call(cast_tensors_tensor_list);

  // Create output value
  auto real_out = AutoGradUtil::MakeMultiOutput(op_run_info->requires_grad, op);
  // Do auto grad
  if (op_run_info->requires_grad) {
    auto op_grad_info = std::make_shared<OpGradInfo>(op_type, op->primitive(),
                                                     std::vector<ValuePtr>({cast_tensors_tensor_list}), real_out);
    AutoGradUtil::SetInferMultiOutputToGrad(op_grad_info, op);
    PyNativeAlgo::PyBoost::DoGrad(op, op_grad_info, op_run_info->async_status);
  } else if (op_type == OperatorType::kInplaceOp) {
    PyNativeAlgo::PyBoost::BumpVersionAsync(op->outputs()[0]->version());
  }
  MS_LOG(DEBUG) << "Run Pyboost_CustomExt end";
  return py::reinterpret_steal<py::object>(tensor::Wrap(real_out));
#else
  return PyNativeAlgo::PyBoost::RunPyFunction(prim, args);
#endif
}

py::object Pyboost_CustomExt(const py::args &args) {
  if (args.size() != kIndex2) {
    MS_LOG(EXCEPTION) << "Two args are needed by RunOp"
                      << ", but got " << args.size();
  }
  const auto &prim = PyNativeAlgo::PyBoost::ConvertPrimitive(args[0]);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, prim->name(),
                                     false, true);
  return PyboostCustomExtBase(prim, args[1]);
}

class CustomExtPrimAdapter : public PrimitiveFunctionAdapter {
 public:
  CustomExtPrimAdapter() : PrimitiveFunctionAdapter() {}
  ~CustomExtPrimAdapter() = default;
  std::string name() override { return "CustomExt"; }
  py::object Call(const py::list &args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, "CustomExt",
                                       false, true);
    return PyboostCustomExtBase(prim::kPrimCustomExt, args);
  }
};

void RegisterCustomizeFunction(py::module *m) {
  (void)py::class_<CustomExtPrimAdapter, PrimitiveFunctionAdapter, std::shared_ptr<CustomExtPrimAdapter>>(
    *m, "CustomExtPrim_")
    .def(py::init<>())
    .def("__call__", &CustomExtPrimAdapter::Call, "Call CustomExt op.");
  m->def("pyboost_custom_ext", &mindspore::pynative::Pyboost_CustomExt, "Encrypt the data.");
}
}  // namespace mindspore::pynative
