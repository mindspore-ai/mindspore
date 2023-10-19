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
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pybind_api/gil_scoped_long_running.h"
using namespace mindspore::kernel::pyboost;
namespace mindspore::ops {
extern OpDef gBaddbmm;
}
namespace mindspore::pynative {
constexpr const int64_t kIndex0 = 0;
constexpr const int64_t kIndex1 = 1;
constexpr const int64_t kIndex2 = 2;
constexpr const int64_t kIndex3 = 3;
constexpr const int64_t kIndex4 = 4;
constexpr const int64_t kIndex5 = 5;
constexpr const int64_t kIndex6 = 6;

py::object Pyboost_Baddbmm(const py::args &args) {
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
  // forward_executor->OpRunInfoUsePrimC(op_run_info);
  // PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args[1],
  //                                               true);
  op_run_info->base_op_run_info.device_target =
    forward_executor->GetCurrentDeviceTarget(op_run_info->op_grad_info->op_prim);
  op_run_info->cell_obj_id = forward_executor->GetCurrentCellObjId();
  // SetCallbackForInputTensor(op_run_info);
  // pynative_executor->StoreAsyncStatus(op_run_info);
  //    const auto &op_name = op_run_info->base_op_run_info.op_name;
  //    // 2. if disable PyTraceAsync, return after infer(half-asynchronous) or run(synchronous mode)
  //    if (!forward_executor->EnablePipeline(op_name)) {
  //      // Wait for async task finish
  //      forward_executor->WaitForwardTask();
  //      PyNativeAlgo::Common::StubNodeToValue(op_run_info);
  //      // RunOp sync
  //      PyNativeExecutorTry(forward_executor->RunOpS, op_run_info);
  //      return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->real_out);
  //    }
  // auto output = std::make_shared<TensorNode>();
  // op_run_info->stub_output = output;
  static Parser parser(&ops::gBaddbmm);
  py::list input_args = args[1];
  parser.Parse(input_args);
  auto input = parser.ToTensor(kIndex0);
  auto batch1 = parser.ToTensor(kIndex1);
  auto batch2 = parser.ToTensor(kIndex2);
  auto beta = parser.ToScalar(kIndex3);
  auto alpha = parser.ToScalar(kIndex4);
  //    auto dispatch_baddbmm = [op_run_info, input, batch1,
  //      batch2, beta, alpha] () {
  //      const TensorPtr &input_tensor = PyNativeAlgo::Common::StubNodeToTensor(input);
  //      const TensorPtr &batch1_tensor = PyNativeAlgo::Common::StubNodeToTensor(batch1);
  //      const TensorPtr &batch2_tensor = PyNativeAlgo::Common::StubNodeToTensor(batch2);
  //      auto op = OpFactory<Baddbmm>::Get().Create(op_run_info->base_op_run_info.device_target);
  //      op->Call(std::move(op_run_info), input_tensor, batch1_tensor, batch2_tensor, beta, alpha);
  //    };
  //    GilReleaseWithCheck release_gil;
  //    forward_executor()->DispatchAnyFrontendTask(dispatch_baddbmm);

  auto op = OpFactory<Baddbmm>::Get().Create(op_run_info->base_op_run_info.device_target);
  auto output = op->Call(input, batch1, batch2, beta, alpha);
  return parser.Wrap(output);
}
void RegisterPyBoostFunction(py::module *m) {
  m->def("pyboost_baddbmm", &mindspore::pynative::Pyboost_Baddbmm, "Encrypt the data.");
}
}  // namespace mindspore::pynative
