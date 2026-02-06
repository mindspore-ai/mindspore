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

#include "include/pynative/utils/pyboost/grad_functions/pyboost_grad_functions.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/pynative/utils/pyboost/grad_functions/value_converter.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/utils/python_adapter.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/utils/pynative/abstract_converter.h"
${composite_headers}

namespace mindspore::runtime {
namespace {
void ChangeInputToAttr(const PrimitivePtr &prim, const ValuePtr &input_names, const mindspore::HashSet<size_t> &input_to_attr, pynative::BaseOpRunInfo *op_run_info) {
  MS_EXCEPTION_IF_NULL(input_names);
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names);
  ValuePtrList new_inputs{};
  std::vector<InputType> new_input_mask{};
  size_t convert_size = 0;
  const auto &inputs = op_run_info->expanded_input_values;
  const auto &input_types = op_run_info->input_types;
  size_t input_size = inputs.size();
  for (size_t i = 0; i < input_size; ++i) {
    const auto &value = inputs[i];
    if (input_types[i] == InputType::kConstant && input_to_attr.find(i) != input_to_attr.end()) {
      MS_LOG(DEBUG) << "start erase input[" << i << "] of op[" + prim->name() + "]";
      if (i >= input_names_vec.size()) {
        MS_LOG(EXCEPTION) << "Index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
      }
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        if (tensor->unsafe_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
          return;
        }
      }
      ++convert_size;
      prim->set_attr(input_names_vec[i], value);
    } else {
      (void)new_inputs.emplace_back(inputs[i]);
      (void)new_input_mask.emplace_back(input_types[i]);
    }
  }
  if (convert_size > 0) {
    (void)prim->AddAttr(kAttrConvertAttrNode, MakeValue(convert_size));
  }
  op_run_info->expanded_input_values = std::move(new_inputs);
  op_run_info->input_types = std::move(new_input_mask);
}

void ConvertConstInputToAttr(const PrimitivePtr &prim, pynative::BaseOpRunInfo *op_run_info) {
  mindspore::HashSet<size_t> input_to_attr = {};
  kernel::pyboost::PyBoostUtils::GetConstInputToAttr(prim, prim->name(), op_run_info->device_target, false, &input_to_attr);
  if (input_to_attr.empty()) {
    return;
  }
  const auto &input_names = prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  ChangeInputToAttr(prim, input_names, input_to_attr, op_run_info);
}

session::BackendOpRunInfoPtr GetBackendOpRunInfo(OpRunnerInfo *op_runner_info) {
  MS_EXCEPTION_IF_NULL(op_runner_info);
  MS_EXCEPTION_IF_NULL(op_runner_info->prim);
  pynative::BaseOpRunInfo base_op_run_info;
  base_op_run_info.op_name = op_runner_info->prim->name();
  base_op_run_info.device_target = op_runner_info->device_target;
  base_op_run_info.expanded_input_values = op_runner_info->inputs;
  base_op_run_info.input_types = op_runner_info->inputs_mask;
  // Do infer and refresh output abstract
  op_runner_info->output_abs = kernel::pyboost::PyBoostUtils::InferByOpDef(op_runner_info->prim, op_runner_info->inputs_abs);
  base_op_run_info.abstract = op_runner_info->output_abs ;
  return std::make_shared<session::BackendOpRunInfo>(base_op_run_info, op_runner_info->prim, false, false);
}

DoGradFunc do_grad_func{nullptr};

pynative::AbstractConverter kAbstractConverter;
} // namespace

void RegisterDoGradFunc(const DoGradFunc &grad_func) {
  do_grad_func = grad_func;
}

const DoGradFunc& GetDoGradFunc() {
  if (do_grad_func == nullptr) {
    MS_LOG(EXCEPTION) << "Do grad func not register!";
  }
  return do_grad_func;
}

PyBoostOpExecute& PyBoostOpExecute::GetInstance() {
  static PyBoostOpExecute instance;
  return instance;
}

bool PyBoostOpExecute::IsPyBoostOpRegistered(const std::string &op_name) {
  return grad_op_func_map_.find(op_name) != grad_op_func_map_.end();
}

void PyBoostOpExecute::Execute(OpRunnerInfo *op_runner_info, VectorRef *op_outputs) {
 runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kExecute,
                                    op_runner_info->prim->name(), false);
 #ifndef ENABLE_TEST
  GilReleaseWithCheck release_gil;
  MS_EXCEPTION_IF_NULL(op_runner_info);
  const auto it = grad_op_func_map_.find(op_runner_info->prim->name());
  // Run op by pyboost
  if (it != grad_op_func_map_.end() && 
      (kernel::pyboost::PyBoostUtils::IsKernelModRegistered(op_runner_info->device_target, op_runner_info->prim->name())
       || kernel::pyboost::PyBoostUtils::IsPyBoostCustomRegistered(op_runner_info->device_target, op_runner_info->prim->name()))) {
    const auto &func = FuncCast<Func>(it->second);
    MS_EXCEPTION_IF_NULL(func);
    func(op_runner_info, op_outputs);
    return;
  }
  // Run op by single op graph
  RunOpDeprecated(op_runner_info, op_outputs);
#else
  RunOpInVm(op_runner_info, op_outputs);
#endif
}

void PyBoostOpExecute::RunPyBoostCall(OpRunnerInfo *op_runner_info, VectorRef *op_outputs) {
  MS_EXCEPTION_IF_NULL(op_runner_info);
  const auto &func = FuncCast<Func>(grad_op_func_map_.at(op_runner_info->prim->name()));
  MS_EXCEPTION_IF_NULL(func);
  func(op_runner_info, op_outputs);
}

void PyBoostOpExecute::RunOpDeprecated(OpRunnerInfo *op_runner_info, VectorRef *op_outputs) {
  // Do infer and refresh output abstract
  op_runner_info->output_abs = kernel::pyboost::PyBoostUtils::InferByOpDef(op_runner_info->prim, op_runner_info->inputs_abs);
  // For call runop
  const auto &backend_op_run_info = GetBackendOpRunInfo(op_runner_info);
  ConvertConstInputToAttr(op_runner_info->prim, &backend_op_run_info->base_op_run_info);
  backend_op_run_info->base_op_run_info.abstract = op_runner_info->output_abs ;
  // Call single op graph run
  backend_op_run_info->base_op_run_info.use_dynamic_shape_process = true;
  backend_op_run_info->op_prim = std::make_shared<Primitive>(*op_runner_info->prim);
  AnfAlgo::SetDynamicAttrToPrim(backend_op_run_info->op_prim);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  op_backend_.Run(backend_op_run_info, backend_op_run_info->base_op_run_info.device_target, device_id, op_outputs);
  GetDoGradFunc()(op_runner_info, op_outputs);
}

void PyBoostOpExecute::RunOpInVm(OpRunnerInfo *op_runner_info, VectorRef *op_outputs) {
  VectorRef args;
  std::transform(op_runner_info->inputs.begin(), op_runner_info->inputs.end(), std::back_inserter(args),
                 [](const auto &value) { return value; });
  py::gil_scoped_acquire gil;
  auto result = kernel::pyboost::PyBoostUtils::RunOperation(op_runner_info->prim, args);
  if (utils::isa<PyObjectRef>(result)) {
    PyObjectRef py_ref = utils::cast<PyObjectRef>(result);
    py::object value = py_ref.object_;
    auto result_v = python_adapter::PyAdapterCallback::PyDataToValue(value);
    if (!result_v->isa<ValueSequence>()) {
      (void)op_outputs->emplace_back(result_v);
    } else {
      auto seq = result_v->cast<ValueSequencePtr>();
          std::transform(seq->value().begin(), seq->value().end(), std::back_inserter(*op_outputs),
                         [](const auto &value) { return value; });
    }
    op_runner_info->output_abs = result_v->ToAbstract()->Broaden();
    GetDoGradFunc()(op_runner_info, op_outputs);
    return;
  }

  MS_LOG(EXCEPTION) << "prim: " << op_runner_info->prim->name() << "did not has vm op!";
}

${function_body}

${register_function_body}

} // namespace mindspore::pynative
