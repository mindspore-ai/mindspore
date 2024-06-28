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

#include "pipeline/pynative/grad/function/auto_generate/pyboost_native_grad_functions.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_function/value_converter.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "backend/graph_compiler/vmimpl.h"
#include "include/common/utils/python_adapter.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/expander/core/node.h"
#include "pipeline/pynative/pynative_utils.h"
#include "runtime/pynative/op_function/pyboost_grad_functions.h"
${include_op_header}

namespace mindspore {
namespace pynative {
std::string NativeFunc::device_target_ = "";

NodePtr NativeFunc::RunOpInVm(const PrimitivePtr &prim, const NodePtrList &inputs) {
  VectorRef args;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(args),
                 [](const auto &node) { return node->Value(); });
  py::gil_scoped_acquire gil;
  auto result = compile::RunOperation(prim, args);
  if (utils::isa<PyObjectRef>(result)) {
    PyObjectRef py_ref = utils::cast<PyObjectRef>(result);
    py::object value = py_ref.object_;
    auto result_v = python_adapter::PyAdapterCallback::PyDataToValue(value);
    auto output_abs = result_v->ToAbstract()->Broaden();
    auto output_node = std::make_shared<expander::FuncNode>(result_v, output_abs, InputType::kOpOutput, inputs[0]->emitter());
    return output_node;
  }
  MS_LOG(EXCEPTION) << "prim: " << prim->name() << "did not has vm op!";
}

NodePtr NativeFunc::RunOpDeprecated(const PrimitivePtr &prim, const NodePtrList &inputs) {
  std::vector<ValuePtr> input_values;
  std::vector<abstract::AbstractBasePtr> abs_list;
  std::vector<InputType> input_masks;
  input_values.reserve(inputs.size());
  abs_list.reserve(inputs.size());
  input_masks.reserve(inputs.size());
  for (const auto &input : inputs) {
    (void)input_values.emplace_back(input->Value());
    (void)abs_list.emplace_back(input->abstract());
    (void)input_masks.emplace_back(input->input_type());
  }
  runtime::OpRunnerInfo op_runner_info{prim, device_target_, input_values, abs_list, input_masks, nullptr};
  VectorRef outputs;
  runtime::PyBoostOpExecute::GetInstance().RunOpDeprecated(&op_runner_info, &outputs);
  auto real_outputs = common::AnfAlgo::TransformVectorRefToMultiValue(outputs);
  if (op_runner_info.output_value_simple_info != nullptr) {
    // Get output abstract
    op_runner_info.output_abs = TransformValueSimpleInfoToAbstract(*op_runner_info.output_value_simple_info);
  }
  ValuePtr value_result;
  MS_EXCEPTION_IF_NULL(op_runner_info.output_abs);
  if (real_outputs.size() == kSizeOne && !op_runner_info.output_abs->isa<abstract::AbstractSequence>()) {
    value_result = real_outputs[kIndex0];
  } else {
    value_result = std::make_shared<ValueTuple>(std::move(real_outputs));
  }
  // Set abstract to tensor cache
  if (op_runner_info.output_value_simple_info != nullptr) {
    PyNativeAlgo::AutoGrad::CacheOutputAbstract(value_result, op_runner_info.output_abs);
  }
  auto result = std::make_shared<expander::FuncNode>(value_result, op_runner_info.output_abs, InputType::kOpOutput, inputs[0]->emitter());
  return result;
}

${function_body}
}
}

