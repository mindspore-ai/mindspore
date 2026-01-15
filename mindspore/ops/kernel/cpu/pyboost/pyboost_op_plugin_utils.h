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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_PYBOOST_OP_PLUGIN_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_PYBOOST_OP_PLUGIN_UTILS_H_
#include <vector>
#include <memory>
#include <type_traits>
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/cpu/op_plugin/op_plugin_utils.h"

// Helper to check if a type is optional
template <typename T>
struct is_std_optional : std::false_type {};

template <typename U>
struct is_std_optional<std::optional<U>> : std::true_type {};

template <typename T>
constexpr bool is_std_optional_v = is_std_optional<std::decay_t<T>>::value;

// Helper to check if a type is int or vector<int>
template <typename T>
struct is_int_or_vector_int : std::false_type {};

template <>
struct is_int_or_vector_int<int64_t> : std::true_type {};

template <>
struct is_int_or_vector_int<std::vector<int64_t>> : std::true_type {};

template <typename T>
constexpr bool is_int_or_vector_int_v = is_int_or_vector_int<std::decay_t<T>>::value;

template <typename... Args>
constexpr bool has_int_or_vector_int_v = (is_int_or_vector_int_v<Args> || ...);

namespace mindspore::kernel::pyboost {
template <typename T>
constexpr bool is_tensor_ptr_v = std::is_same_v<std::decay_t<T>, tensor::TensorPtr>;
template <typename T>
constexpr bool is_value_tuple_ptr_v = std::is_same_v<std::decay_t<T>, ValueTuplePtr>;

// Overload for when any argument is int or vector<int> - returns empty vector
// Reason to have this overload:
// Some pyboost functions pass int or vector<int> as arguments, which are not compatible with the InferOutput function.
// These functions are mainly view functions, which do not really have an op plugin kernel.
template <std::size_t... InplaceIndices, typename... Args>
std::enable_if_t<has_int_or_vector_int_v<Args...>, std::vector<tensor::TensorPtr>> PyboostLaunchOpPluginKernel(
  std::shared_ptr<OpRunner> op, Args &&...args) {
  return {};
}

inline void UpdateOutputShapesFromPlugin(op_plugin::OpPluginKernelParam &param, std::shared_ptr<OpRunner> op) {
  auto *output_info = param.kernel_info.GetOutputShapeInfo();
  if (!output_info) {
    return;
  }
  const auto &outputs = op->outputs();
  auto simple_infer_ptr = op->output_value_simple_info();
  MS_EXCEPTION_IF_NULL(simple_infer_ptr);

  if (outputs.size() != output_info->shape_calculated.size() ||
      outputs.size() != simple_infer_ptr->shape_vector_.size()) {
    MS_LOG(EXCEPTION) << "For '" << op->primitive()->name() << "', output size mismatch: outputs=" << outputs.size()
                      << ", shape_calculated=" << output_info->shape_calculated.size()
                      << ", shape_vector=" << simple_infer_ptr->shape_vector_.size();
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    if (output_info->shape_calculated[i]) {
      op->UpdateOutputShape(outputs[i], output_info->output_shapes[i]);
      simple_infer_ptr->shape_vector_[i] = output_info->output_shapes[i];
    }
  }
}

// The InplaceIndex indicates the input tensor the output corresponds to in a inplace operation.
template <std::size_t... InplaceIndices, typename... Args>
std::enable_if_t<!has_int_or_vector_int_v<Args...>, std::vector<tensor::TensorPtr>> PyboostLaunchOpPluginKernel(
  std::shared_ptr<OpRunner> op, Args &&...args) {
  MS_EXCEPTION_IF_NULL(op->primitive());
  const auto &op_name = op->primitive()->name();
  MS_LOG(DEBUG) << op_name << " calls op plugin kernel.";

  constexpr bool is_inplace = sizeof...(InplaceIndices) > 0;

  if (!is_inplace) {
    op->InferOutput(args...);
  }

  // Set correct outputs for inplace operations
  if constexpr (is_inplace) {
    std::vector<tensor::TensorPtr> effective_outputs;
    auto input_tensors = std::make_tuple(args...);
    effective_outputs = {std::get<InplaceIndices>(input_tensors)...};
    op->set_outputs(effective_outputs);
  }

  const auto device_context = op->device_context();
  MS_EXCEPTION_IF_NULL(device_context);

  // Find tensor arguments for PrepareOpInputs
  auto process_tensor_args = [&](auto &&arg) {
    if constexpr (is_std_optional_v<decltype(arg)>) {
      if constexpr (is_tensor_ptr_v<decltype(arg.value())>) {
        PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), arg);
      }
    } else if constexpr (is_tensor_ptr_v<decltype(arg)> || is_value_tuple_ptr_v<decltype(arg)>) {
      PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), arg);
    }
  };
  (process_tensor_args(args), ...);

  const auto &outputs = op->outputs();
  if constexpr (!is_inplace) {
    PyBoostUtils::PrepareOpOutputs(device_context, 0, outputs);
  }

  op->ProfileTrackerTask();

  // Async
  // pass 'outputs' by value because op->outputs() is occasionally broken in the dispatch task
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, &op_name, outputs, args...]() {
    auto device_context = op->device_context();
    constexpr bool is_inplace_lambda = sizeof...(InplaceIndices) > 0;

    // Process tensor arguments for MallocOpInputs
    auto malloc_tensor_args = [&](auto &&arg) {
      if constexpr (is_std_optional_v<decltype(arg)>) {
        if constexpr (is_tensor_ptr_v<decltype(arg.value())>) {
          PyBoostUtils::MallocOpInputs(device_context, arg);
        }
      } else if constexpr (is_tensor_ptr_v<decltype(arg)> || is_value_tuple_ptr_v<decltype(arg)>) {
        PyBoostUtils::MallocOpInputs(device_context, arg);
      }
    };
    (malloc_tensor_args(args), ...);

    if constexpr (!is_inplace_lambda) {
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
    }

    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), args...);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);
    std::vector<kernel::KernelTensor *> workspace_tensors;
    auto op_plugin_param =
      op_plugin::CreateOpPluginParam(input_address_info.first, output_address_info.first, workspace_tensors);
    auto ret = op_plugin::LaunchOpPluginKernel(op_name, &op_plugin_param);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Launch op plugin kernel failed, op name: " << op_name << ", return code: " << ret;
    }
    UpdateOutputShapesFromPlugin(op_plugin_param, op);
  }));
  op->ProfileTrackerInput(args...);
  op->ProfileTrackerOutput(outputs);
  MS_LOG(DEBUG) << op_name << " op plugin kernel call end";
  op->CreateOutputSimpleInfo();
  return outputs;
}
}  // namespace mindspore::kernel::pyboost
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_PYBOOST_OP_PLUGIN_UTILS_H_
