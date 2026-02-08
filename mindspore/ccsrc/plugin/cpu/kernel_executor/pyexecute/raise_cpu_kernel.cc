/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "plugin/cpu/kernel_executor/pyexecute/raise_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "ir/anf.h"
#include "primitive/framework_op_name.h"
#include "utils/log_adapter.h"
#include "plugin/cpu/kernel_executor/pyexecute/joinedstr_cpu_kernel.h"

namespace mindspore {
namespace kernel {
bool RaiseCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  constexpr size_t min_input_size = 2;
  MS_EXCEPTION_IF_CHECK_FAIL(
    inputs.size() >= min_input_size,
    "Input size should be at least " + std::to_string(min_input_size) + " but got " + std::to_string(inputs.size()));
  MS_EXCEPTION_IF_NULL(primitive_);
  return true;
}

bool RaiseCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                               const std::vector<KernelTensor *> &outputs) {
  constexpr size_t exception_one_input_size = 3;
  constexpr size_t exception_type_abs_index = 0;
  constexpr size_t exception_msg_abs_first_index = 1;
  const size_t exception_msg_abs_last_index = inputs.size() - 2;
  auto exception_type_abs = inputs[exception_type_abs_index];
  MS_EXCEPTION_IF_NULL(exception_type_abs);
  const auto &exception_type_str = GetValue<std::string>(exception_type_abs->BuildValue());
  py::gil_scoped_acquire gil_acquire;
  std::string exception_msg;
  for (size_t index = exception_msg_abs_first_index; index <= exception_msg_abs_last_index; ++index) {
    const auto &input = inputs[index];
    MS_EXCEPTION_IF_NULL(input);
    AbstractBase *object_input = input;
    auto type = input->GetType();
    MS_EXCEPTION_IF_NULL(type);
    bool cur_is_str = object_input->has_user_data("str_exception_result") || type->ToString() == "String";
    const auto &cur_exception_msg = object_input->has_user_data("str_exception_result")
                                      ? *object_input->user_data<string>("str_exception_result")
                                      : ConvertAbsToStr(input);
    // if only one input
    if (inputs.size() == exception_one_input_size) {
      exception_msg = cur_exception_msg;
      break;
    }
    if (index == exception_msg_abs_first_index) {
      exception_msg = cur_is_str ? "('" + cur_exception_msg + "', " : "(" + cur_exception_msg + ", ";
      continue;
    }
    exception_msg = cur_is_str ? exception_msg + "'" + cur_exception_msg + "'" : exception_msg + cur_exception_msg;
    // if is last inputs index
    exception_msg = index == exception_msg_abs_last_index ? exception_msg + ")" : exception_msg + ", ";
  }
  auto iter = exception_types_map.find(exception_type_str);
  if (iter == exception_types_map.end()) {
    MS_LOG(ERROR) << "Found unexpected exception type " << exception_type_str;
    return false;
  }
  auto exception_type = iter->second;
  auto &handler = LogWriter::GetExceptionHandler();
  MS_EXCEPTION_IF_NULL(handler);
  handler(exception_type, exception_msg);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, raise, RaiseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
