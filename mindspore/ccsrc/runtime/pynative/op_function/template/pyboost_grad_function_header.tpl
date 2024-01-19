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

#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_function/value_converter.h"
#include "runtime/pynative/op_function/pyboost_grad_functions.h"
${include_op_header}

namespace mindspore::runtime {
PyBoostOpExecute& PyBoostOpExecute::GetInstance() {
  static PyBoostOpExecute instance;
  return instance;
}

bool PyBoostOpExecute::IsPyBoostOpRegistered(const std::string &op_name) {
  return grad_op_func_map_.find(op_name) != grad_op_func_map_.end();
}

void PyBoostOpExecute::RunPyBoostCall(const PrimitivePtr &prim, const std::string &device_target,
                                      const vector<ValuePtr> &inputs, VectorRef *op_outputs) {
  const auto &func = FuncCast<Func>(grad_op_func_map_.at(prim->name()));
  MS_EXCEPTION_IF_NULL(func);
  func(prim, device_target, inputs, op_outputs);
}

${function_body}

${register_function_body}

} // namespace mindspore::pynative
