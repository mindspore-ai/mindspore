
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

#include <string>

#include "mindspore/ccsrc/utils/base_ref_py.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/python_adapter.h"
#include "ir/anf.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore::ops {
ValuePtr InferValuePyPythonDef(const std::string &op_name, const AbstractBasePtrList &input_args) {
  const std::string python_manually_defined_module = "mindspore.ops.operations.manually_defined.ops_def";
  const std::string python_infer_value_prefix = "infer_value_for_";
  py::function func = python_adapter::GetPyFn(python_manually_defined_module, python_infer_value_prefix + op_name);
  if (py::isinstance<py::none>(func)) {
    MS_LOG(EXCEPTION) << "No python-defined infer value function for " << op_name;
  }

  py::list args;
  for (const auto &abs : input_args) {
    if (abs->isa<abstract::AbstractMonad>()) {
      continue;
    }
    auto value = abs->GetValue();
    args.append(ValueToPyData(value));
  }

  py::object res = func(*args);
  if (py::isinstance<py::none>(res)) {
    MS_LOG(DEBUG) << "Func " << python_infer_value_prefix << op_name << " Call failed!";
    return nullptr;
  }
  ValuePtr res_val = nullptr;
  bool succ = parse::ConvertData(res, &res_val);
  if (!succ) {
    MS_LOG(EXCEPTION) << "Convert " << py::str(res) << " failed for " << op_name;
  }

  return res_val;
}

INFER_VALUE_IMPL_REGISTER(python_impl, InferValuePyPythonDef);
}  // namespace mindspore::ops
