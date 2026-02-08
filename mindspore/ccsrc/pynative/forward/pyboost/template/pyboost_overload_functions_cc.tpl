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
#include "ir/named.h"
#include "pynative/forward/pyboost/auto_generate/tensor_func_utils.h"
#include "include/pynative/forward/pyboost/converter.h"
#include "include/pynative/forward/pyboost/fallback.h"
#include "include/pynative/forward/pyboost/arg_handler_py.h"
#include "include/frontend/jit/trace/trace_recorder_interface.h"
#include "include/pynative/forward/pyboost/auto_generate/pyboost_core.h"
#include "pybind_api/pynative/pynative_api.h"
#include "mindspore/ops/ops_utils/op_utils.h"
${ops_inc}


namespace mindspore::pynative {
${mint_func_classes_def}

void RegisterFunctional(py::module *m) {
  ${pybind_register_code}
}
}  // namespace mindspore::pynative
