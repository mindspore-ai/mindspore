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
#include "pipeline/pynative/op_function/converter.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "mindspore/core/ops/auto_generate/gen_ops_def.h"
${include_op_header}

namespace mindspore::pynative {
${function_body}

${register_function_body}
}// namespace mindspore::pynative
