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

#include "include/common/fallback.h"

#include <queue>

#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"

namespace mindspore {
static std::queue<py::object> py_execute_output_queue;

bool HasPyExecuteOutput() { return !py_execute_output_queue.empty(); }

py::object PopPyExecuteOutput() {
  auto output = py_execute_output_queue.front();
  MS_LOG(DEBUG) << "output: " << output;
  py_execute_output_queue.pop();
  return output;
}

void PushPyExecuteOutput(const py::object &output) {
  MS_LOG(DEBUG) << "output: " << output;
  py_execute_output_queue.push(output);
}
}  // namespace mindspore
