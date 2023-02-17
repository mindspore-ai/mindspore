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
static std::map<ValuePtr, py::object> py_execute_output_map;

bool HasPyExecuteOutput() { return !py_execute_output_queue.empty(); }

bool HasPyExecuteOutput(const ValuePtr &key) {
  auto iter = py_execute_output_map.find(key);
  return iter != py_execute_output_map.end();
}

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

void PushPyExecuteOutput(const ValuePtr &key, const py::object &output) {
  auto iter = py_execute_output_map.find(key);
  if (iter != py_execute_output_map.end()) {
    MS_LOG(DEBUG) << "the key:" << key << "of PyExecute result map is already exist. " << key << ":" << iter->second;
    return;
  }
  MS_LOG(DEBUG) << "insert " << key << ":" << output << " in PyExecute map.";
  py_execute_output_map[key] = output;
}

py::object PopPyExecuteOutput(const ValuePtr &key) {
  auto iter = py_execute_output_map.find(key);
  if (iter == py_execute_output_map.end()) {
    MS_LOG(EXCEPTION) << "Cannot find the key(" << key << ") in result map";
  }
  MS_LOG(DEBUG) << "Get " << key << ":" << iter->second << " in PyExecute map.";
  auto output = iter->second;
  MS_LOG(DEBUG) << "output: " << output;
  py_execute_output_map.erase(iter);
  return output;
}
}  // namespace mindspore
