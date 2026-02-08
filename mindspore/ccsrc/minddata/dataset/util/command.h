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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_COMMAND_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_COMMAND_H_

#include <condition_variable>
#include <mutex>
#include <string>

namespace mindspore {
namespace dataset {
#if !defined(_WIN32) && !defined(_WIN64)
void MonitorLoop(std::mutex *monitor_mtx, std::condition_variable *monitor_cv, bool *monitor_exit_flag,
                 const int32_t &worker_pid, const std::string &op_type);

void PrintPythonStack(const int32_t &worker_id, const std::string &op_type);
#endif
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_COMMAND_H_
