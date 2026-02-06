/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_

#include <cstdint>
#include <mutex>
#include <vector>
#include <string>

namespace mindspore::dataset {
/// \brief Register the custom signal handlers.
extern void RegisterHandlers();

/// \brief Register signal handlers for main process.
extern void RegisterMainHandlers();

/// \brief Register signal handlers for worker process.
extern void RegisterWorkerHandlers();

/// \brief Register workers to be monitored by the watch dog.
extern void RegisterWorkerPIDs(int64_t id, const std::vector<int> &pids);

/// \brief Deregister workers to be monitored by the watch dog.
extern void DeregisterWorkerPIDs(int64_t id);

extern std::mutex shm_msg_id_mtx_;

extern void RegisterShmIDAndMsgID(std::string pid, int32_t shm_id, int32_t msg_id);

extern void ReleaseShmAndMsg();

/// \brief Called in Python Layer of main process
extern void ReleaseShmAndMsgByWorkerPIDs(const std::vector<int> &pids);

/// \brief Manually release the shm_msg_id_mtx_
/// When start multiple dataset iterators in the same time, it is necessary to acquire the lock before launching
/// the map/batch subprocess in execution.cc, and then release this lock in the python _worker_loop(...) function.
extern void UnlockShmIDAndMsgIDMutex();

/// \brief Check if the worker process has exited.
extern std::string CheckIfWorkerExit();
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_
