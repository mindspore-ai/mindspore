/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CONSTANTS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CONSTANTS_H_

#include <set>
#include <map>
#include <chrono>
#include <string>

namespace mindspore {
namespace distributed {
constexpr char kEnvServerNum[] = "MS_SERVER_NUM";
constexpr char kEnvWorkerNum[] = "MS_WORKER_NUM";
constexpr char kEnvSchedulerHost[] = "MS_SCHED_HOST";
constexpr char kEnvSchedulerPort[] = "MS_SCHED_PORT";

constexpr char kEnvRole[] = "MS_ROLE";
constexpr char kEnvRoleOfServer[] = "MS_SERVER";
constexpr char kEnvRoleOfWorker[] = "MS_WORKER";
constexpr char kEnvRoleOfScheduler[] = "MS_SCHED";
const std::set<std::string> kValidRoleName = {kEnvRoleOfServer, kEnvRoleOfWorker, kEnvRoleOfScheduler};

// The distributed execution mode enum.
enum class DistributedExecutionMode { kPSMode = 0, kInvalidMode };

// The operator's label in distributed execution.
constexpr char kOpLabelRankId[] = "rank_id";
constexpr char kOpLabelRole[] = "ms_role";

constexpr char kLocalHost[] = "127.0.0.1";
constexpr int MAX_HOSTNAME_LEN = 1024;
const uint16_t kDefaultSchedPort = 6667;
const uint16_t kMaxPort = 65535;
constexpr uint32_t kDefaultFinishTimeout = 30;

// This macro the current timestamp in milliseconds.
#define CURRENT_TIMESTAMP_MILLI \
  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CONSTANTS_H_
