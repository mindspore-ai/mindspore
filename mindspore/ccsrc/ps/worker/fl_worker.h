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

#ifndef MINDSPORE_CCSRC_PS_WORKER_FL_WORKER_H_
#define MINDSPORE_CCSRC_PS_WORKER_FL_WORKER_H_

#include <memory>
#include <string>
#include <vector>
#include "proto/comm.pb.h"
#include "schema/fl_job_generated.h"
#include "ps/ps_context.h"
#include "ps/core/worker_node.h"
#include "ps/core/cluster_metadata.h"
#include "ps/core/communicator/tcp_communicator.h"

namespace mindspore {
namespace ps {
using FBBuilder = flatbuffers::FlatBufferBuilder;

// The worker standalone training step number(Without communicating with server).
// This is used in hybrid training mode.
constexpr uint32_t kWorkerTrainStepNum = 20 * 65;
constexpr uint32_t kTrainBeginStepNum = 1;
constexpr uint32_t kTrainEndStepNum = 0;

// The worker has to sleep for a while before the networking is completed.
constexpr uint32_t kWorkerSleepTimeForNetworking = 1000;

namespace worker {
// This class is used for hybrid training mode for now. In later version, parameter server mode will also use this class
// as worker.
class FLWorker {
 public:
  static FLWorker &GetInstance() {
    static FLWorker instance;
    return instance;
  }
  void Run();
  bool SendToServer(uint32_t server_rank, void *data, size_t size, core::TcpUserCommand command,
                    std::shared_ptr<std::vector<unsigned char>> *output = nullptr);

 private:
  FLWorker() = default;
  ~FLWorker() = default;
  FLWorker(const FLWorker &) = delete;
  FLWorker &operator=(const FLWorker &) = delete;

  uint32_t server_num_;
  uint32_t worker_num_;
  std::string scheduler_ip_;
  uint16_t scheduler_port_;
  std::shared_ptr<core::WorkerNode> worker_node_;
};
}  // namespace worker
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_WORKER_FL_WORKER_H_
