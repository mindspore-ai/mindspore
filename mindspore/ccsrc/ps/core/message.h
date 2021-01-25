/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_CORE_MESSAGE_H_
#define MINDSPORE_CCSRC_PS_CORE_MESSAGE_H_

#include <string>
#include <memory>

namespace mindspore {
namespace ps {
namespace core {
enum class Protos : uint32_t { RAW = 0, PROTOBUF = 1, FLATBUFFERS = 2 };

enum class Command {
  TERMINATE = 0,
  REGISTER = 1,
  HEARTBEAT = 2,
  SEND_DATA = 3,
  FETCH_SERVER = 4,
  FINISH = 5,
  COLLECTIVE_SEND_DATA = 6
};

enum class Role { SERVER = 0, WORKER = 1, SCHEDULER = 2 };

struct MessageHeader {
  Protos message_proto_ = Protos::RAW;
  uint32_t message_meta_length_ = 0;
  uint64_t message_length_ = 0;
};

struct CommandMeta {
  // the command of this message,for example: register,heartbeat,data
  Command cmd;
  // the request id of this message
  uint64_t request_id;
  // the role of the current node: worker,server,scheduler
  Role role;
  // the current Node rank id,the worker node range is:[0,numOfWorker-1], the server node range is:[0, numOfServer-1]
  int32_t rank_id = 4;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_MESSAGE_H_
