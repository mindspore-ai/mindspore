/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PERF_MSG_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PERF_MSG_H_

#include <cstdint>
#include <limits>
#include <string>
#include "proto/cache_perf.pb.h"
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// All our messages are very small. So we will use the stack version without the need
// to allocate memory.
struct CacheSmallMsg {
  int64_t mtype;
  union {
    char mtext[1];
    struct {
      int32_t type;  // the first 4 bytes is the RequestType
      int32_t proto_sz;
      char proto_buffer[kSharedMessageSize];
    } msg;
  } body;
};
/// A message queue structure between the parent and the child process
class CachePerfMsg {
 public:
  enum MessageType : int16_t {
    kInterrupt = 0,
    kEpochResult = 1,
    kEpochStart = 2,
    kEpochEnd = 3,
    kError = 4,
    // Add new message before it.
    kUnknownMessage = 32767
  };
  CachePerfMsg() : small_msg_{1} {
    small_msg_.body.msg.type = kUnknownMessage;
    small_msg_.body.msg.proto_sz = 0;
    small_msg_.body.msg.proto_buffer[0] = 0;
  }
  ~CachePerfMsg() = default;

  char *GetMutableBuffer() { return small_msg_.body.msg.proto_buffer; }

  Status Send(int32_t qID);

  void SetType(MessageType requestType) { small_msg_.body.msg.type = requestType; }
  void SetProtoBufSz(size_t sz) { small_msg_.body.msg.proto_sz = sz; }

  MessageType GetType() const { return static_cast<MessageType>(small_msg_.body.msg.type); }
  size_t GetProtoBufSz() const { return small_msg_.body.msg.proto_sz; }

  Status Receive(int32_t qID);

 private:
  CacheSmallMsg small_msg_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PERF_MSG_H_
