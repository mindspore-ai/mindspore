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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_MESSAGE_HANDLER_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_MESSAGE_HANDLER_H_

namespace mindspore {
namespace ps {
namespace core {
typedef void (*RefBufferRelCallback)(const void *data, size_t datalen, void *extra);
// MessageHandler class is used to handle requests from clients and send response from server.
// It's the base class of HttpMsgHandler and TcpMsgHandler.
class MessageHandler {
 public:
  MessageHandler() = default;
  virtual ~MessageHandler() = default;

  // Raw data of this message in bytes.
  virtual void *data() const = 0;

  // Raw data size of this message.(Number of bytes)
  virtual size_t len() const = 0;

  virtual bool SendResponse(const void *data, const size_t &len) = 0;
  virtual bool SendResponseInference(const void *data, const size_t &len, RefBufferRelCallback cb) {
    auto ret = SendResponse(data, len);
    if (cb) {
      cb(data, len, nullptr);
    }
    return ret;
  }
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_MESSAGE_HANDLER_H_
