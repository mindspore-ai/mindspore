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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_SOCKET_OPERATION_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_TCP_SOCKET_OPERATION_H_

#include "distributed/rpc/tcp/connection.h"
#include "distributed/rpc/tcp/socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
class TCPSocketOperation : public SocketOperation {
 public:
  int ReceivePeek(Connection *connection, char *recvBuf, uint32_t recvLen) override;
  int Receive(Connection *connection, char *recvBuf, uint32_t totRecvLen, uint32_t *recvLen) override;
  int ReceiveMessage(Connection *connection, struct msghdr *recvMsg, uint32_t recvLen) override;

  int SendMessage(Connection *connection, struct msghdr *sendMsg, uint32_t *sendLen) override;

  void Close(Connection *connection) override;

  void NewConnEventHandler(int fd, uint32_t events, void *context) override;
  void ConnEstablishedEventHandler(int fd, uint32_t events, void *context) override;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
