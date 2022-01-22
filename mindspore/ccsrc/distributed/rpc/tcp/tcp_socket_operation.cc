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

#include "distributed/rpc/tcp/tcp_socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
constexpr int EAGAIN_RETRY = 2;

int TCPSocketOperation::ReceivePeek(Connection *connection, char *recvBuf, uint32_t recvLen) {
  return recv(connection->socket_fd, recvBuf, recvLen, MSG_PEEK);
}

int TCPSocketOperation::Receive(Connection *connection, char *recvBuf, uint32_t totalRecvLen, uint32_t *recvLen) {
  char *curRecvBuf = recvBuf;
  int fd = connection->socket_fd;

  *recvLen = 0;
  while (*recvLen != totalRecvLen) {
    int retval = recv(fd, curRecvBuf, totalRecvLen - *recvLen, static_cast<int>(0));
    if (retval > 0) {
      *recvLen += IntToUint(retval);
      if (*recvLen == totalRecvLen) {
        return UintToInt(totalRecvLen);
      }
      curRecvBuf = curRecvBuf + retval;
      // Failed to receive message.
    } else if (retval < 0) {
      if (EAGAIN == errno) {
        return UintToInt(*recvLen);
      } else if (ECONNRESET == errno || ECONNABORTED == errno || ENOTCONN == errno || EPIPE == errno) {
        connection->error_code = errno;
        return -1;
      } else {
        return UintToInt(*recvLen);
      }
    } else {
      connection->error_code = errno;
      return -1;
    }
  }
  return UintToInt(*recvLen);
}

int TCPSocketOperation::ReceiveMessage(Connection *connection, struct msghdr *recvMsg, uint32_t recvLen) {
  uint32_t totalRecvLen = recvLen;

  if (totalRecvLen == 0) {
    return 0;
  }

  while (totalRecvLen) {
    auto retval = recvmsg(connection->socket_fd, recvMsg, 0);
    if (retval > 0) {
      totalRecvLen -= IntToSize(retval);
      if (totalRecvLen == 0) {
        recvMsg->msg_iovlen = 0;
        break;
      }

      unsigned int iovlen = recvMsg->msg_iovlen;
      if (iovlen > 0) {
        size_t tmpLen = 0;
        for (unsigned int i = 0; i < iovlen; ++i) {
          if (recvMsg->msg_iov[i].iov_len + tmpLen <= (size_t)retval) {
            tmpLen += recvMsg->msg_iov[i].iov_len;
          } else {
            recvMsg->msg_iov[i].iov_len -= IntToSize(retval - tmpLen);
            recvMsg->msg_iov[i].iov_base =
              reinterpret_cast<char *>(recvMsg->msg_iov[i].iov_base) + static_cast<unsigned int>(retval) - tmpLen;

            recvMsg->msg_iov = &recvMsg->msg_iov[i];
            recvMsg->msg_iovlen -= (i + 1);
            break;
          }
        }
      }
    } else if (retval == 0) {
      return UintToInt(-1);
    } else {
      if (EAGAIN == errno) {
        return recvLen - totalRecvLen;
      } else if (ECONNRESET == errno || ECONNABORTED == errno || ENOTCONN == errno || EPIPE == errno) {
        connection->error_code = UintToInt(errno);
        return -1;
      } else {
        return UintToInt(recvLen - totalRecvLen);
      }
    }
  }
  return recvLen;
}

int TCPSocketOperation::SendMessage(Connection *connection, struct msghdr *sendMsg, uint32_t *sendLen) {
  int eagainCount = EAGAIN_RETRY;
  uint32_t totalLen = *sendLen;
  int32_t unsendLen = *sendLen;

  while (*sendLen != 0) {
    int retval = sendmsg(connection->socket_fd, sendMsg, MSG_NOSIGNAL);
    if (retval < 0) {
      --eagainCount;
      if (errno != EAGAIN) {
        connection->error_code = errno;
        unsendLen = -1;
        break;
      } else if (eagainCount == 0) {
        unsendLen = 0;
        break;
      }
    } else {
      *sendLen -= retval;

      if (*sendLen == 0) {
        sendMsg->msg_iovlen = 0;
        break;
      }

      size_t tmpBytes = 0;
      for (unsigned int i = 0; i < sendMsg->msg_iovlen; ++i) {
        if (sendMsg->msg_iov[i].iov_len + tmpBytes < IntToSize(retval)) {
          tmpBytes += sendMsg->msg_iov[i].iov_len;
        } else {
          sendMsg->msg_iov[i].iov_len -= (retval - tmpBytes);
          sendMsg->msg_iov[i].iov_base =
            reinterpret_cast<char *>(sendMsg->msg_iov[i].iov_base) + static_cast<unsigned int>(retval) - tmpBytes;

          sendMsg->msg_iov = &sendMsg->msg_iov[i];
          sendMsg->msg_iovlen -= (i + 1);
          break;
        }
      }
      eagainCount = EAGAIN_RETRY;
    }
  }
  if (unsendLen > 0) {
    unsendLen = UintToInt(totalLen - *sendLen);
  }
  return unsendLen;
}

void TCPSocketOperation::Close(Connection *connection) {
  (void)close(connection->socket_fd);
  connection->socket_fd = -1;
}

// accept new conn event handle
void TCPSocketOperation::NewConnEventHandler(int fd, uint32_t events, void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);
  conn->state = ConnectionState::kConnected;
  return;
}

void TCPSocketOperation::ConnEstablishedEventHandler(int fd, uint32_t events, void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);
  conn->state = ConnectionState::kConnected;
  return;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
