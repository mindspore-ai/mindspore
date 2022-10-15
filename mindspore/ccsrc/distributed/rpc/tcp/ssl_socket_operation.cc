/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ps/core/communicator/ssl_client.h"
#include "ps/core/communicator/ssl_wrapper.h"
#include "utils/ms_exception.h"
#include "distributed/rpc/tcp/ssl_socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
constexpr int SSL_HANDSHAKE_OK = 1;

bool SSLSocketOperation::Initialize() {
  ps::core::SSLWrapper::GetInstance().InitSSL();
  return true;
}
ssize_t SSLSocketOperation::ReceivePeek(Connection *, char *, uint32_t) { return 0; }

int SSLSocketOperation::Receive(Connection *connection, char *recvBuf, size_t totalRecvLen, size_t *recvLen) {
  if (connection == nullptr || recvBuf == nullptr || recvLen == nullptr) {
    return IO_RW_ERROR;
  }
  char *curRecvBuf = recvBuf;
  *recvLen = 0;

  // Continue to receive data util the number of bytes reaches the expectation.
  while (*recvLen != totalRecvLen) {
    auto retval = SSL_read(ssl_, curRecvBuf, SizeToInt(totalRecvLen - *recvLen));
    // Data received successfully.
    if (retval > 0) {
      size_t received_bytes = static_cast<size_t>(retval);
      *recvLen += received_bytes;
      if (*recvLen == totalRecvLen) {
        return IO_RW_OK;
      }
      curRecvBuf = curRecvBuf + received_bytes;
    } else {
      // Failed to receive data.
      if (retval < 0 && errno == EAGAIN) {
        return IO_RW_OK;
      }

      int err = SSL_get_error(ssl_, retval);
      switch (err) {
        case SSL_ERROR_WANT_WRITE:
        case SSL_ERROR_WANT_READ:
          return IO_RW_OK;
        default:
          connection->error_code = err;
          MS_LOG(ERROR) << "Failed to call SSL_read and errno is: " << errno;
          return IO_RW_ERROR;
      }
    }
  }
  return IO_RW_OK;
}

int SSLSocketOperation::ReceiveMessage(Connection *connection, struct msghdr *recvMsg, size_t totalRecvLen,
                                       size_t *recvLen) {
  if (connection == nullptr || recvMsg == nullptr || recvLen == nullptr) {
    return IO_RW_ERROR;
  }
  if (totalRecvLen == 0) {
    return IO_RW_OK;
  }
  *recvLen = 0;
  const size_t msg_idx = 0;

  // Continue to receive data util the number of bytes reaches the expectation.
  while (*recvLen != totalRecvLen) {
    auto retval = SSL_read(ssl_, recvMsg->msg_iov[msg_idx].iov_base, SizeToInt(recvMsg->msg_iov[msg_idx].iov_len));
    // Data received successfully.
    if (retval > 0) {
      size_t received_bytes = static_cast<size_t>(retval);
      *recvLen += received_bytes;
      if (*recvLen == totalRecvLen) {
        recvMsg->msg_iovlen = 0;
        break;
      }

      if (recvMsg->msg_iov[msg_idx].iov_len > received_bytes) {
        recvMsg->msg_iov[msg_idx].iov_len -= received_bytes;
        recvMsg->msg_iov[msg_idx].iov_base =
          reinterpret_cast<char *>(recvMsg->msg_iov[msg_idx].iov_base) + received_bytes;
      } else {
        // Check if the index is out of bounds.
        if (recvMsg->msg_iovlen <= 1) {
          return IO_RW_ERROR;
        }
        recvMsg->msg_iov = &recvMsg->msg_iov[1];
        recvMsg->msg_iovlen -= 1;
      }
    } else {
      // Failed to receive data.
      if (retval < 0 && errno == EAGAIN) {
        return IO_RW_OK;
      }
      int err = SSL_get_error(ssl_, retval);
      switch (err) {
        case SSL_ERROR_WANT_WRITE:
        case SSL_ERROR_WANT_READ:
          return IO_RW_OK;
        default:
          connection->error_code = err;
          MS_LOG(ERROR) << "Failed to call SSL_read and errno is: " << errno;
          return IO_RW_ERROR;
      }
    }
  }
  return IO_RW_OK;
}

int SSLSocketOperation::SendMessage(Connection *connection, struct msghdr *sendMsg, size_t totalSendLen,
                                    size_t *sendLen) {
  if (connection == nullptr || sendMsg == nullptr || sendLen == nullptr) {
    return IO_RW_ERROR;
  }
  *sendLen = 0;
  const size_t msg_idx = 0;

  // Continue to send data util all the bytes have been sent out.
  while (*sendLen != totalSendLen) {
    auto retval = SSL_write(ssl_, sendMsg->msg_iov[msg_idx].iov_base, SizeToInt(sendMsg->msg_iov[msg_idx].iov_len));
    // Data sent successfully.
    if (retval > 0) {
      size_t send_bytes = static_cast<size_t>(retval);
      *sendLen += send_bytes;

      if (*sendLen == totalSendLen) {
        sendMsg->msg_iovlen = 0;
        break;
      }
      if (sendMsg->msg_iov[msg_idx].iov_len > send_bytes) {
        sendMsg->msg_iov[msg_idx].iov_len -= send_bytes;
        sendMsg->msg_iov[msg_idx].iov_base = reinterpret_cast<char *>(sendMsg->msg_iov[msg_idx].iov_base) + send_bytes;
      } else {
        // Check if the index is out of bounds.
        if (msg_idx + 1 >= sendMsg->msg_iovlen) {
          return IO_RW_ERROR;
        }
        sendMsg->msg_iov = &sendMsg->msg_iov[msg_idx + 1];
        sendMsg->msg_iovlen -= 1;
      }
    } else {
      // Failed to send data.
      if (retval < 0 && errno == EAGAIN) {
        return IO_RW_OK;
      }
      int err = SSL_get_error(ssl_, retval);
      switch (err) {
        case SSL_ERROR_WANT_WRITE:
        case SSL_ERROR_WANT_READ:
          return IO_RW_OK;
        default:
          connection->error_code = err;
          MS_LOG(ERROR) << "Failed to call SSL_write and errno is: " << errno;
          return IO_RW_ERROR;
      }
    }
  }
  return IO_RW_OK;
}

void SSLSocketOperation::Close(Connection *connection) {
  // Destroy the ssl.
  if (ssl_ != nullptr) {
    (void)SSL_clear(ssl_);
    SSL_free(ssl_);
    ssl_ = nullptr;
  }
  if (connection != nullptr) {
    // Close the socket.
    (void)close(connection->socket_fd);
    connection->socket_fd = -1;
  }
}

void SSLSocketOperation::NewConnEventHandler(int fd, uint32_t events, void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);
  if (conn == nullptr) {
    return;
  }
  uint32_t error = events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP);
  if (error > 0) {
    conn->state = ConnectionState::kDisconnecting;
    return;
  }

  // Initialize the ssl.
  if (ssl_ == nullptr) {
    ssl_ = SSL_new(ps::core::SSLWrapper::GetInstance().GetSSLCtx());
    if (ssl_ == nullptr) {
      MS_LOG(ERROR) << "Failed to call SSL_new for server fd: " << fd;
      conn->state = ConnectionState::kDisconnecting;
      return;
    }
    (void)SSL_set_fd(ssl_, fd);
    SSL_set_accept_state(ssl_);
  }
  Handshake(fd, conn);
}

void SSLSocketOperation::ConnEstablishedEventHandler(int fd, uint32_t events, void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);
  if (conn == nullptr) {
    return;
  }
  uint32_t error = events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP);
  if (error > 0) {
    conn->state = ConnectionState::kDisconnecting;
    return;
  }

  // Initialize the ssl.
  if (ssl_ == nullptr) {
    ssl_ = SSL_new(ps::core::SSLClient::GetInstance().GetSSLCtx());
    if (ssl_ == nullptr) {
      MS_LOG(ERROR) << "Failed to call SSL_new for client fd: " << fd;
      conn->state = ConnectionState::kDisconnecting;
      return;
    }
    (void)SSL_set_fd(ssl_, fd);
    SSL_set_connect_state(ssl_);
  }
  Handshake(fd, conn);
}

void SSLSocketOperation::Handshake(int fd, Connection *conn) const {
  if (conn == nullptr || conn->recv_event_loop == nullptr || ssl_ == nullptr ||
      conn->state == ConnectionState::kConnected) {
    return;
  }

  int retval = SSL_do_handshake(ssl_);
  // Handshake successfully.
  if (retval == SSL_HANDSHAKE_OK) {
#if OPENSSL_VERSION_NUMBER < 0x10100000L
#ifdef SSL3_FLAGS_NO_RENEGOTIATE_CIPHERS
    if (ssl_->s3) {
      ssl_->s3->flags |= SSL3_FLAGS_NO_RENEGOTIATE_CIPHERS;
    }
#endif
#endif
    (void)conn->recv_event_loop->UpdateEpollEvent(fd, EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLRDHUP);
    conn->state = ConnectionState::kConnected;
    return;
  }

  int err = SSL_get_error(ssl_, retval);
  if (err < 0) {
    MS_LOG(ERROR) << "Invalid error code: " << err;
    return;
  }
  auto err_msg = ERR_reason_error_string(static_cast<uint64_t>(err));
  MS_LOG(WARNING) << "Failed to do the ssl handshake, retval: " << retval << ", errno: " << err
                  << ", err info: " << err_msg;
  if (err == SSL_ERROR_WANT_WRITE) {
    (void)conn->recv_event_loop->UpdateEpollEvent(fd, EPOLLOUT | EPOLLHUP | EPOLLERR | EPOLLRDHUP);
  } else if (err == SSL_ERROR_WANT_READ) {
    (void)conn->recv_event_loop->UpdateEpollEvent(fd, EPOLLIN | EPOLLHUP | EPOLLERR | EPOLLRDHUP);
  } else {
    // Failed to handshake. Throw exception and catch it in main thread.
    try {
      MS_LOG(ERROR) << "ssl handshake info -- retval:" << retval << ", error:" << err << ", errno:" << errno
                    << ", conn:" << conn->send_to.c_str();
      uint64_t error = 0;
      while ((error = ERR_get_error()) > 0) {
        MS_LOG(ERROR) << "ssl handshake errno: " << error << ", err info: " << ERR_reason_error_string(error);
      }
      conn->error_code = err;
      conn->state = ConnectionState::kDisconnecting;
      MS_LOG(EXCEPTION) << "Failed to do the ssl handshake, retval: " << retval << ", errno: " << err
                        << ", err info: " << err_msg;
    } catch (const std::exception &e) {
      MsException::Instance().SetException();
    }
  }
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
