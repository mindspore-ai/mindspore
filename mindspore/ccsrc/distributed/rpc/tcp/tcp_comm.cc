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

#include "distributed/rpc/tcp/tcp_comm.h"

#include <mutex>
#include <utility>
#include <memory>

#include "actor/aid.h"
#include "distributed/rpc/tcp/constants.h"
#include "distributed/rpc/tcp/tcp_socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
void DoDisconnect(int fd, Connection *conn, uint32_t error, int soError) {
  if (LOG_CHECK_EVERY_N()) {
    MS_LOG(INFO) << "Failed to call connect, fd: " << fd << ", to: " << conn->destination.c_str()
                 << ", events: " << error << ", errno: " << soError;
  }

  conn->state = ConnectionState::kDisconnecting;
  conn->error_code = soError;
  conn->event_callback(conn);
  return;
}

void ConnectedEventHandler(int fd, uint32_t events, void *context) {
  uint32_t error = events & (EPOLLERR | EPOLLHUP | EPOLLRDHUP);
  int soError = 0;
  Connection *conn = reinterpret_cast<Connection *>(context);
  conn->socket_operation->ConnEstablishedEventHandler(context);
  if (conn->state == ConnectionState::kDisconnecting) {
    DoDisconnect(fd, conn, error, soError);
    return;
  } else if (conn->state != ConnectionState::kConnected) {
    return;
  }

  if (!conn->ReconnectSourceSocket(fd, events, &soError, error)) {
    DoDisconnect(fd, conn, error, soError);
    return;
  }
  if (conn->write_callback) {
    conn->write_callback(conn);
  }
  return;
}

void OnAccept(int server, uint32_t events, void *arg) {
  if (events & (EPOLLHUP | EPOLLERR)) {
    MS_LOG(ERROR) << "Invalid error event, server fd: " << server << ", events: " << events;
    return;
  }
  TCPComm *tcpmgr = reinterpret_cast<TCPComm *>(arg);
  if (tcpmgr->recv_event_loop_ == nullptr) {
    MS_LOG(ERROR) << "EventLoop is null, server fd: " << server << ", events: " << events;
    return;
  }

  // accept connection
  auto acceptFd = SocketOperation::Accept(server);
  if (acceptFd < 0) {
    MS_LOG(ERROR) << "Failed to call accept, server fd: " << server << ", events: " << events;
    return;
  }

  Connection *conn = new (std::nothrow) Connection();
  if (conn == nullptr) {
    MS_LOG(ERROR) << "Failed to create new connection, server fd:" << server << ", events: " << events
                  << ", accept fd: " << acceptFd;
    if (close(acceptFd) != 0) {
      MS_LOG(ERROR) << "Failed to close fd: " << acceptFd;
    }
    return;
  }

  // init metrics
  conn->send_metrics = new (std::nothrow) SendMetrics();
  if (conn->send_metrics == nullptr) {
    MS_LOG(ERROR) << "Failed to create connection metrics, server fd: " << server << ", events: " << events
                  << ", accept fd: " << acceptFd;
    if (close(acceptFd) != 0) {
      MS_LOG(ERROR) << "Failed to close fd: " << acceptFd;
    }
    delete conn;
    return;
  }

  conn->socket_fd = acceptFd;
  conn->source = tcpmgr->url_;
  conn->peer = SocketOperation::GetPeer(acceptFd);

  conn->is_remote = true;
  conn->recv_event_loop = tcpmgr->recv_event_loop_;
  conn->send_event_loop = tcpmgr->send_event_loop_;

  conn->conn_mutex = tcpmgr->conn_mutex_;
  conn->message_handler = tcpmgr->message_handler_;

  conn->event_callback = TCPComm::EventCallBack;
  conn->write_callback = TCPComm::WriteCallBack;
  conn->read_callback = TCPComm::ReadCallBack;

  int retval = conn->Initialize();
  if (retval != RPC_OK) {
    MS_LOG(ERROR) << "Failed to add accept fd event, server fd: " << server << ", events: " << events
                  << ", accept fd: " << acceptFd;
    if (close(acceptFd) != 0) {
      MS_LOG(ERROR) << "Failed to close fd: " << acceptFd;
    }
    acceptFd = -1;
    delete conn->send_metrics;
    delete conn;
    return;
  }
  tcpmgr->conn_pool_->AddConnection(conn);
}

int DoSend(Connection *conn) {
  int total_send_bytes = 0;
  while (!conn->send_message_queue.empty() || conn->total_send_len != 0) {
    if (conn->total_send_len == 0) {
      conn->FillSendMessage(conn->send_message_queue.front(), conn->source, false);
      conn->send_message_queue.pop();
    }

    int sendLen = conn->socket_operation->SendMessage(conn, &conn->send_kernel_msg, &conn->total_send_len);
    if (sendLen > 0) {
      total_send_bytes += sendLen;
      if (conn->total_send_len == 0) {
        // update metrics
        conn->send_metrics->UpdateError(false);

        conn->output_buffer_size -= conn->send_message->body.size();
        delete conn->send_message;
        conn->send_message = nullptr;
      }
    } else if (sendLen == 0) {
      // EAGAIN
      (void)conn->recv_event_loop->UpdateEpollEvent(conn->socket_fd, EPOLLOUT | EPOLLIN | EPOLLHUP | EPOLLERR);
      break;
    } else {
      // update metrics
      conn->send_metrics->UpdateError(true, conn->error_code);
      conn->state = ConnectionState::kDisconnecting;
      break;
    }
  }
  return total_send_bytes;
}

void TCPComm::SetMessageHandler(const MessageHandler &handler) { message_handler_ = handler; }

bool TCPComm::Initialize() {
  conn_pool_ = std::make_shared<ConnectionPool>();
  MS_EXCEPTION_IF_NULL(conn_pool_);

  conn_mutex_ = std::make_shared<std::mutex>();
  MS_EXCEPTION_IF_NULL(conn_mutex_);

  recv_event_loop_ = new (std::nothrow) EventLoop();
  if (recv_event_loop_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create recv evLoop.";
    return false;
  }

  bool ok = recv_event_loop_->Initialize(TCP_RECV_EVLOOP_THREADNAME);
  if (!ok) {
    MS_LOG(ERROR) << "Failed to init recv evLoop";
    delete recv_event_loop_;
    recv_event_loop_ = nullptr;
    return false;
  }

  send_event_loop_ = new (std::nothrow) EventLoop();
  if (send_event_loop_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create send evLoop.";
    delete recv_event_loop_;
    recv_event_loop_ = nullptr;
    return false;
  }
  ok = send_event_loop_->Initialize(TCP_SEND_EVLOOP_THREADNAME);
  if (!ok) {
    MS_LOG(ERROR) << "Failed to init send evLoop";
    delete recv_event_loop_;
    recv_event_loop_ = nullptr;
    delete send_event_loop_;
    send_event_loop_ = nullptr;
    return false;
  }

  return true;
}

bool TCPComm::StartServerSocket(const std::string &url) {
  server_fd_ = SocketOperation::Listen(url);
  if (server_fd_ < 0) {
    MS_LOG(ERROR) << "Failed to call socket listen, url: " << url.c_str();
    return false;
  }
  url_ = url;
  size_t index = url.find(URL_PROTOCOL_IP_SEPARATOR);
  if (index != std::string::npos) {
    url_ = url.substr(index + sizeof(URL_PROTOCOL_IP_SEPARATOR) - 1);
  }

  // Register read event callback for server socket
  int retval = recv_event_loop_->SetEventHandler(server_fd_, EPOLLIN | EPOLLHUP | EPOLLERR, OnAccept,
                                                 reinterpret_cast<void *>(this));
  if (retval != RPC_OK) {
    MS_LOG(ERROR) << "Failed to add server event, url: " << url.c_str();
    return false;
  }
  MS_LOG(INFO) << "Start server succ, fd: " << server_fd_ << ", url: " << url.c_str();
  return true;
}

bool TCPComm::StartServerSocket() {
  auto ip = SocketOperation::GetLocalIP();
  // The port 0 means that the port will be allocated randomly by the os system.
  auto url = ip + ":0";
  return StartServerSocket(url);
}

int TCPComm::GetServerFd() const { return server_fd_; }

void TCPComm::ReadCallBack(void *connection) {
  const int max_recv_count = 3;
  Connection *conn = reinterpret_cast<Connection *>(connection);
  int count = 0;
  int retval = 0;
  do {
    retval = ReceiveMessage(conn);
    ++count;
  } while (retval > 0 && count < max_recv_count);

  return;
}

void TCPComm::EventCallBack(void *connection) {
  Connection *conn = reinterpret_cast<Connection *>(connection);

  if (conn->state == ConnectionState::kConnected) {
    conn->conn_mutex->lock();
    (void)DoSend(conn);
    conn->conn_mutex->unlock();
  } else if (conn->state == ConnectionState::kDisconnecting) {
    conn->conn_mutex->lock();
    conn->conn_mutex->unlock();
  }
}

void TCPComm::WriteCallBack(void *connection) {
  Connection *conn = reinterpret_cast<Connection *>(connection);
  if (conn->state == ConnectionState::kConnected) {
    conn->conn_mutex->lock();
    (void)DoSend(conn);
    conn->conn_mutex->unlock();
  }
}

/* static method */
int TCPComm::ReceiveMessage(Connection *conn) {
  std::lock_guard<std::mutex> lock(*conn->conn_mutex);
  conn->CheckMessageType();
  switch (conn->recv_message_type) {
    case ParseType::kTcpMsg:
      return conn->ReceiveMessage();
    default:
      return 0;
  }
}

/* static method */
int TCPComm::SetConnectedHandler(Connection *conn) {
  /* add to epoll */
  return conn->recv_event_loop->SetEventHandler(conn->socket_fd,
                                                (uint32_t)(EPOLLOUT | EPOLLHUP | EPOLLRDHUP | EPOLLERR),
                                                ConnectedEventHandler, reinterpret_cast<void *>(conn));
}

/* static method */
int TCPComm::DoConnect(Connection *conn, const struct sockaddr *sa, socklen_t saLen) {
  int retval = 0;
  uint16_t localPort = 0;

  retval = SocketOperation::Connect(conn->socket_fd, sa, saLen, &localPort);
  if (retval != RPC_OK) {
    return RPC_ERROR;
  }

  // Init connection metrics.
  if (conn->send_metrics == nullptr) {
    conn->send_metrics = new (std::nothrow) SendMetrics();
    if (conn->send_metrics == nullptr) {
      return RPC_ERROR;
    }
  }

  // Add the socket of this connection to epoll.
  retval = SetConnectedHandler(conn);
  if (retval != RPC_OK) {
    if (conn->send_metrics != nullptr) {
      delete conn->send_metrics;
      conn->send_metrics = nullptr;
    }
    return RPC_ERROR;
  }
  return RPC_OK;
}

/* static method */
void TCPComm::DropMessage(MessageBase *msg) {
  auto *ptr = msg;
  delete ptr;
  ptr = nullptr;
}

ssize_t TCPComm::Send(MessageBase *msg, bool sync) {
  auto task = [msg, this] {
    std::lock_guard<std::mutex> lock(*conn_mutex_);
    // Search connection by the target address
    Connection *conn = conn_pool_->FindConnection(msg->to.Url());
    if (conn == nullptr) {
      MS_LOG(ERROR) << "Can not found remote link and send fail name: " << msg->name.c_str()
                    << ", from: " << msg->from.Url().c_str() << ", to: " << msg->to.Url().c_str();
      DropMessage(msg);
      int error_no = -1;
      return error_no;
    }

    if (conn->send_message_queue.size() >= SENDMSG_QUEUELEN) {
      MS_LOG(WARNING) << "The message queue is full(max len:" << SENDMSG_QUEUELEN
                      << ") and the name of dropped message is: " << msg->name.c_str() << ", fd: " << conn->socket_fd
                      << ", to: " << conn->destination.c_str();
      DropMessage(msg);
      int error_no = -1;
      return error_no;
    }

    if (conn->state != ConnectionState::kConnected) {
      MS_LOG(WARNING) << "Invalid connection state " << conn->state
                      << " and the name of dropped message is: " << msg->name.c_str() << ", fd: " << conn->socket_fd
                      << ", to: " << conn->destination.c_str();
      DropMessage(msg);
      int error_no = -1;
      return error_no;
    }

    if (conn->total_send_len == 0) {
      conn->FillSendMessage(msg, url_, false);
    } else {
      (void)conn->send_message_queue.emplace(msg);
    }
    return DoSend(conn);
  };
  if (sync) {
    return task();
  } else {
    return send_event_loop_->AddTask(task);
  }
}

void TCPComm::Connect(const std::string &dst_url) {
  (void)recv_event_loop_->AddTask([dst_url, this] {
    std::lock_guard<std::mutex> lock(*conn_mutex_);

    // Search connection by the target address
    Connection *conn = conn_pool_->FindConnection(dst_url);

    if (conn == nullptr) {
      MS_LOG(INFO) << "Can not found link destination: " << dst_url;
      conn = new (std::nothrow) Connection();
      if (conn == nullptr) {
        MS_LOG(ERROR) << "Failed to create new connection and link fail destination: " << dst_url;
        return false;
      }
      conn->source = url_;
      conn->destination = dst_url;

      conn->recv_event_loop = this->recv_event_loop_;
      conn->send_event_loop = this->send_event_loop_;
      conn->conn_mutex = conn_mutex_;
      conn->message_handler = message_handler_;
      conn->InitSocketOperation();

      // Create the client socket.
      SocketAddress addr;
      if (!SocketOperation::GetSockAddr(dst_url, &addr)) {
        MS_LOG(ERROR) << "Failed to get socket address to dest url " << dst_url;
        return false;
      }
      int sock_fd = SocketOperation::CreateSocket(addr.sa.sa_family);
      if (sock_fd < 0) {
        MS_LOG(ERROR) << "Failed to create client tcp socket to dest url " << dst_url;
        return false;
      }

      conn->socket_fd = sock_fd;
      conn->event_callback = TCPComm::EventCallBack;
      conn->write_callback = TCPComm::WriteCallBack;
      conn->read_callback = TCPComm::ReadCallBack;

      int ret = TCPComm::DoConnect(conn, (struct sockaddr *)&addr, sizeof(addr));
      if (ret < 0) {
        MS_LOG(ERROR) << "Failed to do connect and link fail destination: " << dst_url;
        if (conn->socket_operation != nullptr) {
          delete conn->socket_operation;
          conn->socket_operation = nullptr;
        }
        delete conn;
        return false;
      }
      conn_pool_->AddConnection(conn);
    }
    conn_pool_->AddConnInfo(conn->socket_fd, dst_url, nullptr);
    MS_LOG(INFO) << "Connected to destination: " << dst_url;
    return true;
  });
}

bool TCPComm::IsConnected(const std::string &dst_url) {
  Connection *conn = conn_pool_->FindConnection(dst_url);
  if (conn != nullptr && conn->state == ConnectionState::kConnected) {
    return true;
  }
  return false;
}

void TCPComm::Disconnect(const std::string &dst_url) {
  (void)recv_event_loop_->AddTask([dst_url, this] {
    std::lock_guard<std::mutex> lock(*conn_mutex_);
    conn_pool_->DeleteConnection(dst_url);
    return true;
  });
}

Connection *TCPComm::CreateDefaultConn(const std::string &to) {
  Connection *conn = new (std::nothrow) Connection();
  if (conn == nullptr) {
    MS_LOG(ERROR) << "Failed to create new connection and reconnect fail to: " << to.c_str();
    return conn;
  }
  conn->source = url_.data();
  conn->destination = to;
  conn->recv_event_loop = this->recv_event_loop_;
  conn->send_event_loop = this->send_event_loop_;
  conn->conn_mutex = conn_mutex_;
  conn->message_handler = message_handler_;
  conn->InitSocketOperation();
  return conn;
}

void TCPComm::Finalize() {
  if (send_event_loop_ != nullptr) {
    MS_LOG(INFO) << "Delete send event loop";
    send_event_loop_->Finalize();
    delete send_event_loop_;
    send_event_loop_ = nullptr;
  }

  if (recv_event_loop_ != nullptr) {
    MS_LOG(INFO) << "Delete recv event loop";
    recv_event_loop_->Finalize();
    delete recv_event_loop_;
    recv_event_loop_ = nullptr;
  }

  if (server_fd_ > 0) {
    if (close(server_fd_) != 0) {
      MS_LOG(ERROR) << "Failed to close fd: " << server_fd_;
    }
    server_fd_ = -1;
  }

  if (conn_pool_ != nullptr) {
    MS_LOG(INFO) << "Delete connection pool.";
    conn_pool_->Finalize();
    conn_pool_.reset();
    conn_pool_ = nullptr;
  }
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
