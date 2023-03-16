/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "include/backend/distributed/rpc/tcp/constants.h"
#include "distributed/rpc/tcp/tcp_socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
void DoDisconnect(int fd, Connection *conn, uint32_t error, int soError) {
  if (conn == nullptr) {
    return;
  }
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
  if (conn == nullptr || conn->socket_operation == nullptr) {
    return;
  }
  conn->socket_operation->ConnEstablishedEventHandler(fd, events, context);
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
  if (tcpmgr == nullptr || tcpmgr->conn_pool_ == nullptr) {
    return;
  }
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

  // This new connection will be added to connection pool.
  Connection *conn = new (std::nothrow) Connection();
  if (conn == nullptr) {
    MS_LOG(ERROR) << "Failed to create new connection, server fd:" << server << ", events: " << events
                  << ", accept fd: " << acceptFd;
    if (close(acceptFd) != 0) {
      MS_LOG(ERROR) << "Failed to close fd: " << acceptFd;
    }
    return;
  }
  conn->enable_ssl = tcpmgr->enable_ssl_;

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
  conn->destination = SocketOperation::GetPeer(acceptFd);
  conn->peer = conn->destination;

  conn->is_remote = true;
  conn->recv_event_loop = tcpmgr->recv_event_loop_;
  conn->send_event_loop = tcpmgr->send_event_loop_;

  conn->conn_mutex = tcpmgr->conn_mutex_;
  conn->message_handler = tcpmgr->message_handler_;

  conn->event_callback = std::bind(&TCPComm::EventCallBack, tcpmgr, std::placeholders::_1);
  conn->write_callback = std::bind(&TCPComm::WriteCallBack, tcpmgr, std::placeholders::_1);
  conn->read_callback = std::bind(&TCPComm::ReadCallBack, tcpmgr, std::placeholders::_1);

  conn->SetAllocateCallback(tcpmgr->allocate_cb());

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
    conn = nullptr;
    return;
  }
  tcpmgr->conn_pool_->AddConnection(conn);
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

bool TCPComm::StartServerSocket(const std::string &url, const MemAllocateCallback &allocate_cb) {
  server_fd_ = SocketOperation::Listen(url);
  if (server_fd_ < 0) {
    MS_LOG(ERROR) << "Failed to call socket listen, url: " << url.c_str();
    return false;
  }
  url_ = url;
  allocate_cb_ = allocate_cb;
  size_t index = url.find(URL_PROTOCOL_IP_SEPARATOR);
  if (index != std::string::npos) {
    index = index + sizeof(URL_PROTOCOL_IP_SEPARATOR) - 1;
    if (index < url.length()) {
      url_ = url.substr(index);
    }
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

bool TCPComm::StartServerSocket(const MemAllocateCallback &allocate_cb) {
  auto ip = SocketOperation::GetLocalIP();
  // The port 0 means that the port will be allocated randomly by the os system.
  auto url = ip + ":0";
  return StartServerSocket(url, allocate_cb);
}

int TCPComm::GetServerFd() const { return server_fd_; }

void TCPComm::ReadCallBack(void *connection) {
  const int max_recv_count = 3;
  Connection *conn = reinterpret_cast<Connection *>(connection);
  if (conn == nullptr) {
    return;
  }
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
  if (conn == nullptr) {
    return;
  }
  if (conn->state == ConnectionState::kConnected) {
    conn->conn_mutex->lock();
    (void)conn->Flush();
    conn->conn_mutex->unlock();
  } else if (conn->state == ConnectionState::kDisconnecting) {
    std::lock_guard<std::mutex> lock(*conn_mutex_);
    conn_pool_->DeleteConnection(conn->destination);
  }
}

void TCPComm::WriteCallBack(void *connection) {
  Connection *conn = reinterpret_cast<Connection *>(connection);
  if (conn == nullptr) {
    return;
  }
  if (conn->state == ConnectionState::kConnected) {
    conn->conn_mutex->lock();
    (void)conn->Flush();
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
                                                static_cast<uint32_t>(EPOLLOUT | EPOLLHUP | EPOLLRDHUP | EPOLLERR),
                                                ConnectedEventHandler, reinterpret_cast<void *>(conn));
}

/* static method */
int TCPComm::DoConnect(Connection *conn, const struct sockaddr *sa, socklen_t saLen) {
  if (conn == nullptr || conn->recv_event_loop == nullptr || sa == nullptr) {
    return RPC_ERROR;
  }
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

bool TCPComm::Send(MessageBase *msg, size_t *const send_bytes, bool sync) {
  if (msg == nullptr) {
    return false;
  }
  auto task = [msg, send_bytes, this] {
    std::lock_guard<std::mutex> lock(*conn_mutex_);
    // Search connection by the target address
    std::string destination = msg->to.Url();
    Connection *conn = conn_pool_->FindConnection(destination);
    if (conn == nullptr) {
      MS_LOG(ERROR) << "Can not found remote link and send fail name: " << msg->name.c_str()
                    << ", from: " << msg->from.Url().c_str() << ", to: " << destination;
      DropMessage(msg);
      return false;
    }

    if (conn->send_message_queue.size() >= SENDMSG_QUEUELEN) {
      MS_LOG(WARNING) << "The message queue is full(max len:" << SENDMSG_QUEUELEN
                      << ") and the name of dropped message is: " << msg->name.c_str() << ", fd: " << conn->socket_fd
                      << ", to: " << conn->destination.c_str();
      if (!conn->FreeMessageMemory(msg)) {
        MS_LOG(ERROR) << "Failed to free memory of the message.";
      }
      DropMessage(msg);
      return false;
    }

    if (conn->state != ConnectionState::kConnected) {
      MS_LOG(WARNING) << "Invalid connection state " << conn->state
                      << " and the name of dropped message is: " << msg->name.c_str() << ", fd: " << conn->socket_fd
                      << ", to: " << conn->destination.c_str();
      if (!conn->FreeMessageMemory(msg)) {
        MS_LOG(ERROR) << "Failed to free memory of the message.";
      }
      DropMessage(msg);
      return false;
    }

    if (conn->total_send_len == 0) {
      conn->FillSendMessage(msg, url_, false);
    } else {
      (void)conn->send_message_queue.emplace(msg);
    }
    auto bytes = conn->Flush();
    if (send_bytes != nullptr) {
      *send_bytes = bytes;
    }
    return true;
  };
  if (sync) {
    return task();
  } else {
    send_event_loop_->AddTask(task);
    return true;
  }
}

bool TCPComm::Flush(const std::string &dst_url) {
  Connection *conn = conn_pool_->FindConnection(dst_url);
  if (conn == nullptr) {
    MS_LOG(ERROR) << "Can not find the connection to url: " << dst_url;
    return false;
  } else {
    std::lock_guard<std::mutex> lock(*(conn->conn_mutex));
    return (conn->Flush() >= 0);
  }
}

bool TCPComm::Connect(const std::string &dst_url, const MemFreeCallback &free_cb) {
  MS_EXCEPTION_IF_NULL(conn_mutex_);
  MS_EXCEPTION_IF_NULL(conn_pool_);
  if (!free_cb) {
    MS_LOG(EXCEPTION) << "The message callback is empty.";
  }

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
    conn->enable_ssl = enable_ssl_;
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
    conn->event_callback = std::bind(&TCPComm::EventCallBack, this, std::placeholders::_1);
    conn->write_callback = std::bind(&TCPComm::WriteCallBack, this, std::placeholders::_1);
    conn->read_callback = std::bind(&TCPComm::ReadCallBack, this, std::placeholders::_1);

    int ret = TCPComm::DoConnect(conn, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
    if (ret < 0) {
      MS_LOG(ERROR) << "Failed to do connect and link fail destination: " << dst_url;
      if (conn->socket_operation != nullptr) {
        delete conn->socket_operation;
        conn->socket_operation = nullptr;
      }
      delete conn;
      conn = nullptr;
      return false;
    }
    conn->source = SocketOperation::GetLocalIP() + ":" + std::to_string(SocketOperation::GetPort(sock_fd));
    conn->destination = dst_url;

    // Check the state of this new created connection.
    uint32_t interval = 1;
    size_t retry = 3;
    while (conn->state < ConnectionState::kConnected && retry-- > 0) {
      MS_LOG(WARNING) << "Waiting for the state of the connection to " << dst_url << " to be connected...";
      (void)sleep(interval);
    }
    if (conn->state != ConnectionState::kConnected) {
      return false;
    }
    conn_pool_->AddConnection(conn);
    conn->SetMessageFreeCallback(free_cb);
  }
  conn_pool_->AddConnInfo(conn->socket_fd, dst_url, nullptr);
  MS_LOG(INFO) << "Connected to destination: " << dst_url;
  return true;
}

bool TCPComm::IsConnected(const std::string &dst_url) {
  MS_EXCEPTION_IF_NULL(conn_pool_);
  Connection *conn = conn_pool_->FindConnection(dst_url);
  if (conn != nullptr && conn->state == ConnectionState::kConnected) {
    return true;
  }
  return false;
}

bool TCPComm::Disconnect(const std::string &dst_url) {
  MS_EXCEPTION_IF_NULL(conn_mutex_);
  MS_EXCEPTION_IF_NULL(conn_pool_);
  MS_EXCEPTION_IF_NULL(recv_event_loop_);
  MS_EXCEPTION_IF_NULL(send_event_loop_);

  unsigned int interval = 100000;
  size_t retry = 30;
  while (recv_event_loop_->RemainingTaskNum() != 0 && send_event_loop_->RemainingTaskNum() != 0 && retry > 0) {
    (void)usleep(interval);
    retry--;
  }
  if (recv_event_loop_->RemainingTaskNum() > 0 || send_event_loop_->RemainingTaskNum() > 0) {
    MS_LOG(ERROR) << "Failed to disconnect from url " << dst_url
                  << ", because there are still pending tasks to be executed, please try later.";
    return false;
  }
  std::lock_guard<std::mutex> lock(*conn_mutex_);
  auto conn = conn_pool_->FindConnection(dst_url);
  if (conn != nullptr) {
    std::lock_guard<std::mutex> conn_lock(conn->conn_owned_mutex_);
    conn_pool_->DeleteConnection(dst_url);
  }
  return true;
}

Connection *TCPComm::CreateDefaultConn(const std::string &to) {
  Connection *conn = new (std::nothrow) Connection();
  if (conn == nullptr) {
    MS_LOG(ERROR) << "Failed to create new connection and reconnect fail to: " << to.c_str();
    return conn;
  }
  conn->enable_ssl = enable_ssl_;
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
