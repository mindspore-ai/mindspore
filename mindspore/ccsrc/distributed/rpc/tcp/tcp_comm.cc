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
#include "actor/msg.h"
#include "distributed/rpc/tcp/constants.h"
#include "distributed/rpc/tcp/tcp_socket_operation.h"
#include "distributed/rpc/tcp/connection_pool.h"

namespace mindspore {
namespace distributed {
namespace rpc {
bool TCPComm::is_http_msg_ = false;
std::vector<char> TCPComm::advertise_url_;
uint64_t TCPComm::output_buf_size_ = 0;

IOMgr::MessageHandler TCPComm::message_handler_;

int DoConnect(const std::string &to, Connection *conn, ConnectionCallBack event_callback,
              ConnectionCallBack write_callback, ConnectionCallBack read_callback) {
  SocketAddress addr;
  if (!SocketOperation::GetSockAddr(to, &addr)) {
    return -1;
  }
  int sock_fd = SocketOperation::CreateServerSocket(addr.sa.sa_family);
  if (sock_fd < 0) {
    return -1;
  }

  conn->socket_fd = sock_fd;
  conn->event_callback = event_callback;
  conn->write_callback = write_callback;
  conn->read_callback = read_callback;

  int ret = TCPComm::Connect(conn, (struct sockaddr *)&addr, sizeof(addr));
  if (ret < 0) {
    if (close(sock_fd) != 0) {
      MS_LOG(ERROR) << "Failed to close fd:" << sock_fd;
    }
    conn->socket_fd = -1;
    return -1;
  }
  return 0;
}

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
    acceptFd = -1;
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
    acceptFd = -1;
    delete conn;
    return;
  }

  conn->socket_fd = acceptFd;
  conn->source = TCPComm::advertise_url_.data();
  conn->peer = SocketOperation::GetPeer(acceptFd);

  conn->is_remote = true;
  conn->recv_event_loop = tcpmgr->recv_event_loop_;
  conn->send_event_loop = tcpmgr->send_event_loop_;

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
}

void DoSend(Connection *conn) {
  while (!conn->send_message_queue.empty() || conn->total_send_len != 0) {
    if (conn->total_send_len == 0) {
      conn->FillSendMessage(conn->send_message_queue.front(), TCPComm::advertise_url_.data(), TCPComm::IsHttpMsg());
      conn->send_message_queue.pop();
    }

    int sendLen = conn->socket_operation->SendMessage(conn, &conn->send_kernel_msg, &conn->total_send_len);
    if (sendLen > 0) {
      if (conn->total_send_len == 0) {
        // update metrics
        conn->send_metrics->UpdateError(false);

        TCPComm::output_buf_size_ -= conn->send_message->body.size();
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
}

TCPComm::~TCPComm() {
  try {
    Finalize();
  } catch (...) {
    MS_LOG(ERROR) << "Failed to finalize tcp communicator.";
  }
}

void TCPComm::SendExitMsg(const std::string &from, const std::string &to) {
  if (message_handler_ != nullptr) {
    std::unique_ptr<MessageBase> exit_msg = std::make_unique<MessageBase>(MessageBase::Type::KEXIT);
    MS_EXCEPTION_IF_NULL(exit_msg);

    exit_msg->SetFrom(AID(from));
    exit_msg->SetTo(AID(to));

    message_handler_(std::move(exit_msg));
  }
}

void TCPComm::SetMessageHandler(IOMgr::MessageHandler handler) { message_handler_ = handler; }

bool TCPComm::Initialize() {
  if (ConnectionPool::GetConnectionPool() == nullptr) {
    MS_LOG(ERROR) << "Failed to create connection pool.";
    return false;
  }
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

  if (g_httpKmsgEnable >= 0) {
    is_http_msg_ = (g_httpKmsgEnable == 0) ? false : true;
  }

  ConnectionPool::GetConnectionPool()->SetLinkPattern(is_http_msg_);
  return true;
}

bool TCPComm::StartServerSocket(const std::string &url, const std::string &aAdvertiseUrl) {
  server_fd_ = SocketOperation::Listen(url);
  if (server_fd_ < 0) {
    MS_LOG(ERROR) << "Failed to call socket listen, url: " << url.c_str()
                  << ", advertise_url_: " << advertise_url_.data();
    return false;
  }
  url_ = url;
  std::string tmp_url;

  if (aAdvertiseUrl.size() > 0) {
    advertise_url_.resize(aAdvertiseUrl.size());
    advertise_url_.assign(aAdvertiseUrl.begin(), aAdvertiseUrl.end());
    tmp_url = aAdvertiseUrl;
  } else {
    advertise_url_.resize(url_.size());
    advertise_url_.assign(url_.begin(), url_.end());
    tmp_url = url_;
  }

  size_t index = url.find(URL_PROTOCOL_IP_SEPARATOR);
  if (index != std::string::npos) {
    url_ = url.substr(index + sizeof(URL_PROTOCOL_IP_SEPARATOR) - 1);
  }

  index = tmp_url.find(URL_PROTOCOL_IP_SEPARATOR);
  if (index != std::string::npos) {
    tmp_url = tmp_url.substr(index + sizeof(URL_PROTOCOL_IP_SEPARATOR) - 1);
    advertise_url_.resize(tmp_url.size());
    advertise_url_.assign(tmp_url.begin(), tmp_url.end());
  }

  // Register read event callback for server socket
  int retval = recv_event_loop_->SetEventHandler(server_fd_, EPOLLIN | EPOLLHUP | EPOLLERR, OnAccept,
                                                 reinterpret_cast<void *>(this));
  if (retval != RPC_OK) {
    MS_LOG(ERROR) << "Failed to add server event, url: " << url.c_str()
                  << ", advertise_url_: " << advertise_url_.data();
    return false;
  }
  MS_LOG(INFO) << "Start server succ, fd: " << server_fd_ << ", url: " << url.c_str()
               << ", advertise_url_ :" << advertise_url_.data();
  return true;
}

void TCPComm::ReadCallBack(void *context) {
  const int max_recv_count = 3;
  Connection *conn = reinterpret_cast<Connection *>(context);
  int count = 0;
  int retval = 0;
  do {
    retval = ReceiveMessage(conn);
    ++count;
  } while (retval > 0 && count < max_recv_count);

  return;
}

void TCPComm::EventCallBack(void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);

  if (conn->state == ConnectionState::kConnected) {
    Connection::conn_mutex.lock();
    DoSend(conn);
    Connection::conn_mutex.unlock();
  } else if (conn->state == ConnectionState::kDisconnecting) {
    Connection::conn_mutex.lock();
    output_buf_size_ -= conn->output_buffer_size;
    ConnectionPool::GetConnectionPool()->CloseConnection(conn);
    Connection::conn_mutex.unlock();
  }
}

void TCPComm::WriteCallBack(void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);
  if (conn->state == ConnectionState::kConnected) {
    Connection::conn_mutex.lock();
    DoSend(conn);
    Connection::conn_mutex.unlock();
  }
}

int TCPComm::ReceiveMessage(Connection *conn) {
  conn->CheckMessageType();
  switch (conn->recv_message_type) {
    case ParseType::kTcpMsg:
      return conn->ReceiveMessage(message_handler_);

#ifdef HTTP_ENABLED
    case ParseType::KHTTP_REQ:
      if (httpReqCb) {
        return httpReqCb(conn, message_handler_);
      } else {
        conn->state = ConnectionState::kDisconnecting;
        return -1;
      }

    case ParseType::KHTTP_RSP:
      if (httpRspCb) {
        return httpRspCb(conn, message_handler_);
      } else {
        conn->state = ConnectionState::kDisconnecting;
        return -1;
      }
#endif

    default:
      return 0;
  }
}

int TCPComm::SetConnectedHandler(Connection *conn) {
  /* add to epoll */
  return conn->recv_event_loop->SetEventHandler(conn->socket_fd,
                                                (uint32_t)(EPOLLOUT | EPOLLHUP | EPOLLRDHUP | EPOLLERR),
                                                ConnectedEventHandler, reinterpret_cast<void *>(conn));
}

int TCPComm::Connect(Connection *conn, const struct sockaddr *sa, socklen_t saLen) {
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

void TCPComm::Send(MessageBase *msg, const TCPComm *tcpmgr, bool remoteLink, bool isExactNotRemote) {
  std::lock_guard<std::mutex> lock(Connection::conn_mutex);
  Connection *conn = ConnectionPool::GetConnectionPool()->FindConnection(msg->to.Url(), remoteLink, isExactNotRemote);

  // Create a new connection if the connection to target of the message does not existed.
  if (conn == nullptr) {
    if (remoteLink && (!isExactNotRemote)) {
      MS_LOG(ERROR) << "Could not found remote link and send fail name: " << msg->name.c_str()
                    << ", from: " << advertise_url_.data() << ", to: " << msg->to.Url().c_str();
      delete msg;
      return;
    }
    conn = new (std::nothrow) Connection();
    if (conn == nullptr) {
      MS_LOG(ERROR) << "Failed to create new connection and send fail name: " << msg->name.c_str()
                    << ", from: " << advertise_url_.data() << ", to: " << msg->to.Url().c_str();
      delete msg;
      return;
    }
    conn->source = advertise_url_.data();
    conn->destination = msg->to.Url();
    conn->recv_event_loop = tcpmgr->recv_event_loop_;
    conn->send_event_loop = tcpmgr->send_event_loop_;
    conn->InitSocketOperation();

    int ret = DoConnect(msg->to.Url(), conn, TCPComm::EventCallBack, TCPComm::WriteCallBack, TCPComm::ReadCallBack);
    if (ret < 0) {
      MS_LOG(ERROR) << "Failed to do connect and send fail name: " << msg->name.c_str()
                    << ", from: " << advertise_url_.data() << ", to: " << msg->to.Url().c_str();
      if (conn->socket_operation != nullptr) {
        delete conn->socket_operation;
        conn->socket_operation = nullptr;
      }
      delete conn;
      delete msg;
      return;
    }
    ConnectionPool::GetConnectionPool()->AddConnection(conn);
  }

  if (!conn->is_remote && !isExactNotRemote && conn->priority == ConnectionPriority::kPriorityLow) {
    Connection *remoteConn = ConnectionPool::GetConnectionPool()->ExactFindConnection(msg->to.Url(), true);
    if (remoteConn != nullptr && remoteConn->state == ConnectionState::kConnected) {
      conn = remoteConn;
    }
  }

  // Prepare the message.
  if (conn->total_send_len == 0) {
    conn->FillSendMessage(msg, advertise_url_.data(), is_http_msg_);
  } else {
    (void)conn->send_message_queue.emplace(msg);
  }

  // Send the message.
  if (conn->state == ConnectionState::kConnected) {
    DoSend(conn);
  }
}

void TCPComm::SendByRecvLoop(MessageBase *msg, const TCPComm *tcpmgr, bool remoteLink, bool isExactNotRemote) {
  (void)recv_event_loop_->AddTask(
    [msg, tcpmgr, remoteLink, isExactNotRemote] { TCPComm::Send(msg, tcpmgr, remoteLink, isExactNotRemote); });
}

int TCPComm::Send(MessageBase *msg, bool remoteLink, bool isExactNotRemote) {
  return send_event_loop_->AddTask([msg, this, remoteLink, isExactNotRemote] {
    std::lock_guard<std::mutex> lock(Connection::conn_mutex);
    // Search connection by the target address
    bool exactNotRemote = is_http_msg_ || isExactNotRemote;
    Connection *conn = ConnectionPool::GetConnectionPool()->FindConnection(msg->to.Url(), remoteLink, exactNotRemote);
    if (conn == nullptr) {
      if (remoteLink && (!exactNotRemote)) {
        MS_LOG(ERROR) << "Can not found remote link and send fail name: " << msg->name.c_str()
                      << ", from: " << advertise_url_.data() << ", to: " << msg->to.Url().c_str();
        auto *ptr = msg;
        delete ptr;
        ptr = nullptr;

        return;
      }
      this->SendByRecvLoop(msg, this, remoteLink, exactNotRemote);
      return;
    }

    if (conn->state != kConnected && conn->send_message_queue.size() >= SENDMSG_QUEUELEN) {
      MS_LOG(WARNING) << "The name of dropped message is: " << msg->name.c_str() << ", fd: " << conn->socket_fd
                      << ", to: " << conn->destination.c_str() << ", remote: " << conn->is_remote;
      auto *ptr = msg;
      delete ptr;
      ptr = nullptr;

      return;
    }

    if (conn->state == ConnectionState::kClose || conn->state == ConnectionState::kDisconnecting) {
      this->SendByRecvLoop(msg, this, remoteLink, exactNotRemote);
      return;
    }

    if (!conn->is_remote && !exactNotRemote && conn->priority == ConnectionPriority::kPriorityLow) {
      Connection *remoteConn = ConnectionPool::GetConnectionPool()->ExactFindConnection(msg->to.Url(), true);
      if (remoteConn != nullptr && remoteConn->state == ConnectionState::kConnected) {
        conn = remoteConn;
      }
    }

    output_buf_size_ += msg->body.size();
    if (conn->total_send_len == 0) {
      conn->FillSendMessage(msg, advertise_url_.data(), is_http_msg_);
    } else {
      (void)conn->send_message_queue.emplace(msg);
    }

    if (conn->state == ConnectionState::kConnected) {
      DoSend(conn);
    }
  });
}

void TCPComm::CollectMetrics() {
  (void)send_event_loop_->AddTask([this] {
    Connection::conn_mutex.lock();
    Connection *maxConn = ConnectionPool::GetConnectionPool()->FindMaxConnection();
    Connection *fastConn = ConnectionPool::GetConnectionPool()->FindFastConnection();

    if (message_handler_ != nullptr) {
      IntTypeMetrics intMetrics;
      StringTypeMetrics stringMetrics;

      if (maxConn != nullptr) {
        intMetrics.push(maxConn->socket_fd);
        intMetrics.push(maxConn->error_code);
        intMetrics.push(maxConn->send_metrics->accum_msg_count);
        intMetrics.push(maxConn->send_metrics->max_msg_size);
        stringMetrics.push(maxConn->destination);
        stringMetrics.push(maxConn->send_metrics->last_succ_msg_name);
        stringMetrics.push(maxConn->send_metrics->last_fail_msg_name);
      }
      if (fastConn != nullptr && fastConn->IsSame(maxConn)) {
        intMetrics.push(fastConn->socket_fd);
        intMetrics.push(fastConn->error_code);
        intMetrics.push(fastConn->send_metrics->accum_msg_count);
        intMetrics.push(fastConn->send_metrics->max_msg_size);
        stringMetrics.push(fastConn->destination);
        stringMetrics.push(fastConn->send_metrics->last_succ_msg_name);
        stringMetrics.push(fastConn->send_metrics->last_fail_msg_name);
      }
    }

    ConnectionPool::GetConnectionPool()->ResetAllConnMetrics();
    Connection::conn_mutex.unlock();
  });
}

int TCPComm::Send(std::unique_ptr<MessageBase> &&msg, bool remoteLink, bool isExactNotRemote) {
  return Send(msg.release(), remoteLink, isExactNotRemote);
}

void TCPComm::Link(const AID &source, const AID &destination) {
  (void)recv_event_loop_->AddTask([source, destination, this] {
    std::string to = destination.Url();
    std::lock_guard<std::mutex> lock(Connection::conn_mutex);

    // Search connection by the target address
    Connection *conn = ConnectionPool::GetConnectionPool()->FindConnection(to, false, is_http_msg_);

    if (conn == nullptr) {
      MS_LOG(INFO) << "Can not found link source: " << std::string(source).c_str()
                   << ", destination: " << std::string(destination).c_str();
      conn = new (std::nothrow) Connection();
      if (conn == nullptr) {
        MS_LOG(ERROR) << "Failed to create new connection and link fail source: " << std::string(source).c_str()
                      << ", destination: " << std::string(destination).c_str();
        SendExitMsg(source, destination);
        return;
      }
      conn->source = advertise_url_.data();
      conn->destination = to;

      conn->recv_event_loop = this->recv_event_loop_;
      conn->send_event_loop = this->send_event_loop_;
      conn->InitSocketOperation();

      int ret = DoConnect(to, conn, TCPComm::EventCallBack, TCPComm::WriteCallBack, TCPComm::ReadCallBack);
      if (ret < 0) {
        MS_LOG(ERROR) << "Failed to do connect and link fail source: " << std::string(source).c_str()
                      << ", destination: " << std::string(destination).c_str();
        SendExitMsg(source, destination);
        if (conn->socket_operation != nullptr) {
          delete conn->socket_operation;
          conn->socket_operation = nullptr;
        }
        delete conn;
        return;
      }
      ConnectionPool::GetConnectionPool()->AddConnection(conn);
    }
    ConnectionPool::GetConnectionPool()->AddConnInfo(conn->socket_fd, source, destination, SendExitMsg);
    MS_LOG(INFO) << "Link fd: " << conn->socket_fd << ", source: " << std::string(source).c_str()
                 << ", destination: " << std::string(destination).c_str() << ", remote: " << conn->is_remote;
  });
}

void TCPComm::UnLink(const AID &destination) {
  (void)recv_event_loop_->AddTask([destination] {
    std::string to = destination.Url();
    std::lock_guard<std::mutex> lock(Connection::conn_mutex);
    if (is_http_msg_) {
      // When application has set 'LITERPC_HTTPKMSG_ENABLED',it means sending-link is in links map
      // while accepting-link is differently in remoteLinks map. So we only need to delete link in exact links.
      ConnectionPool::GetConnectionPool()->ExactDeleteConnection(to, false);
    } else {
      // When application hasn't set 'LITERPC_HTTPKMSG_ENABLED',it means sending-link and accepting-link is
      // shared
      // So we need to delete link in both links map and remote-links map.
      ConnectionPool::GetConnectionPool()->ExactDeleteConnection(to, false);
      ConnectionPool::GetConnectionPool()->ExactDeleteConnection(to, true);
    }
  });
}

void TCPComm::DoReConnectConn(Connection *conn, std::string to, const AID &source, const AID &destination, int *oldFd) {
  if (!is_http_msg_ && !conn->is_remote) {
    Connection *remoteConn = ConnectionPool::GetConnectionPool()->ExactFindConnection(to, true);
    // We will close remote link in rare cases where sending-link and accepting link coexists
    // simultaneously.
    if (remoteConn != nullptr) {
      MS_LOG(INFO) << "Reconnect, close remote connect fd :" << remoteConn->socket_fd
                   << ", source: " << std::string(source).c_str()
                   << ", destination: " << std::string(destination).c_str() << ", remote: " << remoteConn->is_remote
                   << ", state: " << remoteConn->state;
      ConnectionPool::GetConnectionPool()->CloseConnection(remoteConn);
    }
  }

  MS_LOG(INFO) << "Reconnect, close old connect fd: " << conn->socket_fd << ", source: " << std::string(source).c_str()
               << ", destination: " << std::string(destination).c_str() << ", remote: " << conn->is_remote
               << ", state: " << conn->state;

  *oldFd = conn->socket_fd;

  if (conn->recv_event_loop->DeleteEpollEvent(conn->socket_fd) == RPC_ERROR) {
    MS_LOG(ERROR) << "Failed to delete epoll event: " << conn->socket_fd;
  }
  conn->socket_operation->Close(conn);

  conn->socket_fd = -1;
  conn->recv_len = 0;

  conn->total_recv_len = 0;
  conn->recv_message_type = kUnknown;
  conn->state = kInit;
  if (conn->total_send_len != 0 && conn->send_message != nullptr) {
    delete conn->send_message;
  }
  conn->send_message = nullptr;
  conn->total_send_len = 0;

  if (conn->total_recv_len != 0 && conn->recv_message != nullptr) {
    delete conn->recv_message;
  }
  conn->recv_message = nullptr;
  conn->total_recv_len = 0;

  conn->recv_state = State::kMsgHeader;
}

Connection *TCPComm::CreateDefaultConn(std::string to) {
  Connection *conn = new (std::nothrow) Connection();
  if (conn == nullptr) {
    MS_LOG(ERROR) << "Failed to create new connection and reconnect fail to: " << to.c_str();
    return conn;
  }
  conn->source = advertise_url_.data();
  conn->destination = to;
  conn->recv_event_loop = this->recv_event_loop_;
  conn->send_event_loop = this->send_event_loop_;
  conn->InitSocketOperation();
  return conn;
}

void TCPComm::Reconnect(const AID &source, const AID &destination) {
  (void)send_event_loop_->AddTask([source, destination, this] {
    std::string to = destination.Url();
    std::lock_guard<std::mutex> lock(Connection::conn_mutex);
    Connection *conn = ConnectionPool::GetConnectionPool()->FindConnection(to, false, is_http_msg_);
    if (conn != nullptr) {
      conn->state = ConnectionState::kClose;
    }

    (void)recv_event_loop_->AddTask([source, destination, this] {
      std::string to = destination.Url();
      int oldFd = -1;
      std::lock_guard<std::mutex> lock(Connection::conn_mutex);
      Connection *conn = ConnectionPool::GetConnectionPool()->FindConnection(to, false, is_http_msg_);
      if (conn != nullptr) {
        // connection already exist
        DoReConnectConn(conn, to, source, destination, &oldFd);
      } else {
        // create default connection
        conn = CreateDefaultConn(to);
        if (conn == nullptr) {
          return;
        }
      }
      int ret = DoConnect(to, conn, TCPComm::EventCallBack, TCPComm::WriteCallBack, TCPComm::ReadCallBack);
      if (ret < 0) {
        if (conn->socket_operation != nullptr) {
          delete conn->socket_operation;
          conn->socket_operation = nullptr;
        }
        if (oldFd != -1) {
          conn->socket_fd = oldFd;
        }
        MS_LOG(ERROR) << "Failed to connect and reconnect fail source: " << std::string(source).c_str()
                      << ", destination: " << std::string(destination).c_str();
        ConnectionPool::GetConnectionPool()->CloseConnection(conn);
        return;
      }
      if (oldFd != -1) {
        if (!ConnectionPool::GetConnectionPool()->ReverseConnInfo(oldFd, conn->socket_fd)) {
          MS_LOG(ERROR) << "Failed to swap socket for " << oldFd << " and " << conn->socket_fd;
        }
      } else {
        ConnectionPool::GetConnectionPool()->AddConnection(conn);
      }
      ConnectionPool::GetConnectionPool()->AddConnInfo(conn->socket_fd, source, destination, SendExitMsg);
      MS_LOG(INFO) << "Reconnect fd: " << conn->socket_fd << ", source: " << std::string(source).c_str()
                   << ", destination: " << std::string(destination).c_str();
    });
  });
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
}

// This value is not used for the sub-class TCPComm.
uint64_t TCPComm::GetInBufSize() { return 1; }

uint64_t TCPComm::GetOutBufSize() { return output_buf_size_; }

bool TCPComm::IsHttpMsg() { return is_http_msg_; }
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
