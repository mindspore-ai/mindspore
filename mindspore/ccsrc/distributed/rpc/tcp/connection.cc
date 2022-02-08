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

#include "distributed/rpc/tcp/connection.h"

#include <memory>
#include <utility>

#include "distributed/rpc/tcp/tcp_socket_operation.h"
#include "distributed/rpc/tcp/connection_pool.h"

namespace mindspore {
namespace distributed {
namespace rpc {
// Handle socket events like read/write.
void SocketEventHandler(int fd, uint32_t events, void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);

  if (fd != conn->socket_fd) {
    MS_LOG(ERROR) << "Failed to reuse connection, delete and close fd: " << fd << ", connfd: " << conn->socket_fd
                  << ", event: " << events;
    conn->recv_event_loop->DeleteEpollEvent(fd);
    conn->state = ConnectionState::kDisconnecting;
    if (conn->event_callback != nullptr) {
      conn->event_callback(conn);
    } else {
      MS_LOG(ERROR) << "No event_callback found for fd: " << fd << ", events: " << events;
    }
    return;
  }
  // Handle write event.
  if (events & EPOLLOUT) {
    (void)conn->recv_event_loop->UpdateEpollEvent(fd, EPOLLIN | EPOLLHUP | EPOLLERR);
    if (conn->write_callback != nullptr) {
      conn->write_callback(conn);
    }
  }
  // Handle read event.
  if (events & EPOLLIN) {
    if (conn->read_callback != nullptr) {
      conn->read_callback(conn);
    }
  }
  // Handle disconnect event.
  if (conn->state == ConnectionState::kDisconnecting ||
      (conn->recv_message_type != ParseType::kHttpReq && conn->recv_message_type != ParseType::kHttpRsp &&
       (events & (uint32_t)(EPOLLHUP | EPOLLRDHUP | EPOLLERR)))) {
    if (conn->recv_message_type == ParseType::kTcpMsg) {
      MS_LOG(INFO) << "Event value fd: " << fd << ", events: " << events << ", state: " << conn->state
                   << ", errcode: " << conn->error_code << ", errno: " << errno << ", to: " << conn->destination.c_str()
                   << ", type:" << conn->recv_message_type << ", remote: " << conn->is_remote;
    }
    conn->state = ConnectionState::kDisconnecting;
    if (conn->event_callback != nullptr) {
      conn->event_callback(conn);
    } else {
      MS_LOG(ERROR) << "No event_callback found for fd: " << fd << ", events: " << events;
    }
  }
}

// Handle new connect event.
void NewConnectEventHandler(int fd, uint32_t events, void *context) {
  int retval = 0;
  Connection *conn = reinterpret_cast<Connection *>(context);
  conn->socket_operation->NewConnEventHandler(fd, events, context);

  if (conn->state == ConnectionState::kDisconnecting) {
    conn->Disconnect(fd);
    return;
  } else if (conn->state != ConnectionState::kConnected) {
    // The handshake is not complete
    return;
  }

  retval = conn->recv_event_loop->DeleteEpollEvent(fd);
  if (retval) {
    MS_LOG(ERROR) << "Failed to remove epoll remove connect handler for fd: " << fd;
    return;
  }

  retval = conn->recv_event_loop->SetEventHandler(conn->socket_fd, EPOLLIN | EPOLLHUP | EPOLLRDHUP | EPOLLERR,
                                                  SocketEventHandler, reinterpret_cast<void *>(conn));
  if (retval != RPC_OK) {
    MS_LOG(ERROR) << "Failed to add socket event handler for fd: " << fd << ", events: " << events;
    conn->Disconnect(fd);
    return;
  }

  conn->write_callback(conn);
  SocketEventHandler(fd, events, context);
  return;
}

Connection::Connection()
    : socket_fd(-1),
      deleted(false),
      is_remote(false),
      type(kTcp),
      socket_operation(nullptr),
      state(kInit),
      send_event_loop(nullptr),
      recv_event_loop(nullptr),
      send_metrics(nullptr),
      send_message(nullptr),
      recv_message(nullptr),
      recv_state(kMsgHeader),
      total_recv_len(0),
      total_send_len(0),
      recv_len(0),
      event_callback(nullptr),
      succ_callback(nullptr),
      write_callback(nullptr),
      read_callback(nullptr),
      output_buffer_size(0),
      error_code(0) {
  // Initialize the recv kernel message structure.
  recv_kernel_msg.msg_control = nullptr;
  recv_kernel_msg.msg_controllen = 0;
  recv_kernel_msg.msg_flags = 0;
  recv_kernel_msg.msg_name = nullptr;
  recv_kernel_msg.msg_namelen = 0;
  recv_kernel_msg.msg_iov = recv_io_vec;
  recv_kernel_msg.msg_iovlen = RECV_MSG_IO_VEC_LEN;

  // Initialize the send message header.
  send_metrics = new SendMetrics();
  for (unsigned int i = 0; i < BUSMAGIC_LEN; i++) {
    if (i < sizeof(RPC_MAGICID) - 1) {
      send_msg_header.magic[i] = RPC_MAGICID[i];
    } else {
      send_msg_header.magic[i] = '\0';
    }
  }

  // Initialize the send kernel message structure.
  send_kernel_msg.msg_control = nullptr;
  send_kernel_msg.msg_controllen = 0;
  send_kernel_msg.msg_flags = 0;
  send_kernel_msg.msg_name = nullptr;
  send_kernel_msg.msg_namelen = 0;
  send_kernel_msg.msg_iov = send_io_vec;
  send_kernel_msg.msg_iovlen = SEND_MSG_IO_VEC_LEN;
}

int Connection::Initialize() {
  InitSocketOperation();
  return AddConnnectEventHandler();
}

void Connection::InitSocketOperation() {
  if (socket_operation != nullptr) {
    return;
  }
  socket_operation = new (std::nothrow) TCPSocketOperation();
  MS_EXCEPTION_IF_NULL(socket_operation);
}

bool Connection::ReconnectSourceSocket(int fd, uint32_t events, int *soError, uint32_t error) {
  int retval = 0;
  socklen_t len = sizeof(*soError);

  retval = recv_event_loop->DeleteEpollEvent(fd);
  if (retval) {
    MS_LOG(ERROR) << "Failed to delete event for fd: " << fd << ", event: " << events;
    return false;
  }

  retval = getsockopt(fd, SOL_SOCKET, SO_ERROR, soError, &len);
  if (retval) {
    *soError = errno;
  }
  if (*soError || error) {
    return false;
  }
  retval = recv_event_loop->SetEventHandler(socket_fd, EPOLLIN | EPOLLHUP | EPOLLRDHUP | EPOLLERR, SocketEventHandler,
                                            reinterpret_cast<void *>(this));
  if (retval != RPC_OK) {
    MS_LOG(ERROR) << "Failed to add socket event handler for fd: " << fd << ", events: " << events;
    return false;
  }
  return true;
}

void Connection::Disconnect(int fd) {
  if (LOG_CHECK_EVERY_N()) {
    MS_LOG(INFO) << "New connection fail fd: " << fd << ", state: " << state << ", errno: " << errno
                 << ", to: " << destination.c_str() << ", type: " << recv_message_type;
  }
  state = ConnectionState::kDisconnecting;
  event_callback(this);
  return;
}

void Connection::Close() {
  if (recv_event_loop != nullptr) {
    if (recv_event_loop->DeleteEpollEvent(socket_fd) == RPC_ERROR) {
      MS_LOG(ERROR) << "Failed to delete epoll event " << socket_fd;
    }
  }
  if (!destination.empty()) {
    if (recv_message != nullptr) {
      delete recv_message;
      recv_message = nullptr;
    }
  }

  if (total_send_len != 0 && send_message != nullptr) {
    delete send_message;
    send_message = nullptr;
  }

  MessageBase *tmpMsg = nullptr;
  while (!send_message_queue.empty()) {
    tmpMsg = send_message_queue.front();
    send_message_queue.pop();
    delete tmpMsg;
    tmpMsg = nullptr;
  }

  if (socket_operation != nullptr) {
    socket_operation->Close(this);
    delete socket_operation;
    socket_operation = nullptr;
  }

  if (send_metrics != nullptr) {
    delete send_metrics;
    send_metrics = nullptr;
  }
}

int Connection::ReceiveMessage() {
  bool ok = ParseMessage();
  // If no message parsed, wait for next read
  if (!ok) {
    if (state == ConnectionState::kDisconnecting) {
      return -1;
    }
    return 0;
  }

  std::unique_ptr<MessageBase> msg(recv_message);
  recv_message = nullptr;

  // Call msg handler if set
  if (message_handler != nullptr) {
    message_handler(std::move(msg));
  } else {
    MS_LOG(INFO) << "Message handler was not found";
  }
  return 1;
}

void Connection::CheckMessageType() {
  if (recv_message_type != ParseType::kUnknown) {
    return;
  }

  std::string magic_id = "";
  magic_id.resize(sizeof(RPC_MAGICID) - 1);
  char *buf = const_cast<char *>(magic_id.data());

  int size = socket_operation->ReceivePeek(this, buf, sizeof(RPC_MAGICID) - 1);
  if (size < static_cast<int>(sizeof(RPC_MAGICID) - 1)) {
    if (size == 0) {
      MS_LOG(INFO) << "Set connection disconnecting for fd: " << socket_fd << ", size: " << size
                   << ", magic size: " << static_cast<int>(sizeof(RPC_MAGICID) - 1) << ", errno: " << errno;
      state = ConnectionState::kDisconnecting;
    }
    return;
  }
  if (strncmp(RPC_MAGICID, magic_id.c_str(), sizeof(RPC_MAGICID) - 1) == 0) {
    recv_state = State::kMsgHeader;
    recv_message_type = ParseType::kTcpMsg;
  }
  return;
}

std::string Connection::GenerateHttpMessage(MessageBase *msg) {
  static const std::string postLineBegin = std::string() + "POST /";
  static const std::string postLineEnd = std::string() + " HTTP/1.1\r\n";
  static const std::string userAgentLineBegin = std::string() + "User-Agent: libprocess/";
  static const std::string fromLineBegin = std::string() + "Libprocess-From: ";
  static const std::string connectLine = std::string() + "Connection: Keep-Alive\r\n";
  static const std::string hostLine = std::string() + "Host: \r\n";
  static const std::string chunkedBeginLine = std::string() + "Transfer-Encoding: chunked\r\n\r\n";
  static const std::string chunkedEndLine = std::string() + "\r\n" + "0\r\n" + "\r\n";
  static const std::string commonEndLine = std::string() + "\r\n";

  std::string postLine;
  if (msg->To().Name() != "") {
    postLine = postLineBegin + msg->To().Name() + "/" + msg->Name() + postLineEnd;
  } else {
    postLine = postLineBegin + msg->Name() + postLineEnd;
  }

  std::string userAgentLine = userAgentLineBegin + msg->From().Name() + "@" + advertise_addr_ + commonEndLine;
  std::string fromLine = fromLineBegin + msg->From().Name() + "@" + advertise_addr_ + commonEndLine;

  if (msg->Body().size() > 0) {
    std::ostringstream bodyLine;
    bodyLine << std::hex << msg->Body().size() << "\r\n";
    (void)bodyLine.write(msg->Body().data(), msg->Body().size());
    return postLine + userAgentLine + fromLine + connectLine + hostLine + chunkedBeginLine + bodyLine.str() +
           chunkedEndLine;
  }
  return postLine + userAgentLine + fromLine + connectLine + hostLine + commonEndLine;
}

void Connection::FillSendMessage(MessageBase *msg, const std::string &advertiseUrl, bool isHttpKmsg, int index) {
  index = 0;
  if (msg->type == MessageBase::Type::KMSG) {
    if (!isHttpKmsg) {
      send_to = msg->to;
      send_from = msg->from.Name() + "@" + advertiseUrl;

      send_msg_header.name_len = htonl(static_cast<uint32_t>(msg->name.size()));
      send_msg_header.to_len = htonl(static_cast<uint32_t>(send_to.size()));
      send_msg_header.from_len = htonl(static_cast<uint32_t>(send_from.size()));
      send_msg_header.body_len = htonl(static_cast<uint32_t>(msg->body.size()));

      send_io_vec[index].iov_base = &send_msg_header;
      send_io_vec[index].iov_len = sizeof(send_msg_header);
      ++index;
      send_io_vec[index].iov_base = const_cast<char *>(msg->name.data());
      send_io_vec[index].iov_len = msg->name.size();
      ++index;
      send_io_vec[index].iov_base = const_cast<char *>(send_to.data());
      send_io_vec[index].iov_len = send_to.size();
      ++index;
      send_io_vec[index].iov_base = const_cast<char *>(send_from.data());
      send_io_vec[index].iov_len = send_from.size();
      ++index;
      send_io_vec[index].iov_base = const_cast<char *>(msg->body.data());
      send_io_vec[index].iov_len = msg->body.size();
      ++index;
      send_kernel_msg.msg_iov = send_io_vec;
      send_kernel_msg.msg_iovlen = IntToSize(index);
      total_send_len =
        UlongToUint(sizeof(send_msg_header)) + msg->name.size() + send_to.size() + send_from.size() + msg->body.size();
      send_message = msg;

      // update metrics
      send_metrics->UpdateMax(msg->body.size());
      send_metrics->last_send_msg_name = msg->name;

      return;
    } else {
      if (advertise_addr_.empty()) {
        size_t index = advertiseUrl.find(URL_PROTOCOL_IP_SEPARATOR);
        if (index == std::string::npos) {
          advertise_addr_ = advertiseUrl;
        } else {
          advertise_addr_ = advertiseUrl.substr(index + sizeof(URL_PROTOCOL_IP_SEPARATOR) - 1);
        }
      }
      msg->body = GenerateHttpMessage(msg);
    }

    send_io_vec[index].iov_base = const_cast<char *>(msg->body.data());
    send_io_vec[index].iov_len = msg->body.size();
    ++index;
    send_kernel_msg.msg_iov = send_io_vec;
    send_kernel_msg.msg_iovlen = IntToSize(index);
    total_send_len = UlongToUint(msg->body.size());
    send_message = msg;

    // update metrics
    send_metrics->UpdateMax(msg->body.size());
    send_metrics->last_send_msg_name = msg->name;
  }
}

void Connection::FillRecvMessage() {
  size_t recvNameLen = static_cast<size_t>(recv_msg_header.name_len);
  size_t recvToLen = static_cast<size_t>(recv_msg_header.to_len);
  size_t recvFromLen = static_cast<size_t>(recv_msg_header.from_len);
  size_t recvBodyLen = static_cast<size_t>(recv_msg_header.body_len);
  if (recvNameLen > MAX_KMSG_NAME_LEN || recvToLen > MAX_KMSG_TO_LEN || recvFromLen > MAX_KMSG_FROM_LEN ||
      recvBodyLen > MAX_KMSG_BODY_LEN) {
    MS_LOG(ERROR) << "Drop invalid tcp data.";
    state = ConnectionState::kDisconnecting;
    return;
  }

  int i = 0;
  MessageBase *msg = new (std::nothrow) MessageBase();
  MS_EXCEPTION_IF_NULL(msg);

  msg->name.resize(recvNameLen);
  recv_to.resize(recvToLen);
  recv_from.resize(recvFromLen);
  msg->body.resize(recvBodyLen);

  recv_io_vec[i].iov_base = const_cast<char *>(msg->name.data());
  recv_io_vec[i].iov_len = msg->name.size();
  ++i;
  recv_io_vec[i].iov_base = const_cast<char *>(recv_to.data());
  recv_io_vec[i].iov_len = recv_to.size();
  ++i;
  recv_io_vec[i].iov_base = const_cast<char *>(recv_from.data());
  recv_io_vec[i].iov_len = recv_from.size();
  ++i;
  recv_io_vec[i].iov_base = const_cast<char *>(msg->body.data());
  recv_io_vec[i].iov_len = msg->body.size();
  ++i;

  recv_kernel_msg.msg_iov = recv_io_vec;
  recv_kernel_msg.msg_iovlen = IntToSize(i);
  total_recv_len = UlongToUint(msg->name.size()) + recv_to.size() + recv_from.size() + msg->body.size();
  recv_message = msg;
}

int Connection::AddConnnectEventHandler() {
  return recv_event_loop->SetEventHandler(socket_fd, EPOLLIN | EPOLLHUP | EPOLLERR, NewConnectEventHandler,
                                          reinterpret_cast<void *>(this));
}

bool Connection::ParseMessage() {
  std::string magic_id = "";
  int retval = 0;
  uint32_t recvLen = 0;
  char *recvBuf = nullptr;

  switch (recv_state) {
    // Parse message header.
    case State::kMsgHeader:
      recvBuf = reinterpret_cast<char *>(&recv_msg_header) + recv_len;
      retval = socket_operation->Receive(this, recvBuf, sizeof(MessageHeader) - recv_len, &recvLen);
      if (retval < 0) {
        state = ConnectionState::kDisconnecting;
        recv_len += recvLen;
        return false;
      }
      if ((recvLen + recv_len) != sizeof(MessageHeader)) {
        recv_len += recvLen;
        return false;
      }
      recv_len = 0;

      if (strncmp(recv_msg_header.magic, RPC_MAGICID, sizeof(RPC_MAGICID) - 1) != 0) {
        MS_LOG(ERROR) << "Failed to check magicid, RPC_MAGICID: " << RPC_MAGICID
                      << ", recv magic_id: " << magic_id.c_str();
        state = ConnectionState::kDisconnecting;
        return false;
      }
      ReorderHeader(&recv_msg_header);
      FillRecvMessage();
      if (state == ConnectionState::kDisconnecting) {
        return false;
      }
      recv_state = State::kBody;

    // Parse message body.
    case State::kBody:
      retval = socket_operation->ReceiveMessage(this, &recv_kernel_msg, total_recv_len);
      if (retval != static_cast<int>(total_recv_len)) {
        if (retval < 0) {
          state = ConnectionState::kDisconnecting;
          return false;
        }
        total_recv_len -= IntToSize(retval);
        return false;
      }
      recv_state = State::kMsgHeader;
      break;
    default:
      return false;
  }
  return true;
}

void Connection::ReorderHeader(MessageHeader *header) const {
  header->name_len = ntohl(header->name_len);
  header->to_len = ntohl(header->to_len);
  header->from_len = ntohl(header->from_len);
  header->body_len = ntohl(header->body_len);
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
