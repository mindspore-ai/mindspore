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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_CONNECTION_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_CONNECTION_H_

#include <queue>
#include <string>
#include <mutex>
#include <memory>

#include "actor/msg.h"
#include "distributed/rpc/tcp/constants.h"
#include "distributed/rpc/tcp/event_loop.h"
#include "distributed/rpc/tcp/socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
/*
 * The MessageHeader contains the stats info about the message body.
 */
struct MessageHeader {
  MessageHeader() {
    for (unsigned int i = 0; i < BUSMAGIC_LEN; ++i) {
      if (i < sizeof(RPC_MAGICID) - 1) {
        magic[i] = RPC_MAGICID[i];
      } else {
        magic[i] = '\0';
      }
    }
  }

  char magic[BUSMAGIC_LEN];
  uint32_t name_len{0};
  uint32_t to_len{0};
  uint32_t from_len{0};
  uint32_t body_len{0};
};

/*
 * The SendMetrics is responsible for collecting metrics when sending data through a connection.
 */
struct SendMetrics {
  // Records the message number and max body size.
  void UpdateMax(size_t size) {
    accum_msg_count++;
    if (size > max_msg_size) {
      max_msg_size = size;
    }
  }

  // Records the latest error message.
  void UpdateError(bool fail, int err = 0) {
    if (fail) {
      last_fail_msg_name = last_send_msg_name;
      error_code = err;
    } else {
      last_succ_msg_name = last_send_msg_name;
    }
  }

  // Reset all the metrics info.
  void Reset() {
    accum_msg_count = 0;
    max_msg_size = 0;
    error_code = 0;
    last_succ_msg_name = "";
    last_fail_msg_name = "";
    last_send_msg_name = "";
  }

  // The total number of bytes sent already.
  size_t accum_msg_count{0};

  // The max message body size sent in bytes.
  size_t max_msg_size{0};
  int error_code{0};

  std::string last_succ_msg_name;
  std::string last_fail_msg_name;
  std::string last_send_msg_name;
};

/*
 * Represents a TCP or SSL connection.
 */
struct Connection {
 public:
  Connection();
  ~Connection() = default;

  // Initialize the connection(eg. add some socket event handlers).
  int Initialize();

  // Create a new socket operation if needed.
  void InitSocketOperation();

  // Delete this socket fd(source client socket) and add back to the connection.
  bool ReconnectSourceSocket(int fd, uint32_t events, int *soError, uint32_t error);

  // Disconnect the socket fd from source.
  void Disconnect(int fd);

  // Close this connection.
  void Close();

  int ReceiveMessage();
  void CheckMessageType();

  // Fill the message to be sent based on the input message.
  void FillSendMessage(MessageBase *msg, const std::string &advertiseUrl, bool isHttpKmsg, int index = 0);

  void FillRecvMessage();

  bool IsSame(const Connection *that) {
    return !(that != nullptr && that->destination == destination && that->is_remote == is_remote);
  }

  // The socket used by this connection.
  int socket_fd;

  // Indicates whether this connection is deleted from link manager.
  bool deleted;

  // Indicates the priority of this connection.
  ConnectionPriority priority{ConnectionPriority::kPriorityHigh};

  // Indicates whether this connection is connected from remote client.
  // A connection is remote only when the connection is created by the `OnAccept` callback.
  bool is_remote;

  // TCP or SSL.
  ConnectionType type;

  // The socket address(ip:port) of client and server of this connection.
  std::string source;
  std::string destination;

  // Peer address.
  std::string peer;

  // Specific operations for the socket in this connection.
  SocketOperation *socket_operation;

  // The state of this connection(eg. kInit/kConnecting/..)
  ConnectionState state{kInit};

  // The threads for handling the receive and send requsets on this connection.
  EventLoop *send_event_loop;
  EventLoop *recv_event_loop;

  // Collects data sending metrics.
  SendMetrics *send_metrics;

  // The message data waiting to be sent and receive through this connection..
  MessageBase *send_message;
  MessageBase *recv_message;

  // Owned by the tcp_comm.
  std::shared_ptr<std::mutex> conn_mutex;

  State recv_state;

  // Total length of received and sent messages.
  uint32_t total_recv_len;
  uint32_t total_send_len;
  uint32_t recv_len;

  std::string send_to;
  std::string send_from;
  std::string recv_to;
  std::string recv_from;

  // Message header.
  MessageHeader send_msg_header;
  MessageHeader recv_msg_header;

  // The message structure of kernel.
  struct msghdr send_kernel_msg;
  struct msghdr recv_kernel_msg;

  struct iovec recv_io_vec[RECV_MSG_IO_VEC_LEN];
  struct iovec send_io_vec[SEND_MSG_IO_VEC_LEN];

  ParseType recv_message_type{kUnknown};

  // Callbacks for io events
  ConnectionCallBack event_callback;
  ConnectionCallBack succ_callback;
  ConnectionCallBack write_callback;
  ConnectionCallBack read_callback;

  // Function for handling received messages.
  MessageHandler message_handler;

  // Buffer for messages to be sent.
  std::queue<MessageBase *> send_message_queue;

  uint64_t output_buffer_size;

  // The error code when sending or receiving messages.
  int error_code;

 private:
  // Add handler for socket connect event.
  int AddConnnectEventHandler();

  // Parse message from socket recv buffer.
  bool ParseMessage();

  // Make a http message based on given input message.
  std::string GenerateHttpMessage(MessageBase *msg);

  // Change the header body from network byte order to host byte order.
  void ReorderHeader(MessageHeader *header) const;

  std::string advertise_addr_;
};
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
