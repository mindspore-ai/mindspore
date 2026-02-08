/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <chrono>
#include "plugin/ascend/res_manager/collective/leaper_trans.h"

constexpr size_t kFlowNum = 4;
constexpr size_t kSleepTime = 100;
constexpr size_t kMaxConnectTime = 3000;

namespace mindspore {
namespace device {
namespace ascend {

LeaperTrans &LeaperTrans::GetInstance() {
  static LeaperTrans instance;
  return instance;
}

LeaperTrans::LeaperTrans() {}

LeaperConnInfo LeaperTrans::Connect(std::string src_ip, std::string dst_ip, uint16_t src_port, uint16_t dst_port) {
  MS_LOG(WARNING) << "LeaperTrans try to connect to " << dst_ip << ":" << dst_port << ", src_port = " << src_port;
  LeaperConnInfo conn_info;
  struct sockaddr_in serverAddr;
  struct sockaddr_in clientAddr;
  struct sockaddr_in dstAddr;

  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    perror("socket");
    return conn_info;
  }

  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(src_port);
  serverAddr.sin_addr.s_addr = inet_addr(src_ip.c_str());

  int opt = 1;
  if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    perror("setsockopt");
    close(listen_fd);
    return conn_info;
  }

  if (bind(listen_fd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
    perror("bind");
    return conn_info;
  }

  if (listen(listen_fd, kFlowNum) < 0) {
    perror("listen");
    return conn_info;
  }

  for (uint32_t i = 0; i < kFlowNum; i++) {
    int client_fd = 0;
    if ((client_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      perror("socket");
      ClearConnInfo(&conn_info);
      close(listen_fd);
      return conn_info;
    }

    dstAddr.sin_family = AF_INET;
    dstAddr.sin_port = htons(dst_port);
    dstAddr.sin_addr.s_addr = inet_addr(dst_ip.c_str());
    int err = 0;
    for (size_t j = 0; j < kMaxConnectTime; j++) {
      err = connect(client_fd, (struct sockaddr *)&dstAddr, sizeof(dstAddr));
      if (err >= 0) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kSleepTime));
    }
    if (err < 0) {
      perror("connect");
      ClearConnInfo(&conn_info);
      close(listen_fd);
      return conn_info;
    }
    conn_info.send_fds.push_back(client_fd);
  }

  int addr_len = sizeof(clientAddr);
  for (uint32_t i = 0; i < kFlowNum; i++) {
    int server_fd =
      accept(listen_fd, reinterpret_cast<struct sockaddr *>(&clientAddr), reinterpret_cast<socklen_t *>(&addr_len));
    if (server_fd < 0) {
      perror("accept");
      ClearConnInfo(&conn_info);
      close(listen_fd);
      return conn_info;
    }
    conn_info.recv_fds.push_back(server_fd);
  }

  return conn_info;
}

bool LeaperTrans::SendRecv(const void *send_data, void *recv_data, size_t send_size, size_t recv_size,
                           const LeaperConnInfo &conn_info) {
  std::lock_guard<std::mutex> lock(send_recv_lock_);
  size_t residual;
  size_t send_flows = conn_info.send_fds.size();
  size_t recv_flows = conn_info.recv_fds.size();
  std::vector<std::thread> send_threads(send_flows);
  std::vector<std::thread> recv_threads(recv_flows);
  if (send_data != nullptr) {
    const uint8_t *send_buff = static_cast<const uint8_t *>(send_data);
    MS_LOG(WARNING) << "leaper trans SendRecv, send fds:" << conn_info.send_fds;
    // create send threads
    std::vector<size_t> send_segment_sizes(send_flows, send_size / send_flows);
    residual = send_size % send_flows;
    for (size_t i = 0; i < residual; i++) {
      send_segment_sizes[i]++;
    }
    std::vector<size_t> send_segment_starts(send_flows, 0);
    for (size_t i = 1; i < send_flows; i++) {
      send_segment_starts[i] = send_segment_starts[i - 1] + send_segment_sizes[i - 1];
    }
    for (size_t i = 0; i < send_flows; i++) {
      send_threads[i] = std::thread(&LeaperTrans::SendData, this, conn_info.send_fds[i],
                                    &send_buff[send_segment_starts[i]], send_segment_sizes[i]);
    }
  }

  if (recv_data != nullptr) {
    uint8_t *recv_buff = static_cast<uint8_t *>(recv_data);
    MS_LOG(WARNING) << "leaper trans SendRecv, recv fds:" << conn_info.recv_fds;
    // create recv threads
    std::vector<size_t> recv_segment_sizes(recv_flows, recv_size / recv_flows);
    residual = recv_size % recv_flows;
    for (size_t i = 0; i < residual; i++) {
      recv_segment_sizes[i]++;
    }
    std::vector<size_t> recv_segment_starts(recv_flows, 0);
    for (size_t i = 1; i < recv_flows; i++) {
      recv_segment_starts[i] = recv_segment_starts[i - 1] + recv_segment_sizes[i - 1];
    }
    for (size_t i = 0; i < recv_flows; i++) {
      recv_threads[i] = std::thread(&LeaperTrans::RecvData, this, conn_info.recv_fds[i],
                                    &recv_buff[recv_segment_starts[i]], recv_segment_sizes[i]);
    }
  }

  // wait finish
  if (send_data != nullptr) {
    for (size_t i = 0; i < send_flows; i++) {
      send_threads[i].join();
    }
  }
  if (recv_data != nullptr) {
    for (size_t i = 0; i < recv_flows; i++) {
      recv_threads[i].join();
    }
  }
  return true;
}

void LeaperTrans::SendData(int fd, const uint8_t *data, size_t size) {
  if (size == 0) {
    return;
  }
  int err = ::send(fd, data, size, 0);
  if (err < 0) {
    perror("send");
  }
}

void LeaperTrans::RecvData(int fd, uint8_t *data, size_t size) {
  if (size == 0) {
    return;
  }
  int err = ::recv(fd, data, size, MSG_WAITALL);
  if (err < 0) {
    perror("recv");
  }
}

void LeaperTrans::ClearConnInfo(LeaperConnInfo *conn_info) {
  for (auto fd : conn_info->send_fds) {
    close(fd);
  }
  conn_info->send_fds.clear();

  for (auto fd : conn_info->recv_fds) {
    close(fd);
  }
  conn_info->recv_fds.clear();
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
