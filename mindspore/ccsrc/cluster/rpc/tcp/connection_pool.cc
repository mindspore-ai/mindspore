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

#include "cluster/rpc/tcp/connection_pool.h"

namespace mindspore {
namespace distributed {
namespace rpc {
void ConnectionPool::SetLinkPattern(bool linkPattern) { double_link_ = linkPattern; }

void ConnectionPool::CloseConnection(Connection *conn) {
  if (conn == nullptr) {
    return;
  }

  // Trigger Exit message note that this should be called before erasing link. Because we may chang deleted flag
  // by to in this fun. And if deleted has been set to true, it means Exit message has been send before, do nothing.
  if (!conn->deleted) {
    DeleteConnInfo(conn);
  }
  conn->Close();
  delete conn;
  conn = nullptr;
}

Connection *ConnectionPool::FindConnection(const std::string &dst_url) {
  std::lock_guard<std::mutex> lock(mutex_);
  Connection *conn = nullptr;
  auto iter = connections_.find(dst_url);
  if (iter != connections_.end()) {
    conn = iter->second;
  }
  return conn;
}

void ConnectionPool::DeleteConnection(const std::string &dst_url) {
  Connection *conn = FindConnection(dst_url);
  if (conn != nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    (void)connections_.erase(dst_url);
    CloseConnection(conn);
  }
}

void ConnectionPool::DeleteAllConnections(std::map<std::string, Connection *> *links) const {
  if (links == nullptr) {
    return;
  }
  auto iter = links->begin();
  while (iter != links->end()) {
    Connection *conn = iter->second;
    if (conn == nullptr) {
      continue;
    }
    // erase link
    if (conn->recv_message != nullptr) {
      delete conn->recv_message;
    }
    iter = links->erase(iter);
    delete conn;
    conn = nullptr;
  }
}

void ConnectionPool::AddConnection(Connection *conn) {
  if (conn == nullptr) {
    MS_LOG(ERROR) << "The connection is null";
    return;
  }
  Connection *tmpConn = FindConnection(conn->destination);
  if (tmpConn != nullptr) {
    MS_LOG(INFO) << "unLink fd:" << tmpConn->socket_fd << ",to:" << tmpConn->destination.c_str();
    CloseConnection(tmpConn);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  (void)connections_.emplace(conn->destination, conn);
}

void ConnectionPool::DeleteConnInfo(int fd) {
  auto iter = conn_infos_.find(fd);
  if (iter == conn_infos_.end()) {
    return;
  }
  auto conn_infos = iter->second;
  auto iter2 = conn_infos.begin();

  while (iter2 != conn_infos.end()) {
    auto linkInfo = *iter2;
    if (linkInfo == nullptr) {
      continue;
    }
    if (linkInfo->delete_callback) {
      linkInfo->delete_callback(linkInfo->to, linkInfo->from);
    }
    iter2 = conn_infos.erase(iter2);
    delete linkInfo;
  }
  (void)conn_infos_.erase(fd);
}

void ConnectionPool::DeleteConnInfo(Connection *conn) {
  if (conn == nullptr) {
    return;
  }
  int fd = conn->socket_fd;
  // If run in double link pattern, link fd and send fd must be the same, send Exit message bind on this fd
  if (double_link_) {
    DeleteConnInfo(fd);
    return;
  }

  // If run in single link pattern, link fd and send fd may not be the same, we should send Exit message bind
  // on link fd and remote link fd. Here 'deleted' flag should be set true to avoid duplicate Exit message with
  // same aid.
  conn->deleted = true;
  DeleteConnInfo(conn->socket_fd);

  if (conn->socket_fd != fd) {
    MS_LOG(INFO) << "delete linker bind on link fd:" << conn->socket_fd << ",delete fd:" << fd;
  }
}

void ConnectionPool::DeleteAllConnInfos() {
  auto iter = conn_infos_.begin();
  while (iter != conn_infos_.end()) {
    auto conn_infos = iter->second;
    auto iter2 = conn_infos.begin();

    while (iter2 != conn_infos.end()) {
      auto linkInfo = *iter2;
      iter2 = conn_infos.erase(iter2);
      delete linkInfo;
    }
    iter = conn_infos_.erase(iter);
  }
}

ConnectionInfo *ConnectionPool::FindConnInfo(int fd, const std::string &dst_url) {
  auto iter = conn_infos_.find(fd);
  if (iter == conn_infos_.end()) {
    return nullptr;
  }
  auto conn_infos = iter->second;
  auto iter2 = conn_infos.begin();

  while (iter2 != conn_infos.end()) {
    auto linkInfo = *iter2;
    if (linkInfo == nullptr) {
      continue;
    }
    if (linkInfo->to == dst_url) {
      return linkInfo;
    }
    ++iter2;
  }
  return nullptr;
}

void ConnectionPool::AddConnInfo(int fd, const std::string &dst_url, DeleteCallBack callback) {
  ConnectionInfo *linker = FindConnInfo(fd, dst_url);
  if (linker != nullptr) {
    return;
  }
  // This linker will be deleted in `DeleteConnInfo` or `DeleteAllConnInfos`.
  linker = new (std::nothrow) ConnectionInfo();
  if (linker == nullptr) {
    MS_LOG(ERROR) << "new ConnectionInfo fail dAid:" << dst_url;
    return;
  }
  linker->from = "";
  linker->to = dst_url;
  linker->socket_fd = fd;
  linker->delete_callback = callback;
  (void)conn_infos_[fd].insert(linker);
}

void ConnectionPool::Finalize() {
  DeleteAllConnections(&local_conns_);
  DeleteAllConnections(&remote_conns_);
  DeleteAllConnInfos();
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
