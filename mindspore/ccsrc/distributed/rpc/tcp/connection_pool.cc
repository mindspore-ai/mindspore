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

#include <mutex>
#include "distributed/rpc/tcp/connection_pool.h"

namespace mindspore {
namespace distributed {
namespace rpc {
ConnectionPool *ConnectionPool::conn_pool = new ConnectionPool();

void ConnectionPool::SetLinkPattern(bool linkPattern) { double_link_ = linkPattern; }

void ConnectionPool::CloseConnection(Connection *conn) {
  if (conn == nullptr) {
    return;
  }

  // Trigger Exit message note that this should be called before erasing link. Because we may chang deleted flag
  // by to in this fun. And if deleted has been set to true, it means Exit message has been send before, do nothing.
  if (!conn->deleted) {
    DeleteConnInfo(conn->destination, conn->socket_fd);
  }

  if (!conn->destination.empty()) {
    if (conn->is_remote) {
      (void)remote_conns_.erase(conn->destination);
    } else {
      (void)local_conns_.erase(conn->destination);
    }
  }
  conn->Close();
  delete conn;
  conn = nullptr;
}

Connection *ConnectionPool::FindConnection(const std::string &to, bool remoteLink) {
  Connection *conn = nullptr;
  if (!remoteLink) {
    auto iter = local_conns_.find(to);
    if (iter != local_conns_.end()) {
      conn = iter->second;
      return conn;
    }
  }
  auto iter = remote_conns_.find(to);
  if (iter != remote_conns_.end()) {
    conn = iter->second;
  }
  return conn;
}

Connection *ConnectionPool::ExactFindConnection(const std::string &to, bool remoteLink) {
  Connection *conn = nullptr;
  if (!remoteLink) {
    auto iter = local_conns_.find(to);
    if (iter != local_conns_.end()) {
      conn = iter->second;
    }
  } else {
    auto iter = remote_conns_.find(to);
    if (iter != remote_conns_.end()) {
      conn = iter->second;
    }
  }
  return conn;
}

Connection *ConnectionPool::FindConnection(const std::string &to, bool remoteLink, bool exactNotRemote) {
  if (exactNotRemote) {
    return ExactFindConnection(to, false);
  } else {
    return FindConnection(to, remoteLink);
  }
}

void ConnectionPool::ResetAllConnMetrics() {
  for (const auto &iter : local_conns_) {
    iter.second->send_metrics->Reset();
  }
  for (const auto &iter : remote_conns_) {
    iter.second->send_metrics->Reset();
  }
}

Connection *ConnectionPool::FindMaxConnection() {
  Connection *conn = nullptr;
  size_t count = 0;
  for (const auto &iter : local_conns_) {
    if (iter.second->send_metrics->accum_msg_count > count) {
      count = iter.second->send_metrics->accum_msg_count;
      conn = iter.second;
    }
  }
  for (const auto &iter : remote_conns_) {
    if (iter.second->send_metrics->accum_msg_count > count) {
      count = iter.second->send_metrics->accum_msg_count;
      conn = iter.second;
    }
  }
  return conn;
}

Connection *ConnectionPool::FindFastConnection() {
  Connection *conn = nullptr;
  size_t size = 0;
  for (const auto &iter : local_conns_) {
    if (iter.second->send_metrics->max_msg_size > size) {
      size = iter.second->send_metrics->max_msg_size;
      conn = iter.second;
    }
  }
  for (const auto &iter : remote_conns_) {
    if (iter.second->send_metrics->max_msg_size > size) {
      size = iter.second->send_metrics->max_msg_size;
      conn = iter.second;
    }
  }
  return conn;
}

void ConnectionPool::ExactDeleteConnection(const std::string &to, bool remoteLink) {
  Connection *conn = ExactFindConnection(to, remoteLink);
  if (conn != nullptr) {
    MS_LOG(INFO) << "unLink fd:" << conn->socket_fd << ",to:" << to.c_str() << ",remote:" << remoteLink;
    CloseConnection(conn);
  }
}

void ConnectionPool::DeleteAllConnections(std::map<std::string, Connection *> *links) {
  auto iter = links->begin();
  while (iter != links->end()) {
    Connection *conn = iter->second;
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
    return;
  }
  Connection *tmpConn = ExactFindConnection(conn->destination, conn->is_remote);
  if (tmpConn != nullptr && tmpConn->is_remote == conn->is_remote) {
    MS_LOG(INFO) << "unLink fd:" << tmpConn->socket_fd << ",to:" << tmpConn->destination.c_str();
    CloseConnection(tmpConn);
  }

  if (conn->is_remote) {
    (void)remote_conns_.emplace(conn->destination, conn);
  } else {
    (void)local_conns_.emplace(conn->destination, conn);
  }
}

void ConnectionPool::SetConnPriority(const std::string &to, bool remoteLink, ConnectionPriority pri) {
  Connection *conn = ExactFindConnection(to, remoteLink);
  if (conn != nullptr && conn->is_remote == remoteLink) {
    conn->priority = pri;
  }
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
    if (linkInfo->delete_callback) {
      linkInfo->delete_callback(linkInfo->to, linkInfo->from);
    }
    iter2 = conn_infos.erase(iter2);
    delete linkInfo;
  }
  (void)conn_infos_.erase(fd);
}

void ConnectionPool::DeleteConnInfo(const std::string &to, int fd) {
  // If run in double link pattern, link fd and send fd must be the same, send Exit message bind on this fd
  if (double_link_) {
    DeleteConnInfo(fd);
    return;
  }

  // If run in single link pattern, link fd and send fd may not be the same, we should send Exit message bind
  // on link fd and remote link fd. Here 'deleted' flag should be set true to avoid duplicate Exit message with
  // same aid.
  Connection *nonRemoteConn = ConnectionPool::ExactFindConnection(to, false);
  if (nonRemoteConn != nullptr) {
    nonRemoteConn->deleted = true;
    DeleteConnInfo(nonRemoteConn->socket_fd);

    if (nonRemoteConn->socket_fd != fd) {
      MS_LOG(INFO) << "delete linker bind on link fd:" << nonRemoteConn->socket_fd << ",delete fd:" << fd;
    }
  }

  Connection *remoteConn = ConnectionPool::ExactFindConnection(to, true);
  if (remoteConn != nullptr) {
    remoteConn->deleted = true;
    DeleteConnInfo(remoteConn->socket_fd);

    if (remoteConn->socket_fd != fd) {
      MS_LOG(INFO) << "delete linker bind on remote link fd:" << remoteConn->socket_fd << ",delete fd:" << fd;
    }
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

ConnectionInfo *ConnectionPool::FindConnInfo(int fd, const AID &sAid, const AID &dAid) {
  auto iter = conn_infos_.find(fd);
  if (iter == conn_infos_.end()) {
    return nullptr;
  }
  auto conn_infos = iter->second;
  auto iter2 = conn_infos.begin();

  while (iter2 != conn_infos.end()) {
    auto linkInfo = *iter2;
    if (AID(linkInfo->from) == sAid && AID(linkInfo->to) == dAid) {
      return linkInfo;
    }
    ++iter2;
  }
  return nullptr;
}

void ConnectionPool::AddConnInfo(int fd, const AID &sAid, const AID &dAid, DeleteCallBack callback) {
  ConnectionInfo *linker = FindConnInfo(fd, sAid, dAid);
  if (linker != nullptr) {
    return;
  }
  linker = new (std::nothrow) ConnectionInfo();
  if (linker == nullptr) {
    MS_LOG(ERROR) << "new ConnectionInfo fail sAid:" << std::string(sAid).c_str()
                  << ",dAid:" << std::string(dAid).c_str();
    return;
  }
  linker->from = sAid;
  linker->to = dAid;
  linker->socket_fd = fd;
  linker->delete_callback = callback;
  (void)conn_infos_[fd].insert(linker);
}

bool ConnectionPool::ReverseConnInfo(int fromFd, int toFd) {
  auto iter = conn_infos_.find(fromFd);
  if (iter == conn_infos_.end()) {
    return false;
  }
  auto conn_infos = iter->second;
  (void)conn_infos_.erase(fromFd);
  conn_infos_[toFd] = conn_infos;
  return true;
}

ConnectionPool::~ConnectionPool() {
  try {
    DeleteAllConnections(&local_conns_);
    DeleteAllConnections(&remote_conns_);
    DeleteAllConnInfos();
  } catch (...) {
    MS_LOG(ERROR) << "Failed to release resource for connection pool.";
  }
}

ConnectionPool *ConnectionPool::GetConnectionPool() { return ConnectionPool::conn_pool; }
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
