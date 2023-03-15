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

#include "distributed/rpc/tcp/socket_operation.h"

#include <net/if.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <securec.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <system_error>

#include "actor/log.h"
#include "include/backend/distributed/rpc/tcp/constants.h"

namespace mindspore {
namespace distributed {
namespace rpc {
int SocketOperation::SetSocketKeepAlive(int fd, int keepalive, int keepidle, int keepinterval, int keepcount) {
  int option_val = 0;
  int ret = 0;

  option_val = keepalive;
  ret = setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &option_val, sizeof(option_val));
  if (ret < 0) {
    MS_LOG(ERROR) << "Failed to call setsockopt SO_KEEPALIVE, fd: " << fd << ", errno:" << errno;
    return -1;
  }

  // Send first probe after `interval' seconds.
  option_val = keepidle;
  ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &option_val, sizeof(option_val));
  if (ret < 0) {
    MS_LOG(ERROR) << "Failed to call setsockopt TCP_KEEPIDLE, fd: " << fd << ", errno:" << errno;
    return -1;
  }

  // Send next probes after the specified interval.
  option_val = keepinterval;
  ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &option_val, sizeof(option_val));
  if (ret < 0) {
    MS_LOG(ERROR) << "Failed to call setsockopt TCP_KEEPINTVL, fd: " << fd << ", errno:" << errno;
    return -1;
  }

  /* Consider the socket in error state after three we send three ACK
   * probes without getting a reply. */
  option_val = keepcount;
  ret = setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &option_val, sizeof(option_val));
  if (ret < 0) {
    MS_LOG(ERROR) << "Failed to call setsockopt TCP_KEEPCNT, fd: " << fd << ", errno:" << errno;
    return -1;
  }
  return 0;
}

int SocketOperation::SetSocketOptions(int sock_fd) {
  int option_val = 1;
  int ret = 0;

  ret = setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &option_val, sizeof(option_val));
  if (ret > 0) {
    MS_LOG(ERROR) << "Failed to call setsockopt SO_REUSEADDR, fd: " << sock_fd << ", errno:" << errno;
    return -1;
  }

  ret = setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &option_val, sizeof(option_val));
  if (ret > 0) {
    MS_LOG(ERROR) << "Failed to call setsockopt TCP_NODELAY, fd: " << sock_fd << ", errno:" << errno;
    return -1;
  }

  ret = SetSocketKeepAlive(sock_fd, SOCKET_KEEPALIVE, SOCKET_KEEPIDLE, SOCKET_KEEPINTERVAL, SOCKET_KEEPCOUNT);
  if (ret > 0) {
    MS_LOG(WARNING) << "Failed to call setsockopt keep alive, fd: " << sock_fd;
  }
  return 0;
}

int SocketOperation::CreateSocket(sa_family_t family) {
  int ret = 0;
  int fd = 0;

  // Create server socket
  fd = ::socket(family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    MS_LOG(WARNING) << "Failed to create socket: " << errno;
    return -1;
  }

  ret = SetSocketOptions(fd);
  if (ret < 0) {
    if (close(fd) != 0) {
      MS_LOG(EXCEPTION) << "Failed to close fd: " << fd;
    }
    return -1;
  }
  return fd;
}

std::string SocketOperation::GetLocalIP() {
  // Lookup all the network interfaces on the local machine.
  struct ifaddrs *if_addrs;
  if (getifaddrs(&if_addrs) != 0) {
    MS_LOG(ERROR) << "Failed to lookup local network interfaces.";
    freeifaddrs(if_addrs);
    return "";
  }
  // Find the first physical network interface.
  struct ifaddrs *if_addr = if_addrs;
  MS_EXCEPTION_IF_NULL(if_addr);
  while (if_addr != nullptr) {
    if (if_addr->ifa_addr == nullptr) {
      continue;
    }

    if (if_addr->ifa_addr->sa_family == AF_INET && !(if_addr->ifa_flags & IFF_LOOPBACK)) {
      auto sock_addr = reinterpret_cast<struct sockaddr_in *>(if_addr->ifa_addr);
      MS_EXCEPTION_IF_NULL(sock_addr);

      auto ip_addr = inet_ntoa(sock_addr->sin_addr);
      MS_EXCEPTION_IF_NULL(ip_addr);

      std::string ip(ip_addr, ip_addr + strlen(ip_addr));
      freeifaddrs(if_addrs);
      return ip;
    } else {
      if_addr = if_addr->ifa_next;
    }
  }
  freeifaddrs(if_addrs);
  return "";
}

std::string SocketOperation::GetIP(const std::string &url) {
  size_t index1 = url.find("[");
  if (index1 == std::string::npos) {
    index1 = url.find(URL_PROTOCOL_IP_SEPARATOR);
    if (index1 == std::string::npos) {
      index1 = 0;
    } else {
      index1 = index1 + sizeof(URL_PROTOCOL_IP_SEPARATOR) - 1;
    }
  } else {
    index1 = index1 + 1;
  }

  size_t index2 = url.find("]");
  if (index2 == std::string::npos) {
    index2 = url.rfind(URL_IP_PORT_SEPARATOR);
    if (index2 == std::string::npos) {
      MS_LOG(INFO) << "Couldn't find the character: " << URL_IP_PORT_SEPARATOR << ", url: " << url.c_str();
      return "";
    }
  }

  if (index1 > index2) {
    MS_LOG(INFO) << "Parse ip failed, url: " << url.c_str();
    return "";
  }

  if (index2 >= url.size()) {
    MS_LOG(ERROR) << "Invalid url: " << url;
    return "";
  } else {
    std::string ip = url.substr(index1, index2 - index1);
    SocketAddress addr;

    int result = inet_pton(AF_INET, ip.c_str(), &addr.saIn.sin_addr);
    if (result <= 0) {
      result = inet_pton(AF_INET6, ip.c_str(), &addr.saIn6.sin6_addr);
      if (result <= 0) {
        MS_LOG(INFO) << "Parse ip failed, result: " << result << ", url:" << url.c_str();
        return "";
      }
    }
    return ip;
  }
}

bool SocketOperation::GetSockAddr(const std::string &url, SocketAddress *addr) {
  if (addr == nullptr) {
    return false;
  }
  std::string ip;
  uint16_t port = 0;

  size_t len = sizeof(*addr);
  if (memset_s(addr, len, 0, len) != EOK) {
    MS_LOG(ERROR) << "Failed to call memset_s.";
    return false;
  }

  size_t index1 = url.find(URL_PROTOCOL_IP_SEPARATOR);
  if (index1 == std::string::npos) {
    index1 = 0;
  } else {
    index1 = index1 + sizeof(URL_PROTOCOL_IP_SEPARATOR) - 1;
  }

  size_t index2 = url.rfind(':');
  if (index2 == std::string::npos) {
    MS_LOG(ERROR) << "Couldn't find the character colon.";
    return false;
  }

  ip = url.substr(index1, index2 - index1);
  if (ip.empty()) {
    MS_LOG(ERROR) << "Couldn't find ip in url: " << url.c_str();
    return false;
  }

  size_t idx = index2 + sizeof(URL_IP_PORT_SEPARATOR) - 1;
  if (idx >= url.size()) {
    MS_LOG(ERROR) << "The size of url is invalid";
    return false;
  }
  try {
    port = static_cast<uint16_t>(std::stoul(url.substr(idx)));
  } catch (const std::system_error &e) {
    MS_LOG(ERROR) << "Couldn't find port in url: " << url.c_str();
    return false;
  }

  int result = inet_pton(AF_INET, ip.c_str(), &addr->saIn.sin_addr);
  if (result > 0) {
    addr->saIn.sin_family = AF_INET;
    addr->saIn.sin_port = htons(port);
    return true;
  }

  result = inet_pton(AF_INET6, ip.c_str(), &(addr->saIn6.sin6_addr));
  if (result > 0) {
    addr->saIn6.sin6_family = AF_INET6;
    addr->saIn6.sin6_port = htons(port);
    return true;
  }

  MS_LOG(ERROR) << "Parse ip failed, result: " << result << ", url: " << url.c_str();
  return false;
}

uint16_t SocketOperation::GetPort(int fd) {
  uint16_t port = 0;
  int retval = 0;
  union SocketAddress isa;
  socklen_t isaLen = sizeof(struct sockaddr_storage);

  retval = getsockname(fd, &isa.sa, &isaLen);
  if (retval > 0) {
    MS_LOG(INFO) << "Failed to call getsockname, fd: " << fd << ", ret: " << retval << ", errno: " << errno;
    return port;
  }

  if (isa.sa.sa_family == AF_INET) {
    port = ntohs(isa.saIn.sin_port);
  } else if (isa.sa.sa_family == AF_INET6) {
    port = ntohs(isa.saIn6.sin6_port);
  } else {
    MS_LOG(INFO) << "Unknown fd: " << fd << ", family: " << isa.sa.sa_family;
  }
  return port;
}

std::string SocketOperation::GetPeer(int sock_fd) {
  std::string peer;
  int retval = 0;
  union SocketAddress isa;
  socklen_t isaLen = sizeof(struct sockaddr_storage);

  retval = getpeername(sock_fd, &isa.sa, &isaLen);
  if (retval < 0) {
    MS_LOG(INFO) << "Failed to call getpeername, fd: " << sock_fd << ", ret: " << retval << ", errno: " << errno;
    return peer;
  }

  char ipdotdec[IP_LEN_MAX];
  if (isa.sa.sa_family == AF_INET) {
    if (inet_ntop(AF_INET, reinterpret_cast<void *>(&isa.saIn.sin_addr), ipdotdec, IP_LEN_MAX) == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to call inet_ntop kernel func.";
    }
    peer = std::string(ipdotdec) + ":" + std::to_string(ntohs(isa.saIn.sin_port));
  } else if (isa.sa.sa_family == AF_INET6) {
    if (inet_ntop(AF_INET6, reinterpret_cast<void *>(&isa.saIn6.sin6_addr), ipdotdec, IP_LEN_MAX) == nullptr) {
      MS_LOG(ERROR) << "Failed to call inet_ntop.";
    }
    peer = std::string(ipdotdec) + ":" + std::to_string(ntohs(isa.saIn6.sin6_port));
  } else {
    MS_LOG(INFO) << "Unknown fd: " << sock_fd << ", family: " << isa.sa.sa_family;
  }
  return peer;
}

int SocketOperation::Connect(int sock_fd, const struct sockaddr *sa, socklen_t saLen, uint16_t *boundPort) {
  if (sa == nullptr || boundPort == nullptr) {
    return RPC_ERROR;
  }
  int retval = 0;

  retval = connect(sock_fd, sa, saLen);
  if (retval != 0) {
    if (errno == EINPROGRESS) {
      /* set iomux for write event */
    } else {
      MS_LOG(ERROR) << "Failed to call connect, fd: " << sock_fd << ", ret: " << retval << ", errno: " << errno;
      return retval;
    }
  }

  // to get local port
  *boundPort = GetPort(sock_fd);
  if (*boundPort == 0) {
    return RPC_ERROR;
  }
  return RPC_OK;
}

int SocketOperation::Listen(const std::string &url) {
  int listenFd = 0;
  SocketAddress addr;

  if (!GetSockAddr(url, &addr)) {
    return -1;
  }

  // create server socket
  listenFd = CreateSocket(addr.sa.sa_family);
  if (listenFd < 0) {
    MS_LOG(ERROR) << "Failed to create socket, url: " << url.c_str();
    return -1;
  }

  // bind
  if (::bind(listenFd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(SocketAddress)) != 0) {
    MS_LOG(ERROR) << "Failed to call bind, url: " << url.c_str() << " " << strerror(errno);
    if (close(listenFd) != 0) {
      MS_LOG(EXCEPTION) << "Failed to close fd:" << listenFd;
    }
    return -1;
  }

  // listen
  if (::listen(listenFd, SOCKET_LISTEN_BACKLOG) != 0) {
    MS_LOG(ERROR) << "Failed to call listen, fd: " << listenFd << ", errno: " << errno << ", url: " << url.c_str()
                  << " " << strerror(errno);
    if (close(listenFd) != 0) {
      MS_LOG(EXCEPTION) << "Failed to close fd:" << listenFd;
    }
    return -1;
  }
  return listenFd;
}

int SocketOperation::Accept(int sock_fd) {
  SocketAddress storage;
  socklen_t length = sizeof(storage);

  // accept connection
  auto acceptFd =
    ::accept4(sock_fd, reinterpret_cast<struct sockaddr *>(&storage), &length, SOCK_NONBLOCK | SOCK_CLOEXEC);
  if (acceptFd < 0) {
    MS_LOG(ERROR) << "Failed to call accept, errno: " << errno << ", server: " << sock_fd;
    return acceptFd;
  }
  if (SetSocketOptions(acceptFd) < 0) {
    MS_LOG(ERROR) << "Failed to set socket options for accepted socket: " << acceptFd;
  }
  return acceptFd;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
