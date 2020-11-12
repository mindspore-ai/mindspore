/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ps/core/comm_util.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <regex>

namespace mindspore {
namespace ps {
namespace core {

bool CommUtil::CheckIpWithRegex(const std::string &ip) {
  std::regex pattern("((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?).){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)");
  std::smatch res;
  if (regex_match(ip, res, pattern)) {
    return true;
  }
  return false;
}

bool CommUtil::CheckIp(const std::string &ip) {
  if (!CheckIpWithRegex(ip)) {
    return false;
  }
  int64_t uAddr = inet_addr(ip.c_str());
  if (INADDR_NONE == uAddr) {
    return false;
  }
  return true;
}

void CommUtil::GetAvailableInterfaceAndIP(std::string *interface, std::string *ip) {
  MS_EXCEPTION_IF_NULL(interface);
  MS_EXCEPTION_IF_NULL(ip);
  struct ifaddrs *if_address = nullptr;
  struct ifaddrs *ifa = nullptr;

  interface->clear();
  ip->clear();
  getifaddrs(&if_address);
  for (ifa = if_address; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) {
      continue;
    }

    if (ifa->ifa_addr->sa_family == AF_INET && (ifa->ifa_flags & IFF_LOOPBACK) == 0) {
      char address_buffer[INET_ADDRSTRLEN] = {0};
      void *sin_addr_ptr = &(reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr))->sin_addr;
      MS_EXCEPTION_IF_NULL(sin_addr_ptr);
      const char *net_ptr = inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);
      MS_EXCEPTION_IF_NULL(net_ptr);

      *ip = address_buffer;
      *interface = ifa->ifa_name;
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(if_address);
  freeifaddrs(if_address);
}

}  // namespace core
}  // namespace ps
}  // namespace mindspore
