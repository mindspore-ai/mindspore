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

#include "ps/comm/comm_util.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <regex>

namespace mindspore {
namespace ps {
namespace comm {

bool CommUtil::CheckIpWithRegex(const std::string &ip) {
  std::regex pattern("((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?).){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)");
  std::smatch res;
  if (regex_match(ip, res, pattern)) {
    return true;
  }
  return false;
}

void CommUtil::CheckIp(const std::string &ip) {
  if (!CheckIpWithRegex(ip)) {
    MS_LOG(EXCEPTION) << "Server address" << ip << " illegal!";
  }
  int64_t uAddr = inet_addr(ip.c_str());
  if (INADDR_NONE == uAddr) {
    MS_LOG(EXCEPTION) << "Server address illegal, inet_addr converting failed!";
  }
}
}  // namespace comm
}  // namespace ps
}  // namespace mindspore
