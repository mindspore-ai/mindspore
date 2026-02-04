/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "cluster/topology/utils.h"

#include <cctype>
#include <sstream>
#include <vector>
#include <algorithm>

namespace mindspore {
namespace distributed {
namespace cluster {

bool Utils::IsValidIPv4(const std::string &ip) {
  std::vector<std::string> parts;
  std::stringstream ss(ip);
  std::string part;

  while (std::getline(ss, part, '.')) {
    parts.push_back(part);
  }
  // Check if there are exactly 4 segments.
  if (parts.size() != kNumIpv4Parts) {
    return false;
  }

  for (const auto &p : parts) {
    // Segment must not be empty and must not exceed 3 digits count.
    if (p.empty() || p.size() > kMaxIpv4SegmentDigits) {
      return false;
    }
    // Check if all characters in the segment are digits
    if (!std::all_of(p.begin(), p.end(), ::isdigit)) {
      return false;
    }
    // Check for leading zeros (invalid unless segment is exactly "0")
    if (p.size() > 1 && p[0] == '0') {
      return false;
    }

    int num;
    try {
      num = std::stoi(p);
    } catch (...) {
      return false;
    }
    // Verify the number is within the valid IPv4 segment range (0-255).
    if (num < kMinIpv4SegmentValue || num > kMaxIpv4SegmentValue) {
      return false;
    }
  }

  return true;
}

bool Utils::ParseTcpUrlForIpv4(const std::string &url, std::string *ip, int64_t *port) {
  MS_EXCEPTION_IF_NULL(ip);
  MS_EXCEPTION_IF_NULL(port);

  try {
    const std::string prefix = "tcp://";

    if (url.size() < prefix.size() || url.substr(0, prefix.size()) != prefix) {
      MS_LOG(ERROR) << "Invalid TCP URL: must start with 'tcp://'";
      return false;
    }

    std::string hostPort = url.substr(prefix.size());
    if (hostPort.empty()) {
      MS_LOG(ERROR) << "Invalid TCP URL: missing IP and port";
      return false;
    }

    size_t colonPos = hostPort.find(':');
    if (colonPos == std::string::npos) {
      MS_LOG(ERROR) << "Invalid TCP URL: missing port separator ':'";
      return false;
    }

    // Eliminate IPv6.
    if (hostPort.find(':', colonPos + 1) != std::string::npos) {
      MS_LOG(ERROR) << "Invalid TCP URL: IPv6 addresses are not supported";
      return false;
    }

    *ip = hostPort.substr(0, colonPos);
    if (ip->empty()) {
      MS_LOG(ERROR) << "Invalid TCP URL: missing IP address";
      return false;
    }
    if (!IsValidIPv4(*ip)) {
      MS_LOG(ERROR) << "Invalid TCP URL: invalid IPv4 address - " << *ip;
      return false;
    }

    std::string portStr = hostPort.substr(colonPos + 1);
    if (portStr.empty()) {
      MS_LOG(ERROR) << "Invalid TCP URL: missing port number";
      return false;
    }
    if (!std::all_of(portStr.begin(), portStr.end(), ::isdigit)) {
      MS_LOG(ERROR) << "Invalid TCP URL: port must be numeric - " << portStr;
      return false;
    }

    *port = std::stoll(portStr);
    if (*port < kMinValidPort || *port > kMaxValidPort) {
      MS_LOG(ERROR) << "Invalid TCP URL: port out of range [1, 65535] - " << *port;
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Failed to parse TCP URL: " << e.what();
    return false;
  }
}
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
