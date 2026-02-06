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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_DISTRIBUTED_CLUSTER_UTILS_H
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_DISTRIBUTED_CLUSTER_UTILS_H

#include <string>
#include "utils/log_adapter.h"

namespace mindspore {
namespace distributed {
namespace cluster {
constexpr auto kMinValidPort = 1;
constexpr auto kMaxValidPort = 65535;
constexpr auto kNumIpv4Parts = 4;
constexpr auto kMaxIpv4SegmentDigits = 3;
constexpr auto kMinIpv4SegmentValue = 0;
constexpr auto kMaxIpv4SegmentValue = 255;
class Utils {
 public:
  Utils() = default;
  ~Utils() = default;
  static bool IsValidIPv4(const std::string &ip);
  static bool ParseTcpUrlForIpv4(const std::string &url, std::string *ip, int64_t *port);
};
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_DISTRIBUTED_CLUSTER_UTILS_H
