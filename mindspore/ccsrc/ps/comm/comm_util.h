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

#ifndef MINDSPORE_CCSRC_PS_COMM_COMM_UTIL_H_
#define MINDSPORE_CCSRC_PS_COMM_COMM_UTIL_H_

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>

#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
namespace comm {

class CommUtil {
 public:
  static bool CheckIpWithRegex(const std::string &ip);
  static void CheckIp(const std::string &ip);
};
}  // namespace comm
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_COMM_COMM_UTIL_H_
