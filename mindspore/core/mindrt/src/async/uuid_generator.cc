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

#include "async/uuid_generator.h"
#include <atomic>

namespace mindspore {
namespace uuid_generator {
std::string UUID::ToString() {
  std::ostringstream ret;
  ret << *this;
  return ret.str();
}
}  // namespace uuid_generator

namespace localid_generator {
int GenLocalActorId() {
  static std::atomic<int> localActorId(0);
  return localActorId.fetch_add(1);
}

#ifdef HTTP_ENABLED
// not support muti-thread
int GenHttpClientConnId() {
  static int httpClientConnId = 1;
  if (httpClientConnId == INT_MAX) {
    httpClientConnId = 1;
  }
  return httpClientConnId++;
}

// not support muti-thread
int GenHttpServerConnId() {
  static int httpServerConnId = 1;
  if (httpServerConnId == INT_MAX) {
    httpServerConnId = 1;
  }
  return httpServerConnId++;
}
#endif
}  // namespace localid_generator
}  // namespace mindspore
