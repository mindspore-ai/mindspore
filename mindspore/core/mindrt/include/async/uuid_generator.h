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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_UUID_GENERATOR_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ASYNC_UUID_GENERATOR_H

#include <string>
#include "async/uuid_base.h"

namespace mindspore {
namespace uuid_generator {
struct UUID : public mindspore::uuids::uuid {
 public:
  explicit UUID(const mindspore::uuids::uuid &inputUUID) : mindspore::uuids::uuid(inputUUID) {}
  static UUID GetRandomUUID();
  std::string ToString();
};
}  // namespace uuid_generator

namespace localid_generator {
int GenLocalActorId();

#ifdef HTTP_ENABLED
int GenHttpClientConnId();
int GenHttpServerConnId();
#endif
}  // namespace localid_generator
}  // namespace mindspore
#endif
