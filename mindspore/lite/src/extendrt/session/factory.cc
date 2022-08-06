/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/session/factory.h"
#include <functional>
#include <memory>

#include "extendrt/session/type.h"
#include "extendrt/infer_session.h"

namespace mindspore {
SessionRegistry &SessionRegistry::GetInstance() {
  static SessionRegistry instance;
  return instance;
}

void SessionRegistry::RegSession(const mindspore::SessionType &session_type,
                                 std::function<std::shared_ptr<InferSession>(const SessionConfig &)> creator) {
  session_map_[session_type] = creator;
}

std::shared_ptr<InferSession> SessionRegistry::GetSession(const mindspore::SessionType &session_type,
                                                          const SessionConfig &config) {
  auto it = session_map_.find(session_type);
  if (it == session_map_.end()) {
    return nullptr;
  }
  return it->second(config);
}
}  // namespace mindspore
