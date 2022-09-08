/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/common/session/session_factory.h"
#include <memory>
#include <string>

namespace mindspore {
namespace session {
SessionFactory &SessionFactory::Get() {
  std::call_once(instance_flag_, []() {
    if (instance_ == nullptr) {
      instance_ = std::make_shared<SessionFactory>();
    }
  });
  return *instance_;
}

void SessionFactory::Register(const std::string &device_name, SessionCreator &&session_creator) {
  if (session_creators_.end() == session_creators_.find(device_name)) {
    (void)session_creators_.emplace(device_name, session_creator);
  }
}

std::shared_ptr<SessionBasic> SessionFactory::Create(const std::string &device_name) {
  auto iter = session_creators_.find(device_name);
  if (session_creators_.end() != iter) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return (iter->second)();
  }
  return nullptr;
}
}  // namespace session
}  // namespace mindspore
