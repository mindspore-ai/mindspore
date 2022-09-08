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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <mutex>
#include "utils/ms_utils.h"
#include "backend/common/session/session_basic.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace session {
using SessionCreator = std::function<std::shared_ptr<SessionBasic>()>;
class BACKEND_EXPORT SessionFactory {
 public:
  SessionFactory() = default;
  ~SessionFactory() = default;

  static SessionFactory &Get();
  void Register(const std::string &device_name, SessionCreator &&session_creator);
  std::shared_ptr<SessionBasic> Create(const std::string &device_name);

 private:
  DISABLE_COPY_AND_ASSIGN(SessionFactory)
  std::map<std::string, SessionCreator> session_creators_;
  inline static std::shared_ptr<SessionFactory> instance_;
  inline static std::once_flag instance_flag_;
};

class SessionRegistrar {
 public:
  SessionRegistrar(const std::string &device_name, SessionCreator &&session_creator) {
    SessionFactory::Get().Register(device_name, std::move(session_creator));
  }
  ~SessionRegistrar() = default;
};

#define MS_REG_SESSION(DEVICE_NAME, SESSION_CLASS)                           \
  static const SessionRegistrar g_session_registrar__##DEVICE_NAME##_##_reg( \
    DEVICE_NAME, []() { return std::make_shared<SESSION_CLASS>(); });
}  // namespace session
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_FACTORY_H_
