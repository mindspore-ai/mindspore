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
#ifndef MINDSPORE_LITE_EXTENDRT_SESSION_FACTORY_H_
#define MINDSPORE_LITE_EXTENDRT_SESSION_FACTORY_H_

#include <functional>
#include <memory>

#include "extendrt/session/type.h"
#include "extendrt/infer_session.h"
#include "include/api/context.h"
namespace mindspore {
using InferSessionRegFunc =
  std::function<std::shared_ptr<InferSession>(const std::shared_ptr<Context> &, const ConfigInfos &)>;

class SessionRegistry {
 public:
  SessionRegistry() = default;
  virtual ~SessionRegistry() = default;

  static SessionRegistry &GetInstance();

  void RegSession(const mindspore::SessionType &session_type, const InferSessionRegFunc &creator);

  std::shared_ptr<InferSession> GetSession(const mindspore::SessionType &session_type, const std::shared_ptr<Context> &,
                                           const ConfigInfos &);

 private:
  mindspore::HashMap<SessionType, InferSessionRegFunc> session_map_;
};

class SessionRegistrar {
 public:
  SessionRegistrar(const mindspore::SessionType &session_type, const InferSessionRegFunc &creator) {
    SessionRegistry::GetInstance().RegSession(session_type, creator);
  }
  ~SessionRegistrar() = default;
};

#define REG_SESSION(session_type, creator) static SessionRegistrar g_##session_type##Session(session_type, creator);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_FACTORY_H_
