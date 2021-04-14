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

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_ACTORAPP_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_ACTORAPP_H

#include <memory>
#include <utility>
#include <map>
#include <string>
#include "actor/actor.h"

namespace mindspore {

class MessageLocal : public MessageBase {
 public:
  MessageLocal(const AID &from, const AID &to, const std::string &name, void *aPtr)
      : MessageBase(from, to, name, "LocalMsg", Type::KLOCAL), ptr(aPtr) {}
  ~MessageLocal() {}
  void *ptr;
};

class AppActor : public ActorBase {
 public:
  typedef std::function<void(std::unique_ptr<MessageBase>)> APPBehavior;

  explicit AppActor(const std::string &name) : ActorBase(name) {}
  ~AppActor() {}

  inline int Send(const AID &to, std::unique_ptr<MessageBase> msg) { return ActorBase::Send(to, std::move(msg)); }
  // send T message to the actor
  template <typename M>
  int Send(const std::string &to, const std::string &msgName, std::unique_ptr<M> msg) {
    std::unique_ptr<MessageLocal> localMsg(new (std::nothrow) MessageLocal(GetAID(), to, msgName, msg.release()));
    BUS_OOM_EXIT(localMsg);
    return Send(to, std::move(localMsg));
  }

  // register the message handle
  template <typename T, typename M>
  void Receive(const std::string &msgName, void (T::*method)(const AID &, std::unique_ptr<M>)) {
    APPBehavior behavior = std::bind(&BehaviorBase<T, M>, static_cast<T *>(this), method, std::placeholders::_1);

    if (appBehaviors.find(msgName) != appBehaviors.end()) {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "ACTOR msgName conflict:%s",
                          "a=%s,msg=%s", GetAID().Name().c_str(), msgName.c_str());
      BUS_EXIT("msgName conflicts.");
      return;
    }

    appBehaviors.emplace(msgName, behavior);
    return;
  }

  template <typename T, typename M>
  static void BehaviorBase(T *t, void (T::*method)(const AID &, std::unique_ptr<M>), std::unique_ptr<MessageBase> msg) {
    (t->*method)(msg->From(),
            std::move(std::unique_ptr<M>((reinterpret_cast<M *>static_cast<MessageLocal *>(msg.get())->ptr)));
    return;
  }

 protected:
  // KLOCALMsg handler
  virtual void HandleLocalMsg(std::unique_ptr<MessageBase> msg) {
    auto it = appBehaviors.find(msg->Name());
    if (it != appBehaviors.end()) {
      it->second(std::move(msg));
    } else {
      ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "ACTOR can not finds handler:%s",
                          "a=%s,msg=%s,hdlno=%zd", GetAID().Name().c_str(), msg->Name().c_str(), appBehaviors.size());
    }
  }

 private:
  std::map<std::string, APPBehavior> appBehaviors;
};
}  // namespace mindspore
#endif
