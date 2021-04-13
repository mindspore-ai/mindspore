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

#ifndef MINDSPORE_CORE_MINDRT_SRC_ACTOR_ACTORPOLICY_H
#define MINDSPORE_CORE_MINDRT_SRC_ACTOR_ACTORPOLICY_H

#include <list>
#include <memory>
#include "actor/actorpolicyinterface.h"

namespace mindspore {
class ShardedThread : public ActorPolicy {
 public:
  explicit ShardedThread(const std::shared_ptr<ActorBase> &actor);
  virtual ~ShardedThread();

 protected:
  virtual void Terminate(const ActorBase *actor);
  virtual int EnqueMessage(std::unique_ptr<MessageBase> &msg);
  virtual std::list<std::unique_ptr<MessageBase>> *GetMsgs();
  virtual void Notify();

 private:
  bool ready;
  bool terminated;
  std::shared_ptr<ActorBase> actor;
};

class SingleThread : public ActorPolicy {
 public:
  SingleThread();
  virtual ~SingleThread();

 protected:
  virtual void Terminate(const ActorBase *actor);
  virtual int EnqueMessage(std::unique_ptr<MessageBase> &msg);
  virtual std::list<std::unique_ptr<MessageBase>> *GetMsgs();
  virtual void Notify();

 private:
  std::condition_variable conditionVar;
};
};  // end of namespace mindspore
#endif
