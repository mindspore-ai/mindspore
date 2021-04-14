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

#ifndef MINDSPORE_CORE_MINDRT_SRC_ACTOR_ACTORPOLICYINTERFACE_H
#define MINDSPORE_CORE_MINDRT_SRC_ACTOR_ACTORPOLICYINTERFACE_H

#include <list>
#include <memory>

namespace mindspore {
class ActorPolicy {
 public:
  ActorPolicy() : mailbox1(), mailbox2() {
    enqueMailbox = &mailbox1;
    dequeMailbox = &mailbox2;
    msgCount = 0;
    start = false;
  }
  virtual ~ActorPolicy() {}
  inline void SwapMailbox() {
    std::list<std::unique_ptr<MessageBase>> *temp;
    temp = enqueMailbox;
    enqueMailbox = dequeMailbox;
    dequeMailbox = temp;
    msgCount = 0;
  }

 protected:
  void SetRunningStatus(bool startRun);
  virtual void Terminate(const ActorBase *actor) = 0;
  virtual int EnqueMessage(std::unique_ptr<MessageBase> &msg) = 0;
  virtual std::list<std::unique_ptr<MessageBase>> *GetMsgs() = 0;
  virtual void Notify() = 0;

  std::list<std::unique_ptr<MessageBase>> *enqueMailbox;
  std::list<std::unique_ptr<MessageBase>> *dequeMailbox;

  int msgCount;
  bool start;
  std::mutex mailboxLock;

 private:
  friend class ActorBase;

  std::list<std::unique_ptr<MessageBase>> mailbox1;
  std::list<std::unique_ptr<MessageBase>> mailbox2;
};
};  // end of namespace mindspore
#endif
