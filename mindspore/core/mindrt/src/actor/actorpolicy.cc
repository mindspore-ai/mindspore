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

#include <string>
#include <utility>
#include <list>
#include <memory>
#include "actor/actor.h"
#include "actor/actormgr.h"
#include "actor/actorpolicy.h"

namespace mindspore {

void ActorPolicy::SetRunningStatus(bool startRun) {
  std::lock_guard<std::mutex> lock(mailboxLock);
  this->start = startRun;
  Notify();
}

SingleThread::SingleThread() {}
SingleThread::~SingleThread() {}

void SingleThread::Terminate(const ActorBase *actor) {
  std::string actorName = actor->GetAID().Name();
  MS_LOG(DEBUG) << "ACTOR SingleThread received terminate message, v=" << actorName.c_str();
  // remove actor from actorMgr
  ActorMgr::GetActorMgrRef()->RemoveActor(actorName);
}

int SingleThread::EnqueMessage(std::unique_ptr<MessageBase> &msg) {
  int result;
  {
    std::lock_guard<std::mutex> lock(mailboxLock);
    enqueMailbox->push_back(std::move(msg));
    result = ++msgCount;
  }
  // Notify when the count of message  is from  empty to one.
  if (start && result == 1) {
    conditionVar.notify_one();
  }
  return result;
}
void SingleThread::Notify() {
  if (start && msgCount > 0) {
    conditionVar.notify_one();
  }
}

std::list<std::unique_ptr<MessageBase>> *SingleThread::GetMsgs() {
  std::list<std::unique_ptr<MessageBase>> *result;
  std::unique_lock<std::mutex> lock(mailboxLock);
  conditionVar.wait(lock, [this] { return (!this->enqueMailbox->empty()); });
  SwapMailbox();
  // REF_PRIVATE_MEMBER
  result = dequeMailbox;

  return result;
}

ShardedThread::ShardedThread(const std::shared_ptr<ActorBase> &aActor)
    : ready(false), terminated(false), actor(aActor) {}
ShardedThread::~ShardedThread() {}

void ShardedThread::Terminate(const ActorBase *aActor) {
  std::string actorName = aActor->GetAID().Name();
  MS_LOG(DEBUG) << "ACTOR ShardedThread received terminate message,v=" << actorName.c_str();
  // remove actor from actorMgr
  ActorMgr::GetActorMgrRef()->RemoveActor(actorName);

  mailboxLock.lock();
  terminated = true;
  this->actor = nullptr;
  mailboxLock.unlock();
}

int ShardedThread::EnqueMessage(std::unique_ptr<MessageBase> &msg) {
  int result;
  mailboxLock.lock();
  enqueMailbox->push_back(std::move(msg));
  // true : The actor is running. else  the actor will  be  ready to run.
  if (start && ready == false && terminated == false) {
    ActorMgr::GetActorMgrRef()->SetActorReady(actor);
    ready = true;
  }
  result = ++msgCount;
  mailboxLock.unlock();
  return result;
}

void ShardedThread::Notify() {
  if (start && ready == false && terminated == false && msgCount > 0) {
    ActorMgr::GetActorMgrRef()->SetActorReady(actor);
    ready = true;
  }
}

std::list<std::unique_ptr<MessageBase>> *ShardedThread::GetMsgs() {
  std::list<std::unique_ptr<MessageBase>> *result;
  mailboxLock.lock();
  if (enqueMailbox->empty()) {
    ready = false;
    result = nullptr;
  } else {
    ready = true;
    SwapMailbox();
    result = dequeMailbox;
  }
  mailboxLock.unlock();
  return result;
}
};  // end of namespace mindspore
