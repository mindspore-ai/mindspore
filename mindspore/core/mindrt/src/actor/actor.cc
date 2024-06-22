/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "actor/actor.h"
#include "actor/actormgr.h"
#include "actor/iomgr.h"

namespace mindspore {
ActorBase::ActorBase() : mailbox(nullptr), id("", ActorMgr::GetActorMgrRef()->GetUrl()), actionFunctions() {}

ActorBase::ActorBase(const std::string &name)
    : mailbox(nullptr), id(name, ActorMgr::GetActorMgrRef()->GetUrl()), actionFunctions() {}

ActorBase::ActorBase(const std::string &name, ActorThreadPool *pool)
    : mailbox(nullptr), id(name, ActorMgr::GetActorMgrRef()->GetUrl()), actionFunctions(), pool_(pool) {}

ActorBase::~ActorBase() {}

void ActorBase::Spawn(const std::shared_ptr<ActorBase>, std::unique_ptr<MailBox> mailboxPtr) {
  // lock here or await(). and unlock at Quit() or at await.
  waiterLock.Wait();
  this->mailbox = std::move(mailboxPtr);
}

void ActorBase::Await() {
  std::string actorName = id.Name();
  // lock here or at spawn(). and unlock here or at worker(). wait for the worker to finish.
  MS_LOG(DEBUG) << "ACTOR is waiting for terminate to finish. a=" << actorName.c_str();
  waiterLock.Wait();
  waiterLock.Signal();

  // mailbox's hook may hold the actor reference, we need explicitly free the mailbox to avoid the memory leak. the
  // details can refer to the comments in ActorMgr::Spawn
  delete mailbox.release();
  MS_LOG(DEBUG) << "ACTOR succeeded in waiting. a=" << actorName.c_str();
}
void ActorBase::Terminate() {
  bool flag = false;
  if (terminating_.compare_exchange_strong(flag, true)) {
    std::unique_ptr<MessageBase> msg(new (std::nothrow) MessageBase("Terminate", MessageBase::Type::KTERMINATE));
    MINDRT_OOM_EXIT(msg);
    (void)EnqueMessage(std::move(msg));
  }
}

void ActorBase::HandlekMsg(const std::unique_ptr<MessageBase> &msg) {
  auto it = actionFunctions.find(msg->Name());
  if (it != actionFunctions.end()) {
    ActorFunction &func = it->second;
    func(msg);
  } else {
    MS_LOG(WARNING) << "ACTOR can not find function for message, a=" << id.Name().c_str()
                    << ",m=" << msg->Name().c_str();
  }
}
int ActorBase::EnqueMessage(std::unique_ptr<MessageBase> msg) const {
  int ret = mailbox->EnqueueMessage(std::move(msg));
  return ret;
}

void ActorBase::Quit() {
  Finalize();
  // lock at spawn(), unlock here.
  waiterLock.Signal();
}

void ActorBase::Run() {
  auto msgHandler = [this](const std::unique_ptr<MessageBase> &msg) {
    switch (msg->GetType()) {
      case MessageBase::Type::KMSG:
      case MessageBase::Type::KUDP: {
        if (Filter(msg)) {
          return ERRORCODE_SUCCESS;
        }
        this->HandlekMsg(msg);
        return ERRORCODE_SUCCESS;
      }
      case MessageBase::Type::KHTTP: {
        this->HandleHttp(msg);
        return ERRORCODE_SUCCESS;
      }
      case MessageBase::Type::KASYNC: {
        msg->Run(this);
        return ERRORCODE_SUCCESS;
      }
      case MessageBase::Type::KLOCAL: {
        this->HandleLocalMsg(msg);
        return ERRORCODE_SUCCESS;
      }
      case MessageBase::Type::KTERMINATE: {
        this->Quit();
        return ACTOR_TERMINATED;
      }
      case MessageBase::Type::KEXIT: {
        this->Exited(msg->From());
        return ERRORCODE_SUCCESS;
      }
    }
    return ERRORCODE_SUCCESS;
  };

  if (this->mailbox->TakeAllMsgsEachTime()) {
    while (auto msgs = mailbox->GetMsgs()) {
      for (auto it = msgs->begin(); it != msgs->end(); ++it) {
        std::unique_ptr<MessageBase> &msg = *it;
        if (msg == nullptr) {
          continue;
        }
        if (msgHandler(msg) == ACTOR_TERMINATED) {
          return;
        }
        msg.reset(nullptr);
      }
      msgs->clear();
    }
  } else {
    while (auto msg = mailbox->GetMsg()) {
      if (msgHandler(msg) == ACTOR_TERMINATED) {
        return;
      }
    }
  }
  return;
}

int ActorBase::Send(const AID &to, std::unique_ptr<MessageBase> msg) {
  msg->SetFrom(id);
  return ActorMgr::GetActorMgrRef()->Send(to, std::move(msg));
}
int ActorBase::Send(const AID &to, std::string &&name, std::string &&strMsg, bool remoteLink, bool isExactNotRemote) {
  std::unique_ptr<MessageBase> msg(
    new (std::nothrow) MessageBase(this->id, to, std::move(name), std::move(strMsg), MessageBase::Type::KMSG));
  MINDRT_OOM_EXIT(msg);
  return ActorMgr::GetActorMgrRef()->Send(to, std::move(msg), remoteLink, isExactNotRemote);
}

// register the message handle
void ActorBase::Receive(const std::string &msgName, ActorFunction &&func) {
  if (actionFunctions.find(msgName) != actionFunctions.end()) {
    MS_LOG(ERROR) << "ACTOR function's name conflicts, a=" << id.Name().c_str() << ",f=" << msgName.c_str();
    MINDRT_EXIT("function's name conflicts");
    return;
  }
  actionFunctions.emplace(msgName, std::move(func));
  return;
}

int ActorBase::Link(const AID &to) {
  auto io = ActorMgr::GetIOMgrRef(to);
  if (io != nullptr) {
    if (to.OK()) {
      io->Link(this->GetAID(), to);
      return ERRORCODE_SUCCESS;
    } else {
      return ACTOR_PARAMER_ERR;
    }
  } else {
    return IO_NOT_FIND;
  }
}
int ActorBase::UnLink(const AID &to) {
  auto io = ActorMgr::GetIOMgrRef(to);
  if (io != nullptr) {
    if (to.OK()) {
      io->UnLink(to);
      return ERRORCODE_SUCCESS;
    } else {
      return ACTOR_PARAMER_ERR;
    }
  } else {
    return IO_NOT_FIND;
  }
}

int ActorBase::Reconnect(const AID &to) {
  auto io = ActorMgr::GetIOMgrRef(to);
  if (io != nullptr) {
    if (to.OK()) {
      io->Reconnect(this->GetAID(), to);
      return ERRORCODE_SUCCESS;
    } else {
      return ACTOR_PARAMER_ERR;
    }
  } else {
    return IO_NOT_FIND;
  }
}

uint64_t ActorBase::GetOutBufSize(const AID &to) {
  auto io = ActorMgr::GetIOMgrRef(to);
  if (io != nullptr) {
    return io->GetOutBufSize();
  } else {
    return 0;
  }
}

uint64_t ActorBase::GetInBufSize(const AID &to) {
  auto io = ActorMgr::GetIOMgrRef(to);
  if (io != nullptr) {
    return io->GetInBufSize();
  } else {
    return 0;
  }
}

int ActorBase::AddRuleUdp(const std::string &peer, int recordNum) {
  const std::string udp = MINDRT_UDP;
  auto io = ActorMgr::GetIOMgrRef(udp);
  if (io != nullptr) {
    return io->AddRuleUdp(peer, recordNum);
  } else {
    return 0;
  }
}

void ActorBase::DelRuleUdp(const std::string &peer, bool outputLog) {
  const std::string udp = MINDRT_UDP;
  auto io = ActorMgr::GetIOMgrRef(udp);
  if (io != nullptr) {
    io->DelRuleUdp(peer, outputLog);
  }
}
}  // namespace mindspore
