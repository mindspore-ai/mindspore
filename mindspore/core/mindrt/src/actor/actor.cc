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

#include "actor/actor.h"
#include "actor/actormgr.h"
#include "actor/actorpolicyinterface.h"
#include "actor/iomgr.h"

namespace mindspore {
ActorBase::ActorBase(const std::string &name)
    : actorThread(nullptr), id(name, ActorMgr::GetActorMgrRef()->GetUrl()), actionFunctions() {}

ActorBase::~ActorBase() {}

void ActorBase::Spawn(std::shared_ptr<ActorBase> &actor, std::unique_ptr<ActorPolicy> thread) {
  // lock here or await(). and unlock at Quit() or at aweit.
  waiterLock.lock();

  actorThread = std::move(thread);
}
void ActorBase::SetRunningStatus(bool start) { actorThread->SetRunningStatus(start); }

void ActorBase::Await() {
  std::string actorName = id.Name();
  // lock here or at spawn(). and unlock here or at worker(). wait for the worker to finish.
  MS_LOG(DEBUG) << "ACTOR is waiting for terminate to finish. a=" << actorName.c_str();

  waiterLock.lock();
  waiterLock.unlock();
  MS_LOG(DEBUG) << "ACTOR succeeded in waiting. a=" << actorName.c_str();
}
void ActorBase::Terminate() {
  std::unique_ptr<MessageBase> msg(new (std::nothrow) MessageBase("Terminate", MessageBase::Type::KTERMINATE));
  BUS_OOM_EXIT(msg);
  (void)EnqueMessage(std::move(msg));
}

void ActorBase::HandlekMsg(std::unique_ptr<MessageBase> &msg) {
  auto it = actionFunctions.find(msg->Name());
  if (it != actionFunctions.end()) {
    ActorFunction &func = it->second;
    func(msg);
  } else {
    ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_WARNING, PID_LITEBUS_LOG,
                        "ACTOR can not find function for message (%s)", "a=%s,m=%s", id.Name().c_str(),
                        msg->Name().c_str());
    MS_LOG(WARNING) << "ACTOR can not find function for message, a=" << id.Name().c_str()
                    << ",m=" << msg->Name().c_str();
  }
}
int ActorBase::EnqueMessage(std::unique_ptr<MessageBase> msg) { return actorThread->EnqueMessage(msg); }

void ActorBase::Quit() {
  Finalize();
  actorThread->Terminate(this);
  // lock at spawn(), unlock here.
  waiterLock.unlock();
}

void ActorBase::Run() {
  for (;;) {
    auto msgs = actorThread->GetMsgs();
    if (msgs == nullptr) {
      return;
    }
    for (auto it = msgs->begin(); it != msgs->end(); ++it) {
      std::unique_ptr<MessageBase> &msg = *it;
      if (msg == nullptr) {
        continue;
      }
      AddMsgRecord(msg->Name());
      switch (msg->GetType()) {
        case MessageBase::Type::KMSG:
        case MessageBase::Type::KUDP: {
          if (Filter(msg)) {
            continue;
          }
          this->HandlekMsg(msg);
          break;
        }
        case MessageBase::Type::KHTTP: {
          this->HandleHttp(std::move(msg));
          break;
        }
        case MessageBase::Type::KASYNC: {
          msg->Run(this);
          break;
        }
        case MessageBase::Type::KLOCAL: {
          this->HandleLocalMsg(std::move(msg));
          break;
        }
        case MessageBase::Type::KTERMINATE: {
          this->Quit();
          return;
        }
        case MessageBase::Type::KEXIT: {
          this->Exited(msg->From());
          break;
        }
      }
    }
    msgs->clear();
  }
}

int ActorBase::Send(const AID &to, std::unique_ptr<MessageBase> msg) {
  msg->SetFrom(id);
  return ActorMgr::GetActorMgrRef()->Send(to, std::move(msg));
}
int ActorBase::Send(const AID &to, std::string &&name, std::string &&strMsg, bool remoteLink, bool isExactNotRemote) {
  std::unique_ptr<MessageBase> msg(
    new (std::nothrow) MessageBase(this->id, to, std::move(name), std::move(strMsg), MessageBase::Type::KMSG));
  BUS_OOM_EXIT(msg);
  return ActorMgr::GetActorMgrRef()->Send(to, std::move(msg), remoteLink, isExactNotRemote);
}

// register the message handle
void ActorBase::Receive(const std::string &msgName, ActorFunction &&func) {
  if (actionFunctions.find(msgName) != actionFunctions.end()) {
    MS_LOG(ERROR) << "ACTOR function's name conflicts, a=%s" << id.Name().c_str() << ",f=%s" << msgName.c_str();
    BUS_EXIT("function's name conflicts");
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
  const std::string udp = BUS_UDP;
  auto io = ActorMgr::GetIOMgrRef(udp);
  if (io != nullptr) {
    return io->AddRuleUdp(peer, recordNum);
  } else {
    return 0;
  }
}

void ActorBase::DelRuleUdp(const std::string &peer, bool outputLog) {
  const std::string udp = BUS_UDP;
  auto io = ActorMgr::GetIOMgrRef(udp);
  if (io != nullptr) {
    io->DelRuleUdp(peer, outputLog);
  }
}
}  // namespace mindspore
