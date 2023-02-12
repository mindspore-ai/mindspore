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
#include <map>
#include <list>
#include <string>
#include <memory>
#include <utility>

#include "actor/actormgr.h"
#include "actor/iomgr.h"

namespace mindspore {
ActorMgr ActorMgr::actorMgr;
std::map<std::string, std::shared_ptr<IOMgr>> ActorMgr::ioMgrs;

std::shared_ptr<IOMgr> &ActorMgr::GetIOMgrRef(const std::string &protocol) {
  auto it = ioMgrs.find(protocol);
  if (it != ioMgrs.end()) {
    return it->second;
  } else {
    MS_LOG(DEBUG) << "Can't find IOMgr of protocol " << protocol.c_str();
    static std::shared_ptr<IOMgr> nullIOMgr;
    return nullIOMgr;
  }
}
ActorMgr::ActorMgr() : actors(), procotols(), urls() {
  actors.clear();
  procotols.clear();
  urls.clear();
}

ActorMgr::~ActorMgr() {
  if (inner_pool_ != nullptr) {
    delete inner_pool_;
    inner_pool_ = nullptr;
  }
}

int ActorMgr::Initialize(bool use_inner_pool, size_t actor_thread_num, size_t max_thread_num, size_t actor_queue_size) {
  bool expected = false;
  if (!initialized_.compare_exchange_strong(expected, true)) {
    MS_LOG(DEBUG) << "Actor Manager has been initialized before";
    return MINDRT_OK;
  }
  // create inner thread pool only when specified use_inner_pool
  if (use_inner_pool) {
    ActorThreadPool::set_actor_queue_size(actor_queue_size);
    if (max_thread_num <= actor_thread_num) {
      inner_pool_ = ActorThreadPool::CreateThreadPool(actor_thread_num);
      if (inner_pool_ == nullptr) {
        MS_LOG(ERROR) << "ActorMgr CreateThreadPool failed";
        return MINDRT_ERROR;
      }
    } else {
      inner_pool_ = ActorThreadPool::CreateThreadPool(actor_thread_num, max_thread_num, {});
      if (inner_pool_ == nullptr) {
        MS_LOG(ERROR) << "ActorMgr CreateThreadPool failed";
        return MINDRT_ERROR;
      }
      inner_pool_->SetActorThreadNum(actor_thread_num);
      inner_pool_->SetKernelThreadNum(max_thread_num - actor_thread_num);
    }
    if (inner_pool_ != nullptr) {
      inner_pool_->SetMaxSpinCount(kDefaultSpinCount);
      inner_pool_->SetSpinCountMaxValue();
      inner_pool_->SetKernelThreadMaxSpinCount(kDefaultKernelSpinCount);
      inner_pool_->SetWorkerIdMap();
    }
  }
  return MINDRT_OK;
}

void ActorMgr::SetActorReady(const ActorReference &actor) const {
  // use inner thread pool or actor thread pool created externally
  // priority to use actor thread pool
  MINDRT_OOM_EXIT(actor);
  ActorThreadPool *pool = actor->pool_ ? actor->pool_ : inner_pool_;
  if (pool == nullptr) {
    MS_LOG(ERROR) << "ThreadPool is nullptr, " << actor->pool_ << ", " << inner_pool_
                  << ", actor: " << actor->GetAID().Name();
    return;
  }
  pool->PushActorToQueue(actor.get());
}

const std::string ActorMgr::GetUrl(const std::string &protocol) {
  auto it = procotols.find(protocol);
  if (it != procotols.end()) {
    return it->second;
  } else if (procotols.size() > 0) {
    return procotols.begin()->second;
  } else {
    return "";
  }
}

void ActorMgr::AddUrl(const std::string &protocol, const std::string &url) {
  procotols[protocol] = url;
  AID id("a@" + url);
  (void)urls.insert(id.GetIp() + ":" + std::to_string(id.GetPort()));
  (void)urls.insert(id.GetProtocol() + "://" + id.GetIp() + ":" + std::to_string(id.GetPort()));
  (void)urls.insert(std::string("127.0.0.1:") + std::to_string(id.GetPort()));
  (void)urls.insert(protocol + "://127.0.0.1:" + std::to_string(id.GetPort()));
}

void ActorMgr::AddIOMgr(const std::string &protocol, const std::shared_ptr<IOMgr> &ioMgr) { ioMgrs[protocol] = ioMgr; }

void ActorMgr::RemoveActor(const std::string &name) {
  actorsMutex.lock();
  (void)actors.erase(name);
  actorsMutex.unlock();
}

void ActorMgr::TerminateAll() {
  if (actors.empty()) {
    return;
  }
  // copy all the actors
  std::list<ActorReference> actorsWaiting;
  actorsMutex.lock();
  for (auto actorIt = actors.begin(); actorIt != actors.end(); ++actorIt) {
    actorsWaiting.push_back(actorIt->second);
  }
  actorsMutex.unlock();

  // send terminal msg to all actors.
  for (auto actorIt = actorsWaiting.begin(); actorIt != actorsWaiting.end(); ++actorIt) {
    (*actorIt)->Terminate();
  }

  // wait actor's thread to finish and remove actor.
  for (auto actorIt = actorsWaiting.begin(); actorIt != actorsWaiting.end(); ++actorIt) {
    (*actorIt)->Await();
    RemoveActor((*actorIt)->GetAID().Name());
  }
}

void ActorMgr::Finalize() {
  this->TerminateAll();
  MS_LOG(INFO) << "mindrt Actors finish exiting.";

  // stop all actor threads;
  MS_LOG(INFO) << "mindrt Threads finish exiting.";

  // stop iomgr thread
  for (auto mgrIt = ioMgrs.begin(); mgrIt != ioMgrs.end(); ++mgrIt) {
    MS_LOG(INFO) << "finalize IOMgr=" << mgrIt->first.c_str();
    mgrIt->second->Finalize();
  }

  // delete actor thread pool if use_inner_pool
  delete inner_pool_;
  inner_pool_ = nullptr;
  MS_LOG(INFO) << "mindrt IOMGRS finish exiting.";
}

ActorReference ActorMgr::GetActor(const AID &id) {
#ifndef MS_COMPILE_IOS
  actorsMutex.lock_shared();
#else
  actorsMutex.lock();
#endif
  const auto &actorIt = actors.find(id.Name());
  if (actorIt != actors.end()) {
    auto &result = actorIt->second;
#ifndef MS_COMPILE_IOS
    actorsMutex.unlock_shared();
#else
    actorsMutex.unlock();
#endif
    return result;
  } else {
#ifndef MS_COMPILE_IOS
    actorsMutex.unlock_shared();
#else
    actorsMutex.unlock();
#endif
    MS_LOG(DEBUG) << "can't find ACTOR with name=" << id.Name().c_str();
    return nullptr;
  }
}

int ActorMgr::EnqueueMessage(const mindspore::ActorReference actor, std::unique_ptr<mindspore::MessageBase> msg) {
  return actor->EnqueMessage(std::move(msg));
}

int ActorMgr::Send(const AID &to, std::unique_ptr<MessageBase> msg, bool remoteLink, bool isExactNotRemote) {
  // The destination is local
#ifdef BUILD_LITE
  auto actor = GetActor(to);
  if (actor != nullptr) {
    return EnqueueMessage(actor, std::move(msg));
  } else {
    return ACTOR_NOT_FIND;
  }
#else
  if (IsLocalAddres(to)) {
    auto actor = GetActor(to);
    if (actor != nullptr) {
      if (to.GetProtocol() == MINDRT_UDP && msg->GetType() == MessageBase::Type::KMSG) {
        msg->type = MessageBase::Type::KUDP;
      }
      return EnqueueMessage(actor, std::move(msg));
    } else {
      return ACTOR_NOT_FIND;
    }
  } else {
    // send to remote actor
    if (msg->GetType() != MessageBase::Type::KMSG) {
      MS_LOG(ERROR) << "The msg is not KMSG,it can't send to remote=" << std::string(to).c_str();
      return ACTOR_PARAMER_ERR;
    } else {
      // null
    }
    msg->SetTo(to);
    auto &io = ActorMgr::GetIOMgrRef(to);
    if (io != nullptr) {
      return io->Send(std::move(msg), remoteLink, isExactNotRemote);
    } else {
      MS_LOG(ERROR) << "The protocol is not supported:"
                    << "p=" << to.GetProtocol().c_str() << ",f=" << msg->From().Name().c_str()
                    << ",t=" << to.Name().c_str() << ",m=" << msg->Name().c_str();
      return IO_NOT_FIND;
    }
  }
#endif
}

AID ActorMgr::Spawn(const ActorReference &actor, bool shareThread) {
  actorsMutex.lock();
  if (actors.find(actor->GetAID().Name()) != actors.end()) {
    actorsMutex.unlock();
    MS_LOG(ERROR) << "The actor's name conflicts,name:" << actor->GetAID().Name().c_str();
    MINDRT_EXIT("Actor name conflicts.");
  }
  MS_LOG(DEBUG) << "ACTOR was spawned,a=" << actor->GetAID().Name().c_str();

  if (shareThread) {
    auto mailbox = std::make_unique<NonblockingMailBox>();
    auto hook = std::make_unique<std::function<void()>>([actor]() {
      auto actor_mgr = actor->get_actor_mgr();
      if (actor_mgr != nullptr) {
        actor_mgr->SetActorReady(actor);
      } else {
        ActorMgr::GetActorMgrRef()->SetActorReady(actor);
      }
    });
    // the mailbox has this hook, the hook holds the actor reference, the actor has the mailbox. this is a cycle which
    // will leads to memory leak. in order to fix this issue, we should explicitly free the mailbox when terminate the
    // actor
    mailbox->SetNotifyHook(std::move(hook));
    actor->Spawn(actor, std::move(mailbox));
  } else {
    auto mailbox = std::unique_ptr<MailBox>(new (std::nothrow) BlockingMailBox());
    actor->Spawn(actor, std::move(mailbox));
    ActorMgr::GetActorMgrRef()->SetActorReady(actor);
  }
  (void)this->actors.emplace(actor->GetAID().Name(), actor);
  actorsMutex.unlock();
  // long time
  actor->Init();
  return actor->GetAID();
}

void ActorMgr::Terminate(const AID &id) {
  auto actor = GetActor(id);
  if (actor != nullptr) {
    actor->Terminate();
    // Wait actor's thread to finish.
    actor->Await();
    RemoveActor(id.Name());
  }
}

void ActorMgr::Wait(const AID &id) {
  auto actor = GetActor(id);
  if (actor != nullptr) {
    actor->Await();
  }
}
};  // end of namespace mindspore
