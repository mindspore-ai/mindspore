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

#ifndef MINDSPORE_CORE_MINDRT_SRC_ACTOR_ACTORMGR_H
#define MINDSPORE_CORE_MINDRT_SRC_ACTOR_ACTORMGR_H

#include <atomic>
#include <set>
#include <utility>
#include <map>
#include <memory>
#include <string>
#ifndef MS_COMPILE_IOS
#include <shared_mutex>
#endif
#include <vector>
#include "actor/actor.h"
#include "thread/actor_threadpool.h"
#include "thread/hqueue.h"

namespace mindspore {
class ActorBase;
class IOMgr;
class MS_CORE_API ActorMgr {
 public:
  static inline ActorMgr *GetActorMgrRef() { return &actorMgr; }

  static std::shared_ptr<IOMgr> &GetIOMgrRef(const std::string &protocol = "tcp");

  static inline std::shared_ptr<IOMgr> &GetIOMgrRef(const AID &to) { return GetIOMgrRef(to.GetProtocol()); }

  static void Receive(std::unique_ptr<MessageBase> msg) {
    auto to = msg->To().Name();
    (void)ActorMgr::GetActorMgrRef()->Send(AID(to), std::move(msg));
  }

  ActorThreadPool *GetActorThreadPool() const { return inner_pool_; }

  ActorMgr();
  ~ActorMgr();

  void Finalize();
  // initialize actor manager resource, do not create inner thread pool by default
  int Initialize(bool use_inner_pool = false, size_t actor_thread_num = 1, size_t max_thread_num = 1,
                 size_t actor_queue_size = kMaxHqueueSize, const std::vector<int> &core_list = {});

  void RemoveActor(const std::string &name);
  ActorReference GetActor(const AID &id);
  const std::string GetUrl(const std::string &protocol = "tcp");
  void AddUrl(const std::string &protocol, const std::string &url);
  void AddIOMgr(const std::string &protocol, const std::shared_ptr<IOMgr> &ioMgr);
  int Send(const AID &to, std::unique_ptr<MessageBase> msg, bool remoteLink = false, bool isExactNotRemote = false);
  AID Spawn(const ActorReference &actor, bool shareThread = true);
  void Terminate(const AID &id);
  void TerminateAll();
  void Wait(const AID &pid);
  inline const std::string &GetDelegate() const { return delegate; }

  inline void SetDelegate(const std::string &d) { delegate = d; }
  void SetActorReady(const ActorReference &actor) const;

  void ChildAfterFork();

  // Quit and reset mailbox for actors after process fork, and prepare to spawn.
  void ResetActorAfterFork(const ActorReference &actor);

 private:
  inline bool IsLocalAddres(const AID &id) {
    if (id.Url() == "" || id.Url().empty() || urls.find(id.Url()) != urls.end()) {
      return true;
    } else {
      return false;
    }
  }
  int EnqueueMessage(const ActorReference actor, std::unique_ptr<MessageBase> msg);
  // in order to avoid being initialized many times
  bool initialized_{false};

  // actor manager support running on inner thread pool,
  // or running on other thread pool created independently externally
  ActorThreadPool *inner_pool_{nullptr};

  // Map of all local spawned and running processes.
  std::map<std::string, ActorReference> actors;
#ifndef MS_COMPILE_IOS
  std::shared_mutex actorsMutex;
#else
  std::mutex actorsMutex;
#endif
  std::map<std::string, std::string> procotols;
  std::set<std::string> urls;
  std::string delegate;
  static ActorMgr actorMgr;
  static std::map<std::string, std::shared_ptr<IOMgr> > ioMgrs;
};  // end of class ActorMgr
};  // end of namespace mindspore
#endif
