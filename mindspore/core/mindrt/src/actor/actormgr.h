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

#include <set>
#include <utility>
#include <map>
#include <memory>
#include <string>
#include "actor/actorthread.h"

namespace mindspore {
class ActorBase;
class IOMgr;
class ActorMgr {
 public:
  static inline std::shared_ptr<ActorMgr> &GetActorMgrRef() { return actorMgr; }

  static std::shared_ptr<IOMgr> &GetIOMgrRef(const std::string &protocol = "tcp");

  static inline std::shared_ptr<IOMgr> &GetIOMgrRef(const AID &to) { return GetIOMgrRef(to.GetProtocol()); }

  static void Receive(std::unique_ptr<MessageBase> &&msg) {
    auto to = msg->To().Name();
    (void)ActorMgr::GetActorMgrRef()->Send(to, std::move(msg));
  }

  ActorMgr();
  ~ActorMgr();

  void Finalize();
  void Initialize(int threadCount);
  void RemoveActor(const std::string &name);
  ActorReference GetActor(const AID &id);
  const std::string GetUrl(const std::string &protocol = "tcp");
  void AddUrl(const std::string &protocol, const std::string &url);
  void AddIOMgr(const std::string &protocol, const std::shared_ptr<IOMgr> &ioMgr);
  int Send(const AID &to, std::unique_ptr<MessageBase> msg, bool remoteLink = false, bool isExactNotRemote = false);
  AID Spawn(ActorReference &actor, bool shareThread = true, bool start = true);
  void Terminate(const AID &id);
  void TerminateAll();
  void Wait(const AID &pid);
  inline const std::string &GetDelegate() const { return delegate; }

  inline void SetDelegate(const std::string &d) { delegate = d; }
  inline void SetActorReady(const std::shared_ptr<ActorBase> &actor) { threadPool.EnqueReadyActor(actor); }
  void SetActorStatus(const AID &pid, bool start);

 private:
  inline bool IsLocalAddres(const AID &id) {
    if (id.Url() == "" || id.Url().empty() || urls.find(id.Url()) != urls.end()) {
      return true;
    } else {
      return false;
    }
  }
  // Map of all local spawned and running processes.
  std::map<std::string, ActorReference> actors;
  std::mutex actorsMutex;
  ActorThread threadPool;
  std::map<std::string, std::string> procotols;
  std::set<std::string> urls;
  std::string delegate;
  static std::shared_ptr<ActorMgr> actorMgr;
  static std::map<std::string, std::shared_ptr<IOMgr> > ioMgrs;
};  // end of class ActorMgr
};  // end of namespace mindspore
#endif
