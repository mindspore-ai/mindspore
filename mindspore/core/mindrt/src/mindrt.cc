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

#include <atomic>
#include "src/actor/actormgr.h"
#include "src/actor/iomgr.h"
#include "include/mindrt.hpp"
#include "include/mindrt.h"

extern "C" {
int MindrtInitializeC(const struct MindrtConfig *config) {
  if (config == nullptr) {
    return -1;
  }

  if (config->threadCount == 0) {
    return -1;
  }

  if (config->httpKmsgFlag != 0 && config->httpKmsgFlag != 1) {
    return -1;
  }
  mindspore::SetHttpKmsgFlag(config->httpKmsgFlag);

  return mindspore::Initialize(std::string(config->tcpUrl), std::string(config->tcpUrlAdv), std::string(config->udpUrl),
                               std::string(config->udpUrlAdv), config->threadCount);
}

void MindrtFinalizeC() { mindspore::Finalize(); }
}

namespace mindspore {

namespace local {

static MindrtAddress g_mindrtAddress = MindrtAddress();
static std::atomic_bool g_finalizeMindrtStatus(false);

}  // namespace local

const MindrtAddress &GetMindrtAddress() { return local::g_mindrtAddress; }

class MindrtExit {
 public:
  MindrtExit() { MS_LOG(DEBUG) << "trace: enter MindrtExit()."; }
  ~MindrtExit() {
    MS_LOG(DEBUG) << "trace: enter ~MindrtExit().";
    mindspore::Finalize();
  }
};

int InitializeImp(const std::string &tcpUrl, const std::string &tcpUrlAdv, const std::string &udpUrl,
                  const std::string &udpUrlAdv, int threadCount) {
  MS_LOG(DEBUG) << "mindrt starts.";
  auto ret = ActorMgr::GetActorMgrRef()->Initialize();
  MS_LOG(DEBUG) << "mindrt has started.";
  return ret;
}

int Initialize(const std::string &tcpUrl, const std::string &tcpUrlAdv, const std::string &udpUrl,
               const std::string &udpUrlAdv, int threadCount) {
  /* support repeat initialize  */
  int result = InitializeImp(tcpUrl, tcpUrlAdv, udpUrl, udpUrlAdv, threadCount);
  static MindrtExit mindrtExit;

  return result;
}

AID Spawn(const ActorReference actor, bool sharedThread) {
  if (actor == nullptr) {
    MS_LOG(ERROR) << "Actor is nullptr.";
    MINDRT_EXIT("Actor is nullptr.");
  }

  if (local::g_finalizeMindrtStatus.load() == true) {
    return actor->GetAID();
  } else {
    return ActorMgr::GetActorMgrRef()->Spawn(actor, sharedThread);
  }
}

void Await(const ActorReference &actor) { ActorMgr::GetActorMgrRef()->Wait(actor->GetAID()); }

void Await(const AID &actor) { ActorMgr::GetActorMgrRef()->Wait(actor); }

// brief get actor with aid
ActorReference GetActor(const AID &actor) { return ActorMgr::GetActorMgrRef()->GetActor(actor); }

void Terminate(const AID &actor) { ActorMgr::GetActorMgrRef()->Terminate(actor); }

void TerminateAll() { mindspore::ActorMgr::GetActorMgrRef()->TerminateAll(); }

void Finalize() {
  bool inite = false;
  if (local::g_finalizeMindrtStatus.compare_exchange_strong(inite, true) == false) {
    MS_LOG(DEBUG) << "mindrt has been Finalized.";
    return;
  }

  MS_LOG(DEBUG) << "mindrt starts to finalize.";
  mindspore::ActorMgr::GetActorMgrRef()->Finalize();

  MS_LOG(DEBUG) << "mindrt has been finalized.";
  // flush the log in cache to disk before exiting.
  FlushHLogCache();
}

void SetDelegate(const std::string &delegate) { mindspore::ActorMgr::GetActorMgrRef()->SetDelegate(delegate); }

static int g_mindrtLogPid = 1;
void SetLogPID(int pid) {
  MS_LOG(DEBUG) << "Set Mindrt log PID:" << pid;
  g_mindrtLogPid = pid;
}
int GetLogPID() { return g_mindrtLogPid; }

static int g_httpKmsgEnable = -1;
void SetHttpKmsgFlag(int flag) {
  MS_LOG(DEBUG) << "Set Mindrt http message format:" << flag;
  g_httpKmsgEnable = flag;
}

int GetHttpKmsgFlag() { return g_httpKmsgEnable; }

}  // namespace mindspore
