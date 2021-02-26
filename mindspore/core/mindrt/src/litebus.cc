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

#include <cstdlib>
#include <atomic>
#include "mindrt/src/actor/actormgr.h"
#include "mindrt/src/actor/iomgr.h"
#include "litebus.hpp"
#include "include/litebus.h"

extern "C" {
int LitebusInitializeC(const struct LitebusConfig *config) {
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

void LitebusFinalizeC() { mindspore::Finalize(); }
}

namespace mindspore {

namespace local {

static LitebusAddress *g_litebusAddress = new (std::nothrow) LitebusAddress();
static std::atomic_bool g_finalizeLitebusStatus(false);

}  // namespace local

const LitebusAddress &GetLitebusAddress() {
  BUS_OOM_EXIT(local::g_litebusAddress);
  return *local::g_litebusAddress;
}

void SetThreadCount(int threadCount) { ActorMgr::GetActorMgrRef()->Initialize(threadCount); }

class LiteBusExit {
 public:
  LiteBusExit() {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "trace: enter LiteBusExit()---------");
  }
  ~LiteBusExit() {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "trace: enter ~LiteBusExit()---------");
    mindspore::Finalize();
  }
};

int InitializeImp(const std::string &tcpUrl, const std::string &tcpUrlAdv, const std::string &udpUrl,
                  const std::string &udpUrlAdv, int threadCount) {
  ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "litebus starts ......");

  // start actor's thread
  SetThreadCount(threadCount);

  ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "litebus has started.");
  return BUS_OK;
}

int Initialize(const std::string &tcpUrl, const std::string &tcpUrlAdv, const std::string &udpUrl,
               const std::string &udpUrlAdv, int threadCount) {
  static std::atomic_bool initLitebusStatus(false);
  bool inite = false;
  if (initLitebusStatus.compare_exchange_strong(inite, true) == false) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "litebus has been initialized");
    return BUS_OK;
  }

  int result = BUS_OK;
  result = InitializeImp(tcpUrl, tcpUrlAdv, udpUrl, udpUrlAdv, threadCount);
  static LiteBusExit busExit;

  return result;
}

AID Spawn(ActorReference actor, bool sharedThread, bool start) {
  if (actor == nullptr) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "Actor is nullptr.");
    BUS_EXIT("Actor is nullptr.");
  }

  if (local::g_finalizeLitebusStatus.load() == true) {
    return actor->GetAID();
  } else {
    return ActorMgr::GetActorMgrRef()->Spawn(actor, sharedThread, start);
  }
}
void SetActorStatus(const AID &actor, bool start) { ActorMgr::GetActorMgrRef()->SetActorStatus(actor, start); }

void Await(const ActorReference &actor) { ActorMgr::GetActorMgrRef()->Wait(actor->GetAID()); }

void Await(const AID &actor) { ActorMgr::GetActorMgrRef()->Wait(actor); }

// brief get actor with aid
ActorReference GetActor(const AID &actor) { return ActorMgr::GetActorMgrRef()->GetActor(actor); }

void Terminate(const AID &actor) { ActorMgr::GetActorMgrRef()->Terminate(actor); }

void TerminateAll() { mindspore::ActorMgr::GetActorMgrRef()->TerminateAll(); }

void Finalize() {
  bool inite = false;
  if (local::g_finalizeLitebusStatus.compare_exchange_strong(inite, true) == false) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "litebus has been Finalized.");
    return;
  }

  ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "litebus starts to finalize.");
  mindspore::ActorMgr::GetActorMgrRef()->Finalize();

  ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "litebus has been finalized.");
  // flush the log in cache to disk before exiting.
  FlushHLogCache();
}

void SetDelegate(const std::string &delegate) { mindspore::ActorMgr::GetActorMgrRef()->SetDelegate(delegate); }

static HARES_LOG_PID g_busLogPid = 1;
void SetLogPID(HARES_LOG_PID pid) {
  ICTSBASE_LOG1(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, pid, "Set LiteBus log PID: %u", pid);
  g_busLogPid = pid;
}
HARES_LOG_PID GetLogPID() { return g_busLogPid; }

static int g_httpKmsgEnable = -1;
void SetHttpKmsgFlag(int flag) {
  ICTSBASE_LOG1(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "Set LiteBus http message format:%d", flag);
  g_httpKmsgEnable = flag;
}

int GetHttpKmsgFlag() { return g_httpKmsgEnable; }

}  // namespace mindspore
