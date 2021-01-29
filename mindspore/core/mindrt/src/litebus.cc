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

#include "mindrt/src/actor/sysmgr_actor.h"
#include "mindrt/src/actor/actormgr.h"
#include "mindrt/src/actor/iomgr.h"

//#include "utils/os_utils.hpp"

#include "litebus.hpp"
#include "timer/timertools.h"
#include "litebus.h"

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

constexpr auto LITEBUSTHREADMIN = 3;
constexpr auto LITEBUSTHREADMAX = 100;
constexpr auto LITEBUSTHREADS = 10;

constexpr auto SYSMGR_TIMER_DURATION = 600000;

namespace mindspore {

namespace local {

static LitebusAddress *g_litebusAddress = new (std::nothrow) LitebusAddress();
static std::atomic_bool g_finalizeLitebusStatus(false);

}  // namespace local

const LitebusAddress &GetLitebusAddress() {
  BUS_OOM_EXIT(local::g_litebusAddress);
  return *local::g_litebusAddress;
}

bool SetServerIo(std::shared_ptr<mindspore::IOMgr> &io, std::string &advertiseUrl, const std::string &protocol,
                 const std::string &url) {
#if 0
     if (protocol == "tcp") {
        size_t index = advertiseUrl.find("://");
        if (index != std::string::npos) {
            advertiseUrl = advertiseUrl.substr(index + URL_PROTOCOL_IP_SEPARATOR.size());
        }
        ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "create tcp iomgr. (%s)",
                            "Url=%s,advertiseUrl=%s", url.c_str(), advertiseUrl.c_str());
        if (local::g_litebusAddress == nullptr) {
            ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG,
                          "Couldn't allocate memory for LitebusAddress");
            return false;
        }
        local::g_litebusAddress->scheme = protocol;
        local::g_litebusAddress->ip = AID("test@" + advertiseUrl).GetIp();
        local::g_litebusAddress->port = AID("test@" + advertiseUrl).GetPort();

#ifdef HTTP_ENABLED
        mindspore::HttpIOMgr::EnableHttp();
#endif
        io.reset(new (std::nothrow) mindspore::TCPMgr());
        if (io == nullptr) {
            ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG,
                          "Couldn't allocate memory for TCPMgr");
            return false;
        }
    }
#ifdef UDP_ENABLED
    else if (protocol == "udp") {
        ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "create udp iomgr. (%s)",
                            "Url=%s,advertiseUrl=%s", url.c_str(), advertiseUrl.c_str());

        io.reset(new (std::nothrow) mindspore::UDPMgr());
        if (io == nullptr) {
            ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG,
                          "Couldn't allocate memory for UDPMgr");
            return false;
        }
    }
#endif
    else {

        ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "unsupported protocol. (%s)",
                            "%s", protocol.c_str());
        return false;
    }
#endif
  return true;
}

void SetThreadCount(int threadCount) {
  int tmpThreadCount = LITEBUSTHREADS;
  ActorMgr::GetActorMgrRef()->Initialize(tmpThreadCount);
}

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
  signal(SIGPIPE, SIG_IGN);

  if (!TimerTools::Initialize()) {
    ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "Failed to initialize timer tools");
    return BUS_ERROR;
  }

  // start actor's thread
  SetThreadCount(threadCount);

  mindspore::Spawn(std::make_shared<SysMgrActor>(SYSMGR_ACTOR_NAME, SYSMGR_TIMER_DURATION));

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
  TimerTools::Finalize();

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
