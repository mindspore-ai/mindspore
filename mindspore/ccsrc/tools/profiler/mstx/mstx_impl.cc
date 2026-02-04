/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "tools/profiler/mstx/mstx_impl.h"
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "tools/profiler/profiling.h"
#include "tools/profiler/utils.h"

namespace mindspore {
namespace profiler {
namespace {
bool IsMsptiEnableImpl() {
  bool ret = false;
  const char *envVal = std::getenv("LD_PRELOAD");
  if (envVal == nullptr) {
    return ret;
  }
  static const std::string soName = "libmspti.so";
  std::stringstream ss(envVal);
  std::string path;
  while (std::getline(ss, path, ':')) {
    path = mindspore::profiler::Utils::RealPath(path);
    if ((path.size() > soName.size()) && (path.substr(path.size() - soName.size()) == soName)) {
      ret = true;
      break;
    }
  }
  return ret;
}

bool IsMsleaksEnableImpl() {
  bool ret = false;
  const char *envVal = std::getenv("LD_PRELOAD");
  if (envVal == nullptr) {
    return ret;
  }
  static const std::string soName = "libascend_kernel_hook.so";
  std::stringstream ss(envVal);
  std::string path;
  while (std::getline(ss, path, ':')) {
    path = mindspore::profiler::Utils::RealPath(path);
    if ((path.size() > soName.size()) && (path.substr(path.size() - soName.size()) == soName)) {
      ret = true;
      break;
    }
  }
  return ret;
}

void InitMstxApis() {
  const char *envVal = std::getenv("ASCEND_HOME_PATH");
  if (envVal == nullptr) {
    return;
  }
  std::string ascendPath = std::string(envVal);
  ascendPath = mindspore::profiler::Utils::RealPath(ascendPath);
  if (ascendPath.empty()) {
    return;
  }
  LoadMstxApiSymbol(ascendPath);
}

}  // namespace

static std::mutex g_mstxRangeIdsMtx;
static std::unordered_map<uint64_t, uint64_t> g_mstxInfoTime;

MstxImpl::MstxImpl() {
  InitMstxApis();
  isMstxSupport_ = IsCannSupportMstxApi();
  isMstxDomainSupport_ = IsCannSupportMstxDomainApi();
}

void MstxImpl::ProfEnable() {
  MS_LOG(INFO) << "enable mstx";
  isProfEnable_.store(true);
}

void MstxImpl::ProfDisable() {
  MS_LOG(INFO) << "disable mstx";
  isProfEnable_.store(false);
}

bool MstxImpl::IsMsptiEnable() {
  static bool isEnable = IsMsptiEnableImpl();
  return isEnable;
}

bool MstxImpl::IsMsleaksEnable() {
  static bool isEnable = IsMsleaksEnableImpl();
  return isEnable;
}

bool MstxImpl::IsEnable() { return isProfEnable_.load() || IsMsptiEnable(); }

bool MstxImpl::IsSupportMstxApi(bool withDomain) { return withDomain ? isMstxDomainSupport_ : isMstxSupport_; }

void MstxImpl::MarkAImpl(const std::string &domainName, const char *message, void *stream) {
  if (!IsEnable()) {
    return;
  }
  if (!IsDomainEnable(domainName)) {
    MS_LOG(WARNING) << "Disabled domain: " << domainName;
    return;
  }
  mstxDomainHandle_t domainHandle = GetDomainHandle(domainName);
  // cppcheck-suppress *
  if (!IsSupportMstxApi(domainHandle != nullptr)) {
    return;
  }
  uint64_t startTime = mindspore::profiler::Utils::getClockSyscnt();
  // cppcheck-suppress *
  if (domainHandle) {
    CALL_MSTX_API(mstxDomainMarkA, domainHandle, message, stream);
  } else {
    CALL_MSTX_API(mstxMarkA, message, stream);
  }
  uint64_t endTime = mindspore::profiler::Utils::getClockSyscnt();
  mindspore::profiler::CollectHostInfo(MSTX_MODULE, MSTX_EVENT, MSTX_STAGE_MARK, startTime, endTime);
}

uint64_t MstxImpl::RangeStartAImpl(const std::string &domainName, const char *message, void *stream) {
  uint64_t txTaskId = 0;
  if (!IsEnable()) {
    return txTaskId;
  }
  if (!IsDomainEnable(domainName)) {
    MS_LOG(WARNING) << "Disabled domain: " << domainName;
    return 0;
  }
  mstxDomainHandle_t domainHandle = GetDomainHandle(domainName);
  // cppcheck-suppress *
  if (!IsSupportMstxApi(domainHandle != nullptr)) {
    return txTaskId;
  }
  uint64_t startTime = mindspore::profiler::GetClockSyscnt();
  // cppcheck-suppress *
  if (domainHandle) {
    txTaskId = CALL_MSTX_API(mstxDomainRangeStartA, domainHandle, message, stream);
  } else {
    txTaskId = CALL_MSTX_API(mstxRangeStartA, message, stream);
  }
  if (txTaskId == 0) {
    return txTaskId;
  }
  {
    std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
    g_mstxInfoTime.emplace(txTaskId, startTime);
  }
  return txTaskId;
}

void MstxImpl::RangeEndImpl(const std::string &domainName, uint64_t txTaskId) {
  if (!IsEnable()) {
    return;
  }
  if (!IsDomainEnable(domainName)) {
    MS_LOG(WARNING) << "Disabled domain: " << domainName;
    return;
  }
  mstxDomainHandle_t domainHandle = GetDomainHandle(domainName);
  // cppcheck-suppress *
  if (!IsSupportMstxApi(domainHandle != nullptr)) {
    return;
  }
  if (txTaskId == 0) {
    return;
  }
  uint64_t startTime = 0;
  {
    std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
    auto iter = g_mstxInfoTime.find(txTaskId);
    if (iter == g_mstxInfoTime.end()) {
      MS_LOG(WARNING) << "Failed to find range start time for input range end id " << txTaskId;
      return;
    }
    startTime = iter->second;
    g_mstxInfoTime.erase(iter);
  }
  // cppcheck-suppress *
  if (domainHandle) {
    CALL_MSTX_API(mstxDomainRangeEnd, domainHandle, txTaskId);
  } else {
    CALL_MSTX_API(mstxRangeEnd, txTaskId);
  }
  uint64_t endTime = mindspore::profiler::GetClockSyscnt();
  mindspore::profiler::CollectHostInfo(MSTX_MODULE, MSTX_EVENT, MSTX_STAGE_RANGE, startTime, endTime);
}

mstxDomainHandle_t MstxImpl::DomainCreateAImpl(const char *domainName) {
  if (!IsSupportMstxApi(true)) {
    return nullptr;
  }
  std::string nameStr = std::string(domainName);
  auto iter = domains_.find(nameStr);
  if (iter != domains_.end()) {
    return iter->second;
  }
  mstxDomainHandle_t handle = CALL_MSTX_API(mstxDomainCreateA, domainName);
  if (handle != nullptr) {
    domains_.emplace(nameStr, handle);
  }
  return handle;
}

void MstxImpl::DomainDestroyImpl(mstxDomainHandle_t domain) {
  if (!IsSupportMstxApi(true)) {
    return;
  }
  if (domain == nullptr) {
    return;
  }
  CALL_MSTX_API(mstxDomainDestroy, domain);
}

void MstxImpl::MemRegionsRegisterImpl(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc) {
  if (domain == nullptr) {
    return;
  }
  if (!IsSupportMstxApi(true)) {
    return;
  }
  CALL_MSTX_API(mstxMemRegionsRegister, domain, desc);
}

void MstxImpl::MemRegionsUnregisterImpl(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc) {
  if (domain == nullptr) {
    return;
  }
  if (!IsSupportMstxApi(true)) {
    return;
  }
  CALL_MSTX_API(mstxMemRegionsUnregister, domain, desc);
}

mstxMemHeapHandle_t MstxImpl::MemHeapRegisterImpl(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc) {
  if (domain == nullptr) {
    return nullptr;
  }
  if (!IsSupportMstxApi(true)) {
    return nullptr;
  }
  return CALL_MSTX_API(mstxMemHeapRegister, domain, desc);
}

void MstxImpl::SetDomainImpl(const std::vector<std::string> &domainInclude,
                             const std::vector<std::string> &domainExclude) {
  if (!IsSupportMstxApi(true)) {
    return;
  }
  domainInclude_ = domainInclude;
  domainExclude_ = domainExclude;
}

bool MstxImpl::IsDomainEnable(const std::string &domainName) {
  if (domainInclude_.empty()) {
    if (std::find(domainExclude_.begin(), domainExclude_.end(), domainName) != domainExclude_.end()) {
      return false;
    }
    return true;
  }
  if (std::find(domainInclude_.begin(), domainInclude_.end(), domainName) != domainInclude_.end()) {
    return true;
  }
  return false;
}

mstxDomainHandle_t MstxImpl::GetDomainHandle(const std::string &domainName) {
  if (domains_.find(domainName) != domains_.end()) {
    return domains_[domainName];
  }
  if (domainName == MSTX_DOMAIN_DEFAULT) {
    return nullptr;
  }
  return DomainCreateAImpl(domainName.c_str());
}

MstxRangeGuardImpl::MstxRangeGuardImpl(const char *message, const char *domain, void *stream)
    : id_(0), enabled_(MstxImpl::GetInstance().IsEnable()), domain_(domain) {
  if (enabled_) {
    MSTX_START(id_, message, stream, domain_);
  }
}

MstxRangeGuardImpl::~MstxRangeGuardImpl() {
  if (enabled_) {
    MSTX_END(id_, domain_);
  }
}

}  // namespace profiler
}  // namespace mindspore
