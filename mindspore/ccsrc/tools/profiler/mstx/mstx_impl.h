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

#ifndef MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXIMPL_H_
#define MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXIMPL_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include "include/utils/visible.h"
#include "tools/profiler/mstx/mstx_symbol.h"

namespace mindspore {
namespace profiler {

const char MSTX_MODULE[] = "Ascend";
const char MSTX_EVENT[] = "Mstx";
const char MSTX_STAGE_MARK[] = "Mark";
const char MSTX_STAGE_RANGE[] = "Range";

const char MSTX_DOMAIN_COMMUNICATION[] = "communication";
const char MSTX_DOMAIN_MODEL_PREPARATION[] = "model_preparation";
const char MSTX_DOMAIN_DEFAULT[] = "default";
const char MSTX_DOMAIN_MSLEAKS[] = "mindsporeMemPool";
const char MSTX_GETNEXT[] = "GetNext";

class PROFILER_EXPORT MstxImpl {
 public:
  MstxImpl();
  ~MstxImpl() = default;

  static MstxImpl &GetInstance() {
    static MstxImpl instance;
    return instance;
  }

  void MarkAImpl(const std::string &domainName, const char *message, void *stream);
  uint64_t RangeStartAImpl(const std::string &domainName, const char *message, void *stream);
  void RangeEndImpl(const std::string &domainName, uint64_t txTaskId);
  mstxDomainHandle_t DomainCreateAImpl(const char *domainName);
  void DomainDestroyImpl(mstxDomainHandle_t domain);
  void SetDomainImpl(const std::vector<std::string> &domainInclude, const std::vector<std::string> &domainExclude);

  void MemRegionsRegisterImpl(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);
  void MemRegionsUnregisterImpl(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);
  mstxMemHeapHandle_t MemHeapRegisterImpl(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);

  void ProfEnable();
  void ProfDisable();
  bool IsEnable();
  bool IsMsleaksEnable();

 private:
  bool IsMsptiEnable();
  bool IsSupportMstxApi(bool withDomain);
  bool IsDomainEnable(const std::string &domainName);
  mstxDomainHandle_t GetDomainHandle(const std::string &domainName);

 private:
  std::atomic<bool> isProfEnable_{false};
  bool isMstxSupport_{false};
  bool isMstxDomainSupport_{false};
  std::mutex domainMtx_;
  std::unordered_map<std::string, mstxDomainHandle_t> domains_;
  std::vector<std::string> domainInclude_;
  std::vector<std::string> domainExclude_;
};

struct PROFILER_EXPORT MstxRangeGuardImpl {
  uint64_t id_;
  bool enabled_;
  const char *domain_;

  explicit MstxRangeGuardImpl(const char *message, const char *domain, void *stream = nullptr);
  ~MstxRangeGuardImpl();

  MstxRangeGuardImpl(const MstxRangeGuardImpl &) = delete;
  MstxRangeGuardImpl &operator=(const MstxRangeGuardImpl &) = delete;
};

// MSTX profiling macros for performance measurement.
// Use MSTX_RANGE_GUARD for entire function profiling with minimal code changes.
// Use MSTX_START/MSTX_END/MSTX_START_WITHOUT_DOMAIN/MSTX_END_WITHOUT_DOMAIN for time-sensitive code segments (e.g.,
// operator dispatch) to minimize profiling overhead.

#define MSTX_START(rangeId, message, stream, domainName)                                                 \
  do {                                                                                                   \
    mindspore::profiler::MstxImpl::GetInstance().DomainCreateAImpl(domainName);                          \
    rangeId = mindspore::profiler::MstxImpl::GetInstance().RangeStartAImpl(domainName, message, stream); \
  } while (0)

#define MSTX_END(rangeId, domainName)                                               \
  do {                                                                              \
    mindspore::profiler::MstxImpl::GetInstance().DomainCreateAImpl(domainName);     \
    mindspore::profiler::MstxImpl::GetInstance().RangeEndImpl(domainName, rangeId); \
  } while (0)

#define MSTX_START_WITHOUT_DOMAIN(rangeId, message, stream)                                                          \
  do {                                                                                                               \
    rangeId = mindspore::profiler::MstxImpl::GetInstance().RangeStartAImpl(mindspore::profiler::MSTX_DOMAIN_DEFAULT, \
                                                                           message, stream);                         \
  } while (0)

#define MSTX_END_WITHOUT_DOMAIN(rangeId)                                                                          \
  do {                                                                                                            \
    mindspore::profiler::MstxImpl::GetInstance().RangeEndImpl(mindspore::profiler::MSTX_DOMAIN_DEFAULT, rangeId); \
  } while (0)

#define MSTX_RANGE_GUARD(message, stream, domain) \
  mindspore::profiler::MstxRangeGuardImpl mstx_guard_##__LINE__(message, domain, stream)

}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXIMPL_H_
