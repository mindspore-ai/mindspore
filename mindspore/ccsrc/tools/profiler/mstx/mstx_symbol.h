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

#ifndef MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXSYMBOL_H_
#define MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXSYMBOL_H_
#include <string>
#include "utils/dlopen_macro.h"
#include "include/utils/callback.h"

namespace mindspore {
namespace profiler {
struct mstxDomainRegistration_st {};
typedef struct mstxDomainRegistration_st mstxDomainRegistration_t;
typedef mstxDomainRegistration_t *mstxDomainHandle_t;

struct mstxMemHeap_st;
typedef struct mstxMemHeap_st mstxMemHeap_t;
typedef mstxMemHeap_t *mstxMemHeapHandle_t;

struct mstxMemRegion_st;
typedef struct mstxMemRegion_st mstxMemRegion_t;
typedef mstxMemRegion_t *mstxMemRegionHandle_t;

typedef enum mstxMemType {
  MSTX_MEM_TYPE_VIRTUAL_ADDRESS = 0,
} mstxMemType;

typedef struct mstxMemVirtualRangeDesc_t {
  uint32_t device_id;
  void const *ptr;
  int64_t size;
} mstxMemVirtualRangeDesc_t;

// region
typedef struct mstxMemRegionsRegisterBatch_t {
  mstxMemHeapHandle_t heap;
  mstxMemType regionType;
  size_t regionCount;
  void const *regionDescArray;
  mstxMemRegionHandle_t *regionHandleArrayOut;
} mstxMemRegionsRegisterBatch_t;

typedef enum mstxMemRegionRefType {
  MSTX_MEM_REGION_REF_TYPE_POINTER = 0,
  MSTX_MEM_REGION_REF_TYPE_HANDLE
} mstxMemRegionRefType;

typedef struct mstxMemRegionRef_t {
  mstxMemRegionRefType refType;
  union {
    void const *pointer;
    mstxMemRegionHandle_t handle;
  };
} mstxMemRegionRef_t;

// unregion
typedef struct mstxMemRegionsUnregisterBatch_t {
  size_t refCount;
  mstxMemRegionRef_t const *refArray;
} mstxMemRegionsUnregisterBatch_t;

typedef enum mstxMemHeapUsageType {
  MSTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR = 0,
} mstxMemHeapUsageType;

// heap
typedef struct mstxMemHeapDesc_t {
  mstxMemHeapUsageType usage;
  mstxMemType type;
  void const *typeSpecificDesc;
} mstxMemHeapDesc_t;

ORIGIN_METHOD(mstxMarkA, void, const char *, void *)
ORIGIN_METHOD(mstxRangeStartA, uint64_t, const char *, void *)
ORIGIN_METHOD(mstxRangeEnd, void, uint64_t)
ORIGIN_METHOD(mstxDomainCreateA, mstxDomainHandle_t, const char *)
ORIGIN_METHOD(mstxDomainDestroy, void, mstxDomainHandle_t)
ORIGIN_METHOD(mstxDomainMarkA, void, mstxDomainHandle_t, const char *, void *)
ORIGIN_METHOD(mstxDomainRangeStartA, uint64_t, mstxDomainHandle_t, const char *, void *)
ORIGIN_METHOD(mstxDomainRangeEnd, void, mstxDomainHandle_t, uint64_t)
ORIGIN_METHOD(mstxMemRegionsRegister, void, mstxDomainHandle_t, mstxMemRegionsRegisterBatch_t const *)
ORIGIN_METHOD(mstxMemRegionsUnregister, void, mstxDomainHandle_t, mstxMemRegionsUnregisterBatch_t const *)
ORIGIN_METHOD(mstxMemHeapRegister, mstxMemHeapHandle_t, mstxDomainHandle_t, mstxMemHeapDesc_t const *)

extern mstxMarkAFunObj mstxMarkA_;
extern mstxRangeStartAFunObj mstxRangeStartA_;
extern mstxRangeEndFunObj mstxRangeEnd_;
extern mstxDomainCreateAFunObj mstxDomainCreateA_;
extern mstxDomainDestroyFunObj mstxDomainDestroy_;
extern mstxDomainMarkAFunObj mstxDomainMarkA_;
extern mstxDomainRangeStartAFunObj mstxDomainRangeStartA_;
extern mstxDomainRangeEndFunObj mstxDomainRangeEnd_;
extern mstxMemRegionsRegisterFunObj mstxMemRegionsRegister_;
extern mstxMemRegionsUnregisterFunObj mstxMemRegionsUnregister_;
extern mstxMemHeapRegisterFunObj mstxMemHeapRegister_;

void LoadMstxApiSymbol(const std::string &ascend_path);
bool IsCannSupportMstxApi();
bool IsCannSupportMstxDomainApi();

template <typename Function, typename... Args>
auto RunMstxApi(Function f, const char *file, int line, const char *call_f, const char *func_name, Args... args) {
  MS_LOG(DEBUG) << "Call mstx api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line;
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    if (ret == 0) {
      static auto fail_cb =
        GET_COMMON_CALLBACK(RunFailCallback, void, const char *, int, const char *, const std::string &, bool);
      if (fail_cb != nullptr) {
        fail_cb(file, line, call_f, func_name, true);
      }
      MS_LOG(INFO) << "Call mstx api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line
                   << " failed, return val [" << ret << "].";
    }
    return ret;
  } else {
    return f(args...);
  }
}

#define CALL_MSTX_API(func_name, ...) \
  RunMstxApi(func_name##_, FILE_NAME, __LINE__, __FUNCTION__, #func_name, ##__VA_ARGS__)

}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXSYMBOL_H_
