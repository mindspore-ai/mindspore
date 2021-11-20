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

#ifndef AICPU_OPS_AICPU_CONTEXT_H_
#define AICPU_OPS_AICPU_CONTEXT_H_

#include <sys/types.h>
#include <cstdint>
#include <string>
#include <map>
#include <functional>
#include "common/kernel_util.h"

namespace aicpu {
typedef struct {
  uint32_t device_id;  // device id
  uint32_t tsId;       // ts id
  pid_t host_pid;      // host pid
  uint32_t vf_id;      // vf id
} aicpuContext_t;

enum AicpuRunMode : uint32_t {
  PROCESS_PCIE_MODE = 0,    // 1910/1980/1951 dc, with host mode
  PROCESS_SOCKET_MODE = 1,  // MDC
  THREAD_MODE = 2,          // ctrlcpu/minirc/lhisi
  INVALID_MODE,
};

typedef struct {
  uint32_t stream_id;
  uint64_t task_id;
} streamAndTaskId_t;

typedef enum {
  AICPU_ERROR_NONE = 0,    // success
  AICPU_ERROR_FAILED = 1,  // failed
} status_t;

enum CtxType : int32_t { CTX_DEFAULT = 0, CTX_PROF, CTX_DEBUG };

constexpr auto kContextKeyOpName = "opname";
constexpr auto kContextKeyPhaseOneFlag = "phaseOne";
constexpr auto kContextKeyWaitType = "waitType";
constexpr auto kContextKeyWaitId = "waitId";
constexpr auto kContextKeyStartTick = "startTick";
constexpr auto kContextKeyDrvSubmitTick = "drvSubmitTick";
constexpr auto kContextKeyDrvSchedTick = "drvSchedTick";
constexpr auto kContextKeyKernelType = "kernelType";

/**
 * set aicpu context for current thread.
 * @param [in]ctx aicpu context
 * @return status whether this operation success
 */
AICPU_VISIBILITY_API status_t aicpuSetContext(aicpuContext_t *ctx);

/**
 * get aicpu context from current thread.
 * @param [out]ctx aicpu context
 * @return status whether this operation success
 */
AICPU_VISIBILITY_API status_t aicpuGetContext(aicpuContext_t *ctx);

/**
 * init context for task monitor, called in compute process start.
 * @param [in]aicpu_core_cnt aicpu core number
 * @return status whether this operation success
 */
status_t InitTaskMonitorContext(uint32_t aicpu_core_cnt);

/**
 * set aicpu thread index for task monitor, called in thread callback function.
 * @param [in]thread_index aicpu thread index
 * @return status whether this operation success
 */
status_t SetAicpuThreadIndex(uint32_t thread_index);

/**
 * get aicpu thread index.
 * @return uint32
 */
uint32_t GetAicpuThreadIndex();

/**
 * set op name for task monitor.
 * called in libtf_kernels.so(tf op) or libaicpu_processer.so(others) or cpu kernel framework.
 * @param [in]opname op name
 * @return status whether this operation success
 */
status_t __attribute__((weak)) SetOpname(const std::string &opname);

/**
 * get op name for task monitor
 * @param [in]thread_index thread index
 * @param [out]opname op name
 * @return status whether this operation success
 */
status_t GetOpname(uint32_t thread_index, std::string *opname);

/**
 * get task and stream id.
 * @param [in]task_id task id.
 * @param [in]stream_id stream id.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) GetTaskAndStreamId(uint64_t *task_id, uint32_t *stream_id);

/**
 * set task and stream id.
 * @param [in]task_id task id.
 * @param [in]stream_id stream id.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) SetTaskAndStreamId(uint64_t task_id, uint32_t stream_id);

/**
 * set thread local context of key
 * @param [in]key context key
 * @param [in]value context value
 * @return status whether this operation success
 * @note Deprecated from 20201216, Replaced by SetThreadCtxInfo
 */
status_t __attribute__((weak)) SetThreadLocalCtx(const std::string &key, const std::string &value);

/**
 * get thread local context of key
 * @param [in]key context key
 * @param [out]value context value
 * @return status whether this operation success
 * @note Deprecated from 20201216, Replaced by GetThreadCtxInfo
 */
status_t GetThreadLocalCtx(const std::string &key, std::string *value);

/**
 * remove local context of key
 * @param [in]key context key
 * @return status whether this operation success
 * @note Deprecated from 20201216, Replaced by RemoveThreadCtxInfo
 */
status_t RemoveThreadLocalCtx(const std::string &key);

/**
 * get all thread context info of type
 * @param [in]type: ctx type
 * @param [in]thread_index: thread index
 * @return const std::map<std::string, std::string> &: all thread context info
 */
const std::map<std::string, std::string> &GetAllThreadCtxInfo(aicpu::CtxType type, uint32_t thread_index);

/**
 * set run mode.
 * @param [in]run_mode: run mode.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) SetAicpuRunMode(uint32_t run_mode);

/**
 * get run mode.
 * @param [out]run_mode: run mode.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) GetAicpuRunMode(uint32_t *run_mode);

/**
 * Register callback function by event_id and subevent_id
 * @param event_id event id
 * @param subevent_id subevent id
 * @param func call back function
 */
status_t __attribute__((weak))
RegisterEventCallback(uint32_t event_id, uint32_t subevent_id, std::function<void(void *)> func);

/**
 * Do callback function by event_id and subevent_id
 * @param event_id event id
 * @param subevent_id subevent id
 * @param param event param
 */
status_t __attribute__((weak)) DoEventCallback(uint32_t event_id, uint32_t subevent_id, void *param);

/**
 * Unregister callback function by event_id and subevent_id
 * @param event_id event id
 * @param subevent_id subevent id
 */
status_t __attribute__((weak)) UnRegisterCallback(uint32_t event_id, uint32_t subevent_id);
}  // namespace aicpu

extern "C" {
/**
 * set thread context info of type
 * @param [in]type: ctx type
 * @param [in]key: key of context info
 * @param [in]value: value of context info
 * @return status whether this operation success
 */
AICPU_VISIBILITY_API aicpu::status_t SetThreadCtxInfo(aicpu::CtxType type, const std::string &key,
                                                      const std::string &value);

/**
 * get thread context info of type
 * @param [in]type: ctx type
 * @param [in]key: key of context info
 * @param [out]value: value of context info
 * @return status whether this operation success
 */
AICPU_VISIBILITY_API aicpu::status_t GetThreadCtxInfo(aicpu::CtxType type, const std::string &key, std::string *value);

/**
 * remove thread context info of type
 * @param [in]type: ctx type
 * @param [in]key: key of context info
 * @return status whether this operation success
 */
AICPU_VISIBILITY_API aicpu::status_t RemoveThreadCtxInfo(aicpu::CtxType type, const std::string &key);
}

#endif  // AICPU_OPS_AICPU_CONTEXT_H_
