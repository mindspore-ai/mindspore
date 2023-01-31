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
#include "aicpu_sharder/aicpu_context.h"

#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "common/kernel_log.h"

namespace {
// current thread context
aicpu::aicpuContext_t g_cur_ctx;
// task monitor context
std::unique_ptr<std::string[]> g_opsname(nullptr);
thread_local uint32_t g_thread_index = UINT32_MAX;
uint32_t g_aicpu_core_cnt = 0;
thread_local std::map<std::string, std::string> g_thread_local_ctx;
thread_local aicpu::streamAndTaskId_t g_stream_and_task_id;
// aicpu run mode
uint32_t g_run_mode = aicpu::AicpuRunMode::THREAD_MODE;

// context info
std::mutex default_mutex;
std::vector<std::map<std::string, std::string>> g_default_thread_ctx;
std::mutex prof_mutex;
std::vector<std::map<std::string, std::string>> g_prof_thread_ctx;
std::mutex debug_mutex;
std::vector<std::map<std::string, std::string>> g_debug_thread_ctx;
std::mutex func_map_mutex;
std::map<uint32_t, std::map<uint32_t, std::function<void(void *)>>> g_func_map;

std::map<std::string, std::string> &GetThreadCtx(aicpu::CtxType type, uint32_t thread_index) {
  if (type == aicpu::CTX_DEBUG) {
    std::unique_lock<std::mutex> mutex(default_mutex);
    if (thread_index >= g_debug_thread_ctx.size()) {
      g_debug_thread_ctx.resize(thread_index + 1);
    }
    return g_debug_thread_ctx[thread_index];
  } else if (type == aicpu::CTX_PROF) {
    std::unique_lock<std::mutex> mutex(prof_mutex);
    if (thread_index >= g_prof_thread_ctx.size()) {
      g_prof_thread_ctx.resize(thread_index + 1);
    }
    return g_prof_thread_ctx[thread_index];
  } else {
    std::unique_lock<std::mutex> mutex(debug_mutex);
    if (thread_index >= g_default_thread_ctx.size()) {
      g_default_thread_ctx.resize(thread_index + 1);
    }
    return g_default_thread_ctx[thread_index];
  }
}
}  // namespace

namespace aicpu {
status_t aicpuSetContext(const aicpuContext_t *const ctx) {
  g_cur_ctx = *ctx;
  return AICPU_ERROR_NONE;
}

status_t aicpuGetContext(aicpuContext_t *ctx) {
  *ctx = g_cur_ctx;
  return AICPU_ERROR_NONE;
}

status_t InitTaskMonitorContext(uint32_t aicpu_core_cnt) {
  if (aicpu_core_cnt == 0) {
    AICPU_LOGE("invalid aicpu core count[%u]", aicpu_core_cnt);
    return AICPU_ERROR_FAILED;
  }
  g_aicpu_core_cnt = aicpu_core_cnt;
  AICPU_LOGI("aicpu core count[%u]", aicpu_core_cnt);
  g_opsname.reset(new (std::nothrow) std::string[aicpu_core_cnt]);
  if (g_opsname == nullptr) {
    AICPU_LOGE("malloc ops name memory for task monitor failed");
    return AICPU_ERROR_FAILED;
  }
  for (uint32_t index = 0; index < aicpu_core_cnt; ++index) {
    g_opsname[index] = "null";
  }
  return AICPU_ERROR_NONE;
}

status_t SetAicpuThreadIndex(uint32_t thread_index) {
  g_thread_index = thread_index;
  return AICPU_ERROR_NONE;
}

uint32_t GetAicpuThreadIndex() { return g_thread_index; }

status_t SetOpname(const std::string &opname) {
  if (g_opsname != nullptr && g_thread_index < g_aicpu_core_cnt) {
    AICPU_LOGI("set op name to %s for thread[%u]", opname.c_str(), g_thread_index);
    g_opsname[g_thread_index] = opname;
    return AICPU_ERROR_NONE;
  }
  // maintenance function, if failed just print event log
  AICPU_LOGEVENT(
    "set op name[%s] failed, thread index[%u] should be less than total aicpu core count[%u],"
    " and ops name array addr[%p] cannot null",
    opname.c_str(), g_thread_index, g_aicpu_core_cnt, g_opsname.get());
  return AICPU_ERROR_NONE;
}

status_t GetOpname(uint32_t thread_index, std::string *opname) {
  *opname = "null";
  if (g_opsname != nullptr && thread_index < g_aicpu_core_cnt) {
    *opname = g_opsname[thread_index];
    return AICPU_ERROR_NONE;
  }
  // maintenance function, if failed just print event log
  AICPU_LOGEVENT(
    "get op name failed, thread index[%u] should be less than total aicpu core count[%u],"
    " and ops name array addr[%p] cannot null",
    g_thread_index, g_aicpu_core_cnt, g_opsname.get());
  return AICPU_ERROR_NONE;
}

status_t SetTaskAndStreamId(uint64_t task_id, uint32_t stream_id) {
  g_stream_and_task_id.task_id = task_id;
  g_stream_and_task_id.stream_id = stream_id;
  AICPU_LOGI("Set task_id:[%lu] and stream_id:[%u] success.", task_id, stream_id);
  return AICPU_ERROR_NONE;
}

status_t GetTaskAndStreamId(uint64_t *task_id, uint32_t *stream_id) {
  *task_id = g_stream_and_task_id.task_id;
  *stream_id = g_stream_and_task_id.stream_id;
  AICPU_LOGI("Get task_id:[%lu] and stream_id:[%u] success.", *task_id, *stream_id);
  return AICPU_ERROR_NONE;
}

status_t SetAicpuRunMode(uint32_t run_mode) {
  g_run_mode = run_mode;
  AICPU_LOGI("Set run_mode:[%u] success.", run_mode);
  return AICPU_ERROR_NONE;
}

status_t GetAicpuRunMode(uint32_t *run_mode) {
  *run_mode = g_run_mode;
  AICPU_LOGI("Get run_mode:[%u] success.", *run_mode);
  return AICPU_ERROR_NONE;
}

status_t SetThreadLocalCtx(const std::string &key, const std::string &value) {
  if (key.empty()) {
    AICPU_LOGE("set thread local context failed, key is empty");
    return AICPU_ERROR_FAILED;
  }
  try {
    g_thread_local_ctx[key] = value;
  } catch (std::exception &e) {
    AICPU_LOGE("set thread local context failed, %s", e.what());
    return AICPU_ERROR_FAILED;
  }
  return AICPU_ERROR_NONE;
}

status_t GetThreadLocalCtx(const std::string &key, std::string *value) { return AICPU_ERROR_NONE; }

status_t RemoveThreadLocalCtx(const std::string &key) {
  auto iter = g_thread_local_ctx.find(key);
  if (iter != g_thread_local_ctx.end()) {
    (void)g_thread_local_ctx.erase(iter);
    return AICPU_ERROR_NONE;
  }
  AICPU_LOGE("remove thread local context failed, no such key[%s]", key.c_str());
  return AICPU_ERROR_FAILED;
}

const std::map<std::string, std::string> &GetAllThreadCtxInfo(aicpu::CtxType type, uint32_t thread_index) {
  AICPU_LOGI("Get all thread ctx info begin, thread index:%u", thread_index);
  auto &ctx = GetThreadCtx(type, thread_index);
  return ctx;
}

status_t RegisterEventCallback(uint32_t event_id, uint32_t subevent_id, std::function<void(void *)> func) {
  std::lock_guard<std::mutex> lock(func_map_mutex);
  std::map<uint32_t, std::function<void(void *)>> &sub_map = g_func_map[event_id];
  auto it = sub_map.insert(std::make_pair(subevent_id, func));
  if (it.second == false) {
    AICPU_LOGE(
      "register event call function failed, repulicate register callback function by event_id[%u] "
      "subevent_id[%u]",
      event_id, subevent_id);
    return AICPU_ERROR_FAILED;
  }
  return AICPU_ERROR_NONE;
}

status_t DoEventCallback(uint32_t event_id, uint32_t subevent_id, void *param) {
  std::lock_guard<std::mutex> lock(func_map_mutex);
  auto iter = g_func_map.find(event_id);
  if (iter == g_func_map.end()) {
    AICPU_LOGE("do event callback function failed, cannot find callback function by event_id[%u] subevent_id[%u]",
               event_id, event_id);
    return AICPU_ERROR_FAILED;
  }

  std::map<uint32_t, std::function<void(void *)>> &sub_map = iter->second;
  auto sub_iter = sub_map.find(subevent_id);
  if (sub_iter == sub_map.end()) {
    AICPU_LOGE("do event callback function failed, cannot find callback function by event_id[%u] subevent_id[%u]",
               event_id, event_id);
    return AICPU_ERROR_FAILED;
  }
  (sub_iter->second)(param);
  // erase func after call
  (void)sub_map.erase(sub_iter);
  return AICPU_ERROR_NONE;
}

status_t UnRegisterCallback(uint32_t event_id, uint32_t subevent_id) {
  std::lock_guard<std::mutex> lock(func_map_mutex);
  auto iter = g_func_map.find(event_id);
  if (iter == g_func_map.end()) {
    AICPU_LOGEVENT(
      "skip unregister event callback function, cannot find callback function by event_id[%u] "
      "subevent_id[%u]",
      event_id, event_id);
    return AICPU_ERROR_NONE;
  }

  std::map<uint32_t, std::function<void(void *)>> &sub_map = iter->second;
  auto sub_iter = sub_map.find(subevent_id);
  if (sub_iter == sub_map.end()) {
    AICPU_LOGEVENT(
      "skip unregister event callback function, cannot find callback function by event_id[%u] "
      "subevent_id[%u]",
      event_id, event_id);
    return AICPU_ERROR_NONE;
  }
  (void)sub_map.erase(sub_iter);
  return AICPU_ERROR_NONE;
}
}  // namespace aicpu

aicpu::status_t SetThreadCtxInfo(aicpu::CtxType type, const std::string &key, const std::string &value) {
  if (key.empty()) {
    AICPU_LOGE("Set thread context failed, context type[%d], key is empty", type);
    return aicpu::AICPU_ERROR_FAILED;
  }

  auto &ctx = GetThreadCtx(type, g_thread_index);
  try {
    ctx[key] = value;
  } catch (std::exception &e) {
    AICPU_LOGE("Set thread context failed, context type[%d], %s", type, e.what());
    return aicpu::AICPU_ERROR_FAILED;
  }
  return aicpu::AICPU_ERROR_NONE;
}

aicpu::status_t GetThreadCtxInfo(aicpu::CtxType type, const std::string &key, std::string *value) {
  if (key.empty()) {
    AICPU_LOGE("Get thread context failed, context type[%d], key is empty", type);
    return aicpu::AICPU_ERROR_FAILED;
  }

  auto &ctx = GetThreadCtx(type, g_thread_index);
  auto iter = ctx.find(key);
  if (iter != ctx.end()) {
    *value = iter->second;
    return aicpu::AICPU_ERROR_NONE;
  }
  AICPU_LOGE("Get thread context failed, context type[%d], no such key[%s]", type, key.c_str());
  return aicpu::AICPU_ERROR_FAILED;
}

aicpu::status_t RemoveThreadCtxInfo(aicpu::CtxType type, const std::string &key) {
  auto &ctx = GetThreadCtx(type, g_thread_index);
  auto iter = ctx.find(key);
  if (iter != ctx.end()) {
    (void)ctx.erase(iter);
    return aicpu::AICPU_ERROR_NONE;
  }
  AICPU_LOGE("Remove thread context failed, context type[%d], no such key[%s]", type, key.c_str());
  return aicpu::AICPU_ERROR_FAILED;
}
