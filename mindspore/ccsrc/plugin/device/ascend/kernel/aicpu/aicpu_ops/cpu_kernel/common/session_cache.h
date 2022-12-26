/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#ifndef AICPU_CONTEXT_COMMON_SESSION_CACHE_H
#define AICPU_CONTEXT_COMMON_SESSION_CACHE_H

#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "cpu_kernel/common/kernel_cache.h"

namespace aicpu {
template <class C>
class SessionCache {
 public:
  static SessionCache<C> &Instance() {
    static SessionCache<C> instance;
    return instance;
  }

  /*
   * run and cache kernel.
   * @param param: kernel context
   * @param session_id: sesson id
   * @param stream_id: stream id
   * @param sess_flag: whether it's a session scene, true use session id, false
   * @param blkdim_info: Op's blkdim_info
   * use stream id
   * @return int32_t: 0 indicates success, while the others fail
   */
  template <class T>
  int32_t RunCpuKernelWithBlock(void *param, uint64_t session_id, uint64_t stream_id, bool sess_flag,
                                struct BlkDimInfo *blkdim_info) {
    std::shared_ptr<KernelCache<C>> kernel = nullptr;
    if (sess_flag) {
      KERNEL_LOG_DEBUG("SessionCache KernelCache from session, id[%llu].", session_id);
      std::unique_lock<std::mutex> lock(session_mutex_);
      int32_t ret = GetOrCreateKernelCache<T>(session_kernel_cache_, session_id, sess_flag, kernel);
      if (ret != 0) {
        return ret;
      }
    } else {
      KERNEL_LOG_DEBUG("SessionCache KernelCache from stream, id[%llu].", stream_id);
      std::unique_lock<std::mutex> lock(stream_mutex_);
      int32_t ret = GetOrCreateKernelCache<T>(stream_kernel_cache_, stream_id, sess_flag, kernel);
      if (ret != 0) {
        return ret;
      }
    }
    return kernel->RunCpuKernelWithBlock(param, blkdim_info);
  }

  /*
   * run and cache kernel.
   * @param param: kernel context
   * @param session_id: sesson id
   * @param stream_id: stream id
   * @param sess_flag: whether it's a session scene, true use session id, false
   * use stream id
   * @return int32_t: 0 indicates success, while the others fail
   */
  template <class T>
  int32_t RunKernel(void *param, uint64_t session_id, uint64_t stream_id, bool sess_flag) {
    std::shared_ptr<KernelCache<C>> kernel = nullptr;
    if (sess_flag) {
      KERNEL_LOG_DEBUG("SessionCache KernelCache from session, id[%llu].", session_id);
      std::unique_lock<std::mutex> lock(session_mutex_);
      int32_t ret = GetOrCreateKernelCache<T>(session_kernel_cache_, session_id, sess_flag, kernel);
      if (ret != 0) {
        return ret;
      }
    } else {
      KERNEL_LOG_DEBUG("SessionCache KernelCache from stream, id[%llu].", stream_id);
      std::unique_lock<std::mutex> lock(stream_mutex_);
      int32_t ret = GetOrCreateKernelCache<T>(stream_kernel_cache_, stream_id, sess_flag, kernel);
      if (ret != 0) {
        return ret;
      }
    }
    return kernel->RunKernel(param);
  }

 private:
  SessionCache() = default;
  ~SessionCache() = default;
  SessionCache(const SessionCache &) = delete;
  SessionCache(SessionCache &&) = delete;
  SessionCache &operator=(const SessionCache &) = delete;
  SessionCache &operator=(SessionCache &&) = delete;

  template <class T>
  int32_t GetOrCreateKernelCache(std::map<uint64_t, std::shared_ptr<KernelCache<C>>> &kernel_map, uint64_t id,
                                 bool sess_flag, std::shared_ptr<KernelCache<C>> &kernel) {
    auto iter = kernel_map.find(id);
    if (iter != kernel_map.end()) {
      KERNEL_LOG_DEBUG("Get kernel from cache success, id[%llu].", id);
      kernel = iter->second;
    } else {
      KernelCache<C> *cache = new (std::nothrow) T();
      if (cache == nullptr) {
        KERNEL_LOG_DEBUG("Create kernel cache failed, id[%llu].", id);
        return -1;
      }
      kernel = std::shared_ptr<KernelCache<C>>(cache);
      int32_t ret = kernel->Init(sess_flag);
      if (ret != 0) {
        return ret;
      }
      kernel_map.insert(std::make_pair(id, kernel));
      KERNEL_LOG_DEBUG("Create kernel cache, id[%llu].", id);
    }
    return 0;
  }

 private:
  std::mutex stream_mutex_;
  std::map<uint64_t, std::shared_ptr<KernelCache<C>>> stream_kernel_cache_;  // key is stream id
  std::mutex session_mutex_;
  std::map<uint64_t, std::shared_ptr<KernelCache<C>>> session_kernel_cache_;  // key is session id
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_SESSION_CACHE_H
