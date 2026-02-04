/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CACHE_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CACHE_H_

#include <string>
#include <vector>
#include <utility>
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"

namespace mindspore::device::ascend {
typedef aclOpExecutor *(*GetExecCache)(uint64_t, uint64_t *);
typedef void (*InitCacheThreadLocal)();
typedef void (*UnInitCacheThreadLocal)();
typedef void (*SetHashKey)(uint64_t);
typedef bool (*CanUseCache)(const char *);

constexpr int g_hash_buf_size = 16384;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern BACKEND_EXPORT thread_local char g_hash_buf[g_hash_buf_size];
extern BACKEND_EXPORT thread_local int g_hash_offset;
extern BACKEND_EXPORT bool cache_unavailable_first_print;

inline void UninitCacheThreadLocal() {
  static const auto uninit_cache_thread_local = device::ascend::GetOpApiFunc("UnInitPTACacheThreadLocal");
  UnInitCacheThreadLocal uninit_cache_thread_local_func =
    reinterpret_cast<UnInitCacheThreadLocal>(uninit_cache_thread_local);
  if (uninit_cache_thread_local_func) {
    uninit_cache_thread_local_func();
  }
}

inline int32_t SetExecutorRepeatable(const std::string &workspace_api_name, aclOpExecutor *executor) {
  int32_t repeat_ret = 0;
  static const auto aclSetAclOpExecutorRepeatable = reinterpret_cast<device::ascend::_aclSetAclOpExecutorRepeatable>(
    device::ascend::GetOpApiFunc("aclSetAclOpExecutorRepeatable"));
  if (aclSetAclOpExecutorRepeatable == nullptr) {
    repeat_ret = -1;
    if (cache_unavailable_first_print) {
      MS_LOG(WARNING) << "The aclSetAclOpExecutorRepeatable is unavailable, which results in aclnn cache miss.";
    }
    cache_unavailable_first_print = false;
  } else {
    repeat_ret = aclSetAclOpExecutorRepeatable(executor);
    if (repeat_ret != 0) {
      MS_LOG(INFO) << workspace_api_name << " don't support cache, repeat_ret is " << repeat_ret;
      MS_VLOG(VL_ACLNN_OP) << workspace_api_name << " don't support cache, repeat_ret is " << repeat_ret;
    }
  }
  return repeat_ret;
}

inline void MemcpyToBuf(const void *data_expression, size_t size_expression) {
  if (size_expression == 0) {
    return;
  }
  if (MS_UNLIKELY(static_cast<uint64_t>(g_hash_offset) > SIZE_MAX - size_expression)) {
    MS_LOG(ERROR) << "Hash buf is overflow.";
    return;
  }
  if (g_hash_offset + size_expression >= g_hash_buf_size) {
    g_hash_offset = g_hash_buf_max_size;
    return;
  }
  auto ret = memcpy_sp(g_hash_buf + g_hash_offset, g_hash_buf_size - g_hash_offset, data_expression, size_expression);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Failed to memcpy!";
  }
  g_hash_offset += size_expression;
}

// Add core num to hash
BACKEND_EXPORT void GatherCoreNumHash();

// Old cache hash for kbk only when cache is disabled.
BACKEND_EXPORT void GatherInfo(mindspore::kernel::KernelTensor *);
BACKEND_EXPORT void GatherInfo(const std::pair<mindspore::kernel::KernelTensor *, bool> &);
BACKEND_EXPORT void GatherInfo(const std::vector<mindspore::kernel::KernelTensor *> &);

BACKEND_EXPORT void GatherInfo(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherInfo(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherInfo(const std::vector<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherInfo(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherInfo(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherInfo(const std::vector<tensor::TensorPtr> &);

template <typename T>
BACKEND_EXPORT void GatherInfo(const T &value) {
  MemcpyToBuf(&value, sizeof(T));
}

template <typename T>
BACKEND_EXPORT void GatherInfo(std::optional<T> value) {
  if (value.has_value()) {
    GatherInfo(value.value());
  }
}

BACKEND_EXPORT void GatherInfo(const string &);
BACKEND_EXPORT void GatherInfo(const std::optional<string> &);

BACKEND_EXPORT void GatherInfo(const ScalarPtr &);
BACKEND_EXPORT void GatherInfo(const std::optional<ScalarPtr> &);

BACKEND_EXPORT void GatherInfo(const TypePtr &);
BACKEND_EXPORT void GatherInfo(const std::optional<TypePtr> &);

template <typename T>
BACKEND_EXPORT void GatherInfo(const std::vector<T> &values) {
  MemcpyToBuf(values.data(), values.size() * sizeof(T));
}

BACKEND_EXPORT inline void GatherInfo(TypeId type_id) { MemcpyToBuf(&type_id, sizeof(int)); }

BACKEND_EXPORT void GatherInfo();

template <typename T, typename... Args>
void GatherInfo(const T &arg, const Args &...args) {
  if (runtime::RuntimeConf::GetInstance()->IsEnableResLimit()) {
    GatherCoreNumHash();
  }
  GatherInfo(arg);
  GatherInfo(args...);
}

BACKEND_EXPORT void RefreshAddr(mindspore::kernel::KernelTensor *);
inline void RefreshAddr(const std::pair<mindspore::kernel::KernelTensor *, bool> &tensor_and_trans) {
  RefreshAddr(tensor_and_trans.first);
}

BACKEND_EXPORT void RefreshAddr(const std::pair<mindspore::kernel::KernelTensor *, bool> &);

inline void RefreshAddr(const std::vector<mindspore::kernel::KernelTensor *> &tensor_list) {
  for (auto tensor : tensor_list) {
    RefreshAddr(tensor);
  }
}

BACKEND_EXPORT void RefreshAddr(const mindspore::tensor::TensorPtr &tensor);

template <typename Args>
void RefreshAddr(const Args &values) {}

inline void RefreshAddr() {}

template <typename T, typename... Args>
void RefreshAddr(const T &arg, const Args &...args) {
  RefreshAddr(arg);
  RefreshAddr(args...);
}

BACKEND_EXPORT uint64_t calc_hash_id();
BACKEND_EXPORT uint64_t gen_hash(const void *key, const int len, const uint32_t seed = 0xdeadb0d7);

template <typename... Args>
bool HitCache(const char *aclnn_api, aclOpExecutor **executor, uint64_t *workspace_size, const Args &...args) {
  static const auto get_exec_cache = device::ascend::GetOpApiFunc("PTAGetExecCache");
  static const auto init_cache_thread_local = device::ascend::GetOpApiFunc("InitPTACacheThreadLocal");
  static const auto set_hash_key = device::ascend::GetOpApiFunc("SetPTAHashKey");
  static const auto can_use_cache = device::ascend::GetOpApiFunc("CanUsePTACache");
  GetExecCache get_exec_cache_func = reinterpret_cast<GetExecCache>(get_exec_cache);
  InitCacheThreadLocal init_cache_thread_local_func = reinterpret_cast<InitCacheThreadLocal>(init_cache_thread_local);
  SetHashKey set_hash_key_func = reinterpret_cast<SetHashKey>(set_hash_key);
  CanUseCache can_use_cache_func = reinterpret_cast<CanUseCache>(can_use_cache);
  bool has_func = get_exec_cache_func && init_cache_thread_local_func && set_hash_key_func;
  bool can_use = can_use_cache_func && can_use_cache_func(aclnn_api);
  if (!has_func || !can_use) {
    return false;
  }
  init_cache_thread_local_func();
  g_hash_offset = 0;
  GatherInfo(std::string(aclnn_api), args...);
  uint64_t hash_id = calc_hash_id();
  set_hash_key_func(hash_id);
  MS_EXCEPTION_IF_NULL(executor);
  *executor = get_exec_cache_func(hash_id, workspace_size);
  if (*executor == nullptr) {
    return false;
  }
  UninitCacheThreadLocal();
  return true;
}

template <typename... Args>
bool HitCacheSingle(const char *aclnn_api, aclOpExecutor **executor, uint64_t *workspace_size, uint64_t *hash_id,
                    const Args &...args) {
  static const auto get_exec_cache = device::ascend::GetOpApiFunc("PTAGetExecCache");
  static const auto init_cache_thread_local = device::ascend::GetOpApiFunc("InitPTACacheThreadLocal");
  static const auto set_hash_key = device::ascend::GetOpApiFunc("SetPTAHashKey");
  static const auto can_use_cache = device::ascend::GetOpApiFunc("CanUsePTACache");
  GetExecCache get_exec_cache_func = reinterpret_cast<GetExecCache>(get_exec_cache);
  InitCacheThreadLocal init_cache_thread_local_func = reinterpret_cast<InitCacheThreadLocal>(init_cache_thread_local);
  SetHashKey set_hash_key_func = reinterpret_cast<SetHashKey>(set_hash_key);
  CanUseCache can_use_cache_func = reinterpret_cast<CanUseCache>(can_use_cache);
  bool has_func = get_exec_cache_func && init_cache_thread_local_func && set_hash_key_func;
  bool can_use = can_use_cache_func && can_use_cache_func(aclnn_api);
  if (!has_func || !can_use) {
    return false;
  }
  init_cache_thread_local_func();
  g_hash_offset = 0;

  MS_EXCEPTION_IF_NULL(hash_id);
  if (*hash_id == 0) {
    GatherInfo(std::string(aclnn_api), args...);
    *hash_id = calc_hash_id();
  } else {
    RefreshAddr(args...);
  }

  set_hash_key_func(*hash_id);
  MS_EXCEPTION_IF_NULL(executor);
  *executor = get_exec_cache_func(*hash_id, workspace_size);
  if (*executor == nullptr) {
    return false;
  }
  UninitCacheThreadLocal();
  return true;
}

// New cache hash for kbk and pyboost.
BACKEND_EXPORT void GatherHash(mindspore::kernel::KernelTensor *);
BACKEND_EXPORT void GatherHash(const std::pair<mindspore::kernel::KernelTensor *, bool> &);
BACKEND_EXPORT void GatherHash(const std::vector<mindspore::kernel::KernelTensor *> &);
BACKEND_EXPORT void GatherHash(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherHash(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherHash(const std::vector<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherHash(const mindspore::tensor::TensorPtr &);
BACKEND_EXPORT void GatherHash(const std::optional<tensor::TensorPtr> &);
BACKEND_EXPORT void GatherHash(const std::vector<tensor::TensorPtr> &);

template <typename T>
BACKEND_EXPORT void GatherHash(const T &value) {
  GatherInfo(value);
}

BACKEND_EXPORT void GatherHash();

template <typename T, typename... Args>
void GatherHash(const T &arg, const Args &...args) {
  GatherHash(arg);
  GatherHash(args...);
}

template <typename... Args>
uint64_t AclnnHash(const std::string &arg, const Args &...args) {
  g_hash_offset = 0;
  if (runtime::RuntimeConf::GetInstance()->IsEnableResLimit()) {
    GatherCoreNumHash();
  }
  GatherHash(arg, args...);
  return calc_hash_id();
}
}  // namespace  mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CACHE_H_
