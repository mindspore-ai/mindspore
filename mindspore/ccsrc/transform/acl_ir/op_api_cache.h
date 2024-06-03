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
#include "transform/acl_ir/op_api_convert.h"

namespace mindspore::transform {
typedef aclOpExecutor *(*GetExecCache)(uint64_t, uint64_t *);
typedef void (*InitCacheThreadLocal)();
typedef void (*UnInitCacheThreadLocal)();
typedef void (*SetHashKey)(uint64_t);
typedef bool (*CanUseCache)(const char *);

constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern thread_local char g_hash_buf[g_hash_buf_size];
extern thread_local int g_hash_offset;

inline void MemcpyToBuf(const void *data_expression, size_t size_expression) {
  if (size_expression == 0) {
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

// Old cache hash.
void GatherInfo(mindspore::kernel::KernelTensor *);
void GatherInfo(const std::pair<mindspore::kernel::KernelTensor *, bool> &);
void GatherInfo(const std::vector<mindspore::kernel::KernelTensor *> &);
void GatherInfo(const device::DeviceAddressPtr &);

void GatherInfo(const mindspore::tensor::BaseTensorPtr &);
void GatherInfo(const std::optional<tensor::BaseTensorPtr> &);
void GatherInfo(const std::vector<tensor::BaseTensorPtr> &);
void GatherInfo(const mindspore::tensor::TensorPtr &);
void GatherInfo(const std::optional<tensor::TensorPtr> &);
void GatherInfo(const std::vector<tensor::TensorPtr> &);

template <typename T>
void GatherInfo(const T &value) {
  MemcpyToBuf(&value, sizeof(T));
}

template <typename T>
void GatherInfo(std::optional<T> value) {
  if (value.has_value()) {
    GatherInfo(value.value());
  }
}

void GatherInfo(const string &);
void GatherInfo(const std::optional<string> &);

void GatherInfo(const ScalarPtr &);
void GatherInfo(const std::optional<ScalarPtr> &);

void GatherInfo(const TypePtr &);
void GatherInfo(const std::optional<TypePtr> &);

template <typename T>
void GatherInfo(const std::vector<T> &values) {
  MemcpyToBuf(values.data(), values.size() * sizeof(T));
}

inline void GatherInfo(TypeId type_id) { MemcpyToBuf(&type_id, sizeof(int)); }

void GatherInfo();

template <typename T, typename... Args>
void GatherInfo(const T &arg, const Args &... args) {
  GatherInfo(arg);
  GatherInfo(args...);
}

void RefreshAddr(mindspore::kernel::KernelTensor *);
void RefreshAddr(const std::pair<mindspore::kernel::KernelTensor *, bool> &);
void RefreshAddr(const device::DeviceAddressPtr &device_address);
void RefreshAddr(const mindspore::tensor::TensorPtr &tensor);
inline void RefreshAddr(const std::vector<mindspore::kernel::KernelTensor *> &tensor_list) {
  for (auto tensor : tensor_list) {
    RefreshAddr(tensor);
  }
}

template <typename Args>
void RefreshAddr(const Args &values) {}

inline void RefreshAddr() {}

template <typename T, typename... Args>
void RefreshAddr(const T &arg, const Args &... args) {
  RefreshAddr(arg);
  RefreshAddr(args...);
}

uint64_t calc_hash_id();
uint64_t gen_hash(const void *key, const int len, const uint32_t seed = 0xdeadb0d7);

template <typename... Args>
bool HitCache(const char *aclnn_api, aclOpExecutor **executor, uint64_t *workspace_size, const Args &... args) {
  static const auto get_exec_cache = transform::GetOpApiFunc("PTAGetExecCache");
  static const auto init_cache_thread_local = transform::GetOpApiFunc("InitPTACacheThreadLocal");
  static const auto set_hash_key = transform::GetOpApiFunc("SetPTAHashKey");
  static const auto can_use_cache = transform::GetOpApiFunc("CanUsePTACache");
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
  *executor = get_exec_cache_func(hash_id, workspace_size);
  static const auto uninit_cache_thread_local = transform::GetOpApiFunc("UnInitPTACacheThreadLocal");
  UnInitCacheThreadLocal uninit_cache_thread_local_func =
    reinterpret_cast<UnInitCacheThreadLocal>(uninit_cache_thread_local);
  uninit_cache_thread_local_func();
  if (*executor == nullptr) {
    return false;
  }
  return true;
}

template <typename... Args>
uint64_t CalcOpApiHash(const std::string &arg, const Args &... args) {
  g_hash_offset = 0;
  GatherInfo(arg, args...);
  return calc_hash_id();
}

template <typename... Args>
bool HitCacheSingle(const char *aclnn_api, aclOpExecutor **executor, uint64_t *workspace_size, uint64_t *hash_id,
                    const Args &... args) {
  static const auto get_exec_cache = transform::GetOpApiFunc("PTAGetExecCache");
  static const auto init_cache_thread_local = transform::GetOpApiFunc("InitPTACacheThreadLocal");
  static const auto set_hash_key = transform::GetOpApiFunc("SetPTAHashKey");
  static const auto can_use_cache = transform::GetOpApiFunc("CanUsePTACache");
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

  if (*hash_id == 0) {
    GatherInfo(std::string(aclnn_api), args...);
    *hash_id = calc_hash_id();
  } else {
    RefreshAddr(args...);
  }

  set_hash_key_func(*hash_id);
  *executor = get_exec_cache_func(*hash_id, workspace_size);
  if (*executor == nullptr) {
    return false;
  }
  return true;
}

// New cache hash.
void GatherHash(mindspore::kernel::KernelTensor *);
void GatherHash(const std::pair<mindspore::kernel::KernelTensor *, bool> &);
void GatherHash(const std::vector<mindspore::kernel::KernelTensor *> &);
void GatherHash(const device::DeviceAddressPtr &);

void GatherHash(const mindspore::tensor::TensorPtr &);
void GatherHash(const std::optional<tensor::TensorPtr> &);
void GatherHash(const std::vector<tensor::TensorPtr> &);

template <typename T>
void GatherHash(const T &value) {
  GatherInfo(value);
}

void GatherHash();

template <typename T, typename... Args>
void GatherHash(const T &arg, const Args &... args) {
  GatherHash(arg);
  GatherHash(args...);
}

template <typename... Args>
uint64_t AclnnHash(const std::string &arg, const Args &... args) {
  g_hash_offset = 0;
  GatherHash(arg, args...);
  return calc_hash_id();
}
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_CACHE_H_
