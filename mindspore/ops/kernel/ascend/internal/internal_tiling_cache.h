/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_OP_CACHE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_OP_CACHE_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/internal.h"
#include "kernel/ascend/internal/tiling_mem_mgr.h"

namespace mindspore {
namespace kernel {
constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern BACKEND_EXPORT thread_local char g_hash_buf[g_hash_buf_size];
extern BACKEND_EXPORT thread_local int g_hash_offset;

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

BACKEND_EXPORT uint64_t calc_hash_id();
BACKEND_EXPORT uint64_t gen_hash(const void *key, const int len, const uint32_t seed = 0xdeadb0d7);

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

struct TilingCacheItem {
  std::atomic<int64_t> ref_count_{0};
  internal::TilingInfoPtr tiling_info_;
  void *host_addr_;
  size_t size_;

  TilingCacheItem(const internal::TilingInfoPtr &tiling_info, void *host_addr, size_t size)
      : ref_count_(1), tiling_info_(tiling_info), host_addr_(host_addr), size_(size) {}
};
using TilingCacheItemPtr = std::shared_ptr<TilingCacheItem>;

template <typename T>
inline void GatherSingleInfo(const std::string &, const T &input) {
  GatherHash(input);
}

template <>
inline void GatherSingleInfo(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs) {
  for (auto &input : inputs) {
    auto type = input->type_id();
    if (type == kObjectTypeTensorType) {
      GatherHash(input);
      GatherHash(input->format());
    } else if (type == kObjectTypeNumber) {
      auto data_type = input->dtype_id();
      switch (data_type) {
        case kNumberTypeBool: {
          auto value = input->GetValueWithCheck<bool>();
          GatherHash(value);
          break;
        }
        case kNumberTypeInt32: {
          auto value = input->GetValueWithCheck<int32_t>();
          GatherHash(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<int64_t>();
          GatherHash(value);
          break;
        }
        case kNumberTypeFloat32: {
          auto value = input->GetValueWithCheck<float>();
          GatherHash(value);
          break;
        }
        case kNumberTypeFloat64: {
          auto value = input->GetValueWithCheck<double>();
          GatherHash(value);
          break;
        }
        default:
          MS_LOG(INTERNAL_EXCEPTION) << "Unsupported dtype " << data_type << ", kernel: " << kernel_name;
      }
    } else if (type == kObjectTypeTuple || type == kObjectTypeList) {
      auto data_type = input->dtype_id();
      switch (data_type) {
        case kNumberTypeInt32: {
          auto value = input->GetValueWithCheck<std::vector<int32_t>>();
          GatherHash(value);
          break;
        }
        case kNumberTypeInt64: {
          auto value = input->GetValueWithCheck<std::vector<int64_t>>();
          GatherHash(value);
          break;
        }
        default:
          MS_LOG(INTERNAL_EXCEPTION) << "Unsupported dtype " << data_type << ", kernel: " << kernel_name;
      }
    } else if (type == kMetaTypeNone) {
      // skip
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Unsupported input type " << type << ", kernel: " << kernel_name;
    }
  }
}

inline void GatherHashsForKey(const std::string &) {}

template <typename T, typename... Args>
inline void GatherHashsForKey(const std::string &kernel_name, T first, Args... args) {
  GatherSingleInfo(kernel_name, first);
  GatherHashsForKey(kernel_name, args...);
}

class InternalTilingCache {
 public:
  InternalTilingCache() = default;
  ~InternalTilingCache() = default;

  static InternalTilingCache &GetInstance() {
    static InternalTilingCache tiling_cache;
    return tiling_cache;
  }

  TilingCacheItemPtr Bind(uint64_t key);
  void Unbind(const TilingCacheItemPtr &item);
  bool Insert(uint64_t key, const TilingCacheItemPtr &ti_ptr);
  std::vector<TilingCacheItemPtr> CombOutSuspectedUselessItems();
  void SetItemToPermanent(TilingCacheItemPtr ti_ptr);

  template <typename... Args>
  static inline uint64_t GenerateKey(const std::string &kernel_name, const std::vector<KernelTensor *> &inputs,
                                     Args... args) {
    g_hash_offset = 0;
    GatherHash(kernel_name);

    GatherHashsForKey(kernel_name, inputs, args...);
    auto hash_id = calc_hash_id();
    return hash_id;
  }

 private:
  std::unordered_map<uint64_t, TilingCacheItemPtr> cache_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_OP_CACHE_H_
