/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_AICPU_ENVIRON_H_
#define AICPU_KERNELS_NORMALIZED_AICPU_ENVIRON_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "inc/kernel_log.h"

namespace aicpu {
constexpr int64_t kObjectTypeTensorType = 17;
constexpr int64_t kObjectTypeEnvType = 26;
// index of input or output
enum Index : size_t {
  kIndex0 = 0,
  kIndex1,
  kIndex2,
  kIndex3,
  kIndex4,
  kIndex5,
  kIndex6,
  kIndex7,
  kIndex8,
  kIndex9,
  kIndex10,
  kIndex11,
  kIndex12,
  kIndex13,
  kIndex14,
  kIndex15,
  kIndex16,
};

struct EnvironValue {
  EnvironValue() : addr_(nullptr), size_(0), value_type_(kObjectTypeTensorType) {}
  EnvironValue(void *address_addr, size_t address_size, int32_t value_type)
      : addr_(address_addr), size_(address_size), value_type_(value_type) {}

  void *addr_;
  size_t size_;
  int32_t value_type_;
};

using EnvironValuePtr = std::shared_ptr<EnvironValue>;

// Environ is the meaning expression of map.
class Environ {
 public:
  explicit Environ(int64_t handle) : handle_(handle) {}
  virtual ~Environ() = default;

  void Set(int64_t key, const EnvironValuePtr &value) { values_[key] = value; }

  EnvironValuePtr Get(int64_t key) {
    if (values_.count(key) > 0) {
      return values_[key];
    }
    return nullptr;
  }

  void Clear(CpuKernelContext &ctx) {
    // Foreach values to free the value addr.
    for (auto &value : values_) {
      CUST_KERNEL_CHECK_NULLPTR_VOID(ctx, value.second, "value.second is null.");
      free(value.second->addr_);
    }
    values_.clear();
    handle_ = 0;
  }

 private:
  // The handle is unique for each env.
  int64_t handle_ = 0;

  // Store the tensors in map, as <key, tensor>.
  std::map<int64_t, EnvironValuePtr> values_;
};
using EnvironPtr = std::shared_ptr<Environ>;
}  // namespace aicpu

#endif  // AICPU_KERNELS_NORMALIZED_AICPU_ENVIRON_H_
