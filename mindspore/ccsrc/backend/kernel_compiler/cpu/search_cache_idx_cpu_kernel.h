/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEARCH_CACHE_IDX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEARCH_CACHE_IDX_CPU_KERNEL_H_

#include <math.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

#define NULLTAG 0

namespace mindspore {
namespace kernel {
template <typename T>
struct HashmapEntry {
  T key;
  T value;
  T step;
  T tag;

  bool IsEmpty() {
    if (this->tag == NULLTAG)
      return true;
    else
      return false;
  }

  bool IsUsing(const T &train_step) {
    if (this->step >= (train_step - 1))
      return true;
    else
      return false;
  }

  bool IsKey(const T &emb_idx) {
    if (this->key == emb_idx)
      return true;
    else
      return false;
  }

  void SetEmpty() { this->tag = NULLTAG; }
};

template <typename T>
T HashFunc(const T &key, const size_t &m) {
  return (T)(((0.6180339 * key) - floor(0.6180339 * key)) * m);
}

class SearchCacheIdxCPUKernel : public CPUKernel {
 public:
  SearchCacheIdxCPUKernel() = default;
  ~SearchCacheIdxCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

 private:
  size_t batch_size_{1};
  size_t hashmap_length_{1};
  size_t step_{0};
  int64_t emb_max_num = 999999999;
  int64_t cache_max_num = 999999999;
  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(SearchCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  SearchCacheIdxCPUKernel);

MS_REG_CPU_KERNEL(SearchCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  SearchCacheIdxCPUKernel);

MS_REG_CPU_KERNEL(SearchCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  SearchCacheIdxCPUKernel);

MS_REG_CPU_KERNEL(SearchCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  SearchCacheIdxCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEARCH_CACHE_IDX_CPU_KERNEL_H_
