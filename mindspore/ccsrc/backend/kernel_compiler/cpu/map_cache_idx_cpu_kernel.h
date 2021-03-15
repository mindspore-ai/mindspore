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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_CACHE_IDX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_CACHE_IDX_CPU_KERNEL_H_

#include <math.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class MapCacheIdxCPUKernel : public CPUKernel {
 public:
  MapCacheIdxCPUKernel() = default;
  ~MapCacheIdxCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

 private:
  size_t batch_size_{1};
  size_t hashmap_length_{1};
  TypeId dtype_{kTypeUnknown};
  CNodeWeakPtr node_wpt_;
};

MS_REG_CPU_KERNEL(MapCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  MapCacheIdxCPUKernel);

MS_REG_CPU_KERNEL(MapCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  MapCacheIdxCPUKernel);

MS_REG_CPU_KERNEL(MapCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  MapCacheIdxCPUKernel);

MS_REG_CPU_KERNEL(MapCacheIdx,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  MapCacheIdxCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEARCH_CACHE_IDX_CPU_KERNEL_H_
