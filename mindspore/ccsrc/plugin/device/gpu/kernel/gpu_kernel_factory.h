/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPUKERNELFACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPUKERNELFACTORY_H_

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "kernel/kernel_build_info.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
using NativeGpuKernelModCreater = std::function<NativeGpuKernelMod *()>;
class NativeGpuKernelModFactory {
 public:
  ~NativeGpuKernelModFactory() = default;

  static NativeGpuKernelModFactory &GetInstance();

  void Register(const std::string &kernel_name, const KernelAttr &kernel_attr, NativeGpuKernelModCreater &&creator);

  NativeGpuKernelMod *Create(const std::string &kernel_name, const CNodePtr &apply_kernel);

  bool SearchRegistered(const std::string &kernel_name, const KernelBuildInfoPtr &kernel_info);

  std::string SupportedTypeList(const std::string &kernel_name);

  std::vector<KernelAttr> GetGpuSupportedList(const std::string &kernel_name);

  // Judge whether is registered kernel.
  bool IsRegistered(const std::string &kernel_name);

  bool ReducePrecision(const std::string &kernel_name,
                       std::shared_ptr<mindspore::kernel::KernelBuildInfo::KernelBuildInfoBuilder> builder);

  std::pair<std::vector<size_t>, TypeId> reduce_flag_{{}, kNumberTypeInt64};

 private:
  NativeGpuKernelModFactory() = default;

  NativeGpuKernelModFactory(NativeGpuKernelModFactory const &);

  NativeGpuKernelModFactory &operator=(const NativeGpuKernelModFactory &);

  std::pair<bool, size_t> GpuKernelAttrCheck(const std::string &kernel_name, const KernelBuildInfo *kernel_info);
  void CheckSM(const KernelBuildInfo *kernel_info, const size_t &input_index);
  bool CheckIOParam(const std::string &kernel_name, const KernelBuildInfo *kernel_info,
                    std::vector<std::pair<KernelAttr, NativeGpuKernelModCreater>> *iter_second, size_t attr_index);

  // Set output and input ref map to kernel info which will be used by graph compiler.
  void SetRefMapToKernelInfo(const std::string &kernel_name, size_t index, device::KernelInfo *kernel_info);

  // map to maintain kernel and creator, KernelAttr object and creator must be registered as a pair.
  std::map<std::string, std::vector<std::pair<KernelAttr, NativeGpuKernelModCreater>>> map_kernel_name_to_creater_;
};

class GpuKernelRegister {
 public:
  GpuKernelRegister(const std::string &kernel_name, const KernelAttr &kernel_attr,
                    NativeGpuKernelModCreater &&creator) {
    NativeGpuKernelModFactory::GetInstance().Register(kernel_name, kernel_attr, std::move(creator));
  }
  ~GpuKernelRegister() = default;
};

#define UNIQUE_KERNEL_NAME(kernel) KERNEL_NAME(g_##kernel##_gpu_kernel_reg, __COUNTER__)
#define KERNEL_NAME(kernel, cnt) MERGE(kernel, cnt)
#define MERGE(kernel, cnt) kernel##cnt

#define MS_REG_GPU_KERNEL(OPNAME, OPCLASS)                                                                   \
  static_assert(std::is_base_of<NativeGpuKernelMod, OPCLASS>::value, " must be base of NativeGpuKernelMod"); \
  static const GpuKernelRegister UNIQUE_KERNEL_NAME(OPNAME)(#OPNAME, KernelAttr(), []() { return new OPCLASS(); });

// regular register of fixed accuracy kernels
#define MS_REG_GPU_KERNEL_REGULAR(OPNAME, ATTR, OPCLASS)                                                     \
  static_assert(std::is_base_of<NativeGpuKernelMod, OPCLASS>::value, " must be base of NativeGpuKernelMod"); \
  static const GpuKernelRegister UNIQUE_KERNEL_NAME(OPNAME)(#OPNAME, ATTR, []() { return new OPCLASS(); });

// register of mixed accuracy kernels which use template and maintain one typename, ignore input num
#define MS_REG_GPU_KERNEL_SAME(OPNAME, ATTR, OPCLASS, T)                                                        \
  static_assert(std::is_base_of<NativeGpuKernelMod, OPCLASS<T>>::value, " must be base of NativeGpuKernelMod"); \
  static const GpuKernelRegister UNIQUE_KERNEL_NAME(OPNAME)(#OPNAME, ATTR, []() { return new OPCLASS<T>(); });

// register of mixed accuracy kernels which use template and maintain one typename
#define MS_REG_GPU_KERNEL_ONE(OPNAME, ATTR, OPCLASS, T)                                                         \
  static_assert(std::is_base_of<NativeGpuKernelMod, OPCLASS<T>>::value, " must be base of NativeGpuKernelMod"); \
  static const GpuKernelRegister UNIQUE_KERNEL_NAME(OPNAME)(#OPNAME, ATTR, []() { return new OPCLASS<T>(); });

// register of mixed accuracy kernels which use template and maintain two typename
#define MS_REG_GPU_KERNEL_TWO(OPNAME, ATTR, OPCLASS, T, S)                                                         \
  static_assert(std::is_base_of<NativeGpuKernelMod, OPCLASS<T, S>>::value, " must be base of NativeGpuKernelMod"); \
  static const GpuKernelRegister UNIQUE_KERNEL_NAME(OPNAME)(#OPNAME, ATTR, []() { return new OPCLASS<T, S>(); });

// register of mixed accuracy kernels which use template and maintain three typename
#define MS_REG_GPU_KERNEL_THREE(OPNAME, ATTR, OPCLASS, T, S, G)                                                       \
  static_assert(std::is_base_of<NativeGpuKernelMod, OPCLASS<T, S, G>>::value, " must be base of NativeGpuKernelMod"); \
  static const GpuKernelRegister UNIQUE_KERNEL_NAME(OPNAME)(#OPNAME, ATTR, []() { return new OPCLASS<T, S, G>(); });
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPUKERNELFACTORY_H_
