/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_KERNEL_GPU_GPUKERNELFACTORY_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_GPUKERNELFACTORY_H_

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "device/gpu/kernel_info_setter.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace kernel {
using mindspore::device::gpu::KernelAttr;
using GpuKernelCreater = std::function<GpuKernel *()>;
class GpuKernelFactory {
 public:
  ~GpuKernelFactory() = default;

  static GpuKernelFactory &GetInstance();

  void Register(const std::string &kernel_name, const KernelAttr &kernel_attr, GpuKernelCreater &&creater);

  GpuKernel *Create(const std::string &kernel_name, const CNodePtr &apply_kernel);

  bool SearchRegistered(const std::string &kernel_name, const KernelBuildInfoPtr &kernel_info);

  std::string SupportedTypeList(const std::string &kernel_name);

 private:
  GpuKernelFactory() = default;

  GpuKernelFactory(GpuKernelFactory const &);

  GpuKernelFactory &operator=(const GpuKernelFactory &);

  std::pair<bool, size_t> GpuKernelAttrCheck(const std::string &kernel_name, const KernelBuildInfo *kernel_info);
  void CheckIOParam(const std::string &kernel_name, const KernelBuildInfo *kernel_info,
                    std::vector<std::pair<KernelAttr, GpuKernelCreater>> *iter_second, size_t attr_index);
  // map to maintain kernel and creater, KernelAttr object and creater must be registered as a pair.
  std::map<std::string, std::vector<std::pair<KernelAttr, GpuKernelCreater>>> map_kernel_name_to_creater_;
};

class GpuKernelRegister {
 public:
  GpuKernelRegister(const std::string &kernel_name, const KernelAttr &kernel_attr, GpuKernelCreater &&creater) {
    GpuKernelFactory::GetInstance().Register(kernel_name, kernel_attr, std::move(creater));
  }
};

#define MS_REG_GPU_KERNEL(OPNAME, OPCLASS)                                                 \
  static_assert(std::is_base_of<GpuKernel, OPCLASS>::value, " must be base of GpuKernel"); \
  static const GpuKernelRegister g_##OPNAME##_gpu_kernel_reg(#OPNAME, KernelAttr(), []() { return new OPCLASS(); });

// regular register of fixed accuracy kernels
#define MS_REG_GPU_KERNEL_REGULAR(OPNAME, ATTR, OPCLASS)                                   \
  static_assert(std::is_base_of<GpuKernel, OPCLASS>::value, " must be base of GpuKernel"); \
  static const GpuKernelRegister g_##OPNAME##_gpu_kernel_reg(#OPNAME, ATTR, []() { return new OPCLASS(); });

// register of mixed accuracy kernels which use template and maintain one typename, ignore input num
#define MS_REG_GPU_KERNEL_SAME(OPNAME, ATTR, OPCLASS, T)                                      \
  static_assert(std::is_base_of<GpuKernel, OPCLASS<T>>::value, " must be base of GpuKernel"); \
  static const GpuKernelRegister g_##OPNAME##_##T##_gpu_kernel_reg(#OPNAME, ATTR, []() { return new OPCLASS<T>(); });

// register of mixed accuracy kernels which use template and maintain one typename
#define MS_REG_GPU_KERNEL_ONE(OPNAME, ATTR, OPCLASS, T)                                       \
  static_assert(std::is_base_of<GpuKernel, OPCLASS<T>>::value, " must be base of GpuKernel"); \
  static const GpuKernelRegister g_##OPNAME##_##T##_gpu_kernel_reg(#OPNAME, ATTR, []() { return new OPCLASS<T>(); });

// register of mixed accuracy kernels which use template and maintain two typename
#define MS_REG_GPU_KERNEL_TWO(OPNAME, ATTR, OPCLASS, T, S)                                       \
  static_assert(std::is_base_of<GpuKernel, OPCLASS<T, S>>::value, " must be base of GpuKernel"); \
  static const GpuKernelRegister g_##OPNAME##_##T##_##S##_gpu_kernel_reg(#OPNAME, ATTR,          \
                                                                         []() { return new OPCLASS<T, S>(); });
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_GPUKERNELFACTORY_H_
