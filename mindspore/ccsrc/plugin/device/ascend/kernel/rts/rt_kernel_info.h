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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RT_KERNEL_INFO_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RT_KERNEL_INFO_H

#include <memory>
#include <functional>
#include <map>
#include <string>
#include <set>
#include <vector>
#include <utility>

#include "ir/dtype.h"
#include "kernel/kernel_build_info.h"
#include "kernel/kernel.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
class RtKerDesc {
 public:
  virtual ~RtKerDesc() {}
  virtual std::vector<std::shared_ptr<kernel::KernelBuildInfo>> GetKernelInfo(const CNodePtr &) {
    return std::vector<std::shared_ptr<kernel::KernelBuildInfo>>{};
  }
};

using RtKerDescCreater = std::function<std::shared_ptr<RtKerDesc>()>;
class RtKerDescFactory {
  RtKerDescFactory() = default;
  ~RtKerDescFactory() = default;

 public:
  static RtKerDescFactory &Get();
  void Register(const std::string &name, RtKerDescCreater &&fun);
  static std::shared_ptr<RtKerDesc> Create(const std::string &name);

 private:
  std::map<std::string, RtKerDescCreater> fmap_;
};

class RtKerDescRegister {
 public:
  RtKerDescRegister(const std::string &name, RtKerDescCreater &&fun) {
    RtKerDescFactory::Get().Register(name, std::move(fun));
  }
  ~RtKerDescRegister() = default;
};

#define MS_REG_RTKERNEL_DESC_REG(KNAME, clazz)                                           \
  static_assert(std::is_base_of<RtKerDesc, clazz>::value, " must be base of RtKerDesc"); \
  static const RtKerDescRegister g_##KNAME##_##_rtkernel_desc_reg(#KNAME, []() { return std::make_shared<clazz>(); });

#define MS_REG_RTKERNEL_DESC(KNAME, clazz) MS_REG_RTKERNEL_DESC_REG(KNAME, clazz)

void GetRtKelInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RT_KERNEL_INFO_H
