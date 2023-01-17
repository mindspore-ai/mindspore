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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RT_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RT_KERNEL_H

#include <vector>
#include <utility>
#include <memory>
#include <map>
#include <string>
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "kernel/task_stream.h"

namespace mindspore {
namespace kernel {
class RtKernel : public AscendKernelMod {
 public:
  RtKernel();
  ~RtKernel() override;
  virtual bool Init(const AnfNodePtr &anf_node);
  void SetInputSizeList(const std::vector<size_t> &size_list) override;
  void SetOutputSizeList(const std::vector<size_t> &size_list) override;
  void SetWorkspaceSizeList(const std::vector<size_t> &size_list) override;
  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;

 protected:
  mutable std::vector<size_t> mutable_input_size_list_;
  mutable std::vector<size_t> mutable_output_size_list_;
  mutable std::vector<size_t> mutable_workspace_size_list_;
};

using RTKernelPtr = std::shared_ptr<RtKernel>;

using RtKernelCreater = std::function<std::shared_ptr<RtKernel>()>;
class RtKernelFactory {
  RtKernelFactory() = default;
  ~RtKernelFactory() = default;

 public:
  static RtKernelFactory &Get();
  void Register(const std::string &name, RtKernelCreater &&fun);
  static std::shared_ptr<RtKernel> Create(const std::string &name);

 private:
  std::map<string, RtKernelCreater> fmap_;
};

class RtKernelRegister {
 public:
  RtKernelRegister(const std::string &name, RtKernelCreater &&fun) {
    RtKernelFactory::Get().Register(name, std::move(fun));
  }
  ~RtKernelRegister() = default;
};

#define MS_REG_RTKERNEL_REG(KNAME, clazz)                                              \
  static_assert(std::is_base_of<RtKernel, clazz>::value, " must be base of RtKernel"); \
  static const RtKernelRegister g_##KNAME##_##_RtKernel_reg(#KNAME, []() { return std::make_shared<clazz>(); });

#define MS_REG_RTKERNEL(KNAME, clazz) MS_REG_RTKERNEL_REG(KNAME, clazz)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RT_KERNEL_H
