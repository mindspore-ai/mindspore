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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCCL_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCCL_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "kernel/kernel.h"
#include "plugin/device/ascend/kernel/hccl/hcom_util.h"
#include "hccl/hcom.h"
#include "hccl/hccl_types.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
class HcclKernel : public KernelMod {
 public:
  // =========================================New interface==========================================================
  HcclKernel();
  ~HcclKernel() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  void SetIsGraphMode(bool is_graph_mode) { is_graph_mode_ = is_graph_mode; }

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in hccl kernel module.";
  }

 protected:
  virtual HcclDataType GetHcclDataType() const;
  virtual void CalLoopSize();
  bool CalcTypeShapeAndCount(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  std::vector<std::vector<int64_t>> hccl_kernel_input_shape_list_;
  std::vector<std::vector<int64_t>> hccl_kernel_output_shape_list_;
  std::vector<HcclDataType> hccl_data_type_list_;
  std::vector<std::string> hccl_format_list_;
  uint64_t hccl_count_;
  HcclReduceOp op_type_;
  uint32_t root_id_;
  uint32_t src_rank_;
  uint32_t dest_rank_;
  std::string group_;
  HcclComm comm_;
  std::mutex hccl_mutex_;
  std::condition_variable cond_;
  ulong loop_size_{0};
  bool is_graph_mode_{false};
};

extern int64_t op_tag;
using HcclKernelCreater = std::function<std::shared_ptr<HcclKernel>()>;

class HcclKernelFactory {
  HcclKernelFactory() = default;
  ~HcclKernelFactory() = default;

 public:
  static HcclKernelFactory &Get();
  void Register(const string &name, HcclKernelCreater &&fun);
  static std::shared_ptr<HcclKernel> Get(const string &name);

 private:
  std::map<string, HcclKernelCreater> hccl_kernel_map_;
};

class HcclKernelRegister {
 public:
  HcclKernelRegister(const string &name, HcclKernelCreater &&fun) {
    HcclKernelFactory::Get().Register(name, std::move(fun));
  }
  ~HcclKernelRegister() = default;
};

#define MS_HCCL_REG_KERNEL_REG(KNAME, clazz)                                               \
  static_assert(std::is_base_of<HcclKernel, clazz>::value, " must be base of HcclKernel"); \
  static const HcclKernelRegister g_##KNAME##_##_kernel_reg(#KNAME, []() {                 \
    std::shared_ptr<clazz> ptr = nullptr;                                                  \
    ptr = std::make_shared<clazz>();                                                       \
    MS_EXCEPTION_IF_NULL(ptr);                                                             \
    return ptr;                                                                            \
  });

#define MS_HCCL_REG_KERNEL(KNAME, clazz) MS_HCCL_REG_KERNEL_REG(KNAME, clazz)
}  // namespace kernel
}  // namespace mindspore
#endif
