/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "runtime/rt.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_ext_info_handle.h"

namespace mindspore {
namespace kernel {
class AicpuOpKernelMod : public KernelMod {
 public:
  // =========================================New interface==========================================================
  AicpuOpKernelMod() : unknow_type_(::ge::UnknowShapeOpType::DEPEND_IN_SHAPE) {}

  ~AicpuOpKernelMod() override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  bool IsNeedUpdateOutputShapeAndSize() override;

  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not supported in AicpuOpKernelMod.";
  }

  void SetExtInfo(const std::string &ext_info, size_t input_num, size_t output_num);
  void SetNodeName(const std::string &node_name);
  void SetCustSo(const std::string &cust_so);
  void SetIsDynamicShape(bool is_dynamic_shape) { is_dynamic_shape_ = is_dynamic_shape; }
  void SetNodeScopeName(const std::string &scope_name) { node_scope_name_ = scope_name; }

  /**
   *  @brief Build AICPU Engine kernel structure, and allocate device memory for offline task generate
   *  @return SUCCESS
   *  @return FAIL
   *
   */
  void CreateCpuKernelInfo(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  void CloseTdtWingManQueue();

  // =======================Old interface, will deleted after all kernel modified used new interface=================

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    MS_LOG(EXCEPTION) << "Deprecated aicpu kernel module launch interface.";
  }

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override {
    MS_LOG(EXCEPTION) << "Deprecated aicpu kernel module resize interface.";
  }

 protected:
  std::string args_;
  std::string ext_info_;
  std::string node_name_;
  std::string node_so_;
  bool cust_kernel_{false};
  std::string node_def_str_;

  void *ext_info_addr_dev_ = nullptr;
  size_t ext_info_size_ = 0;
  std::shared_ptr<device::ascend::AicpuExtInfoHandler> ext_info_handler_ = nullptr;
  ::ge::UnknowShapeOpType unknow_type_;
  void *stream_ = nullptr;

 private:
  void AllocateExtInfoDeviceAddr();
  void FreeExtInfoDeviceAddr();
  bool CheckDeviceSupportBlockingAicpuOpProcess() const;
  void ParseNodeNameAndNodeSo();
  void CreateAsyncWaitEventAndUpdateEventInfo();
  bool IsOutputAllEmptyTensor(const std::vector<KernelTensor *> &outputs);
  std::vector<int64_t> input_list_;
  std::vector<int64_t> output_list_;
  rtEvent_t rt_event_ = nullptr;
  bool is_blocking_;  // is op has asyncflag
  bool need_skip_execute_ = false;
  bool is_output_all_empty_tensor_{false};
  bool is_dynamic_shape_{false};
  uint32_t stream_id_{0};
  std::string node_scope_name_;
};

using AicpuOpKernelModPtr = std::shared_ptr<AicpuOpKernelMod>;
using AicputOpKernelModPtrList = std::vector<AicpuOpKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_MOD_H_
