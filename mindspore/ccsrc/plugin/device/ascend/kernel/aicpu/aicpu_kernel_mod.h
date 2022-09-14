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
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
namespace mindspore {
namespace kernel {
class AicpuOpKernelMod : public AscendKernelMod {
 public:
  AicpuOpKernelMod();
  explicit AicpuOpKernelMod(const AnfNodePtr &anf_node_ptr);

  ~AicpuOpKernelMod() override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;

  void SetInputList(const std::vector<int64_t> &input_list);
  void SetOutputList(const std::vector<int64_t> &output_list);
  void SetAnfNode(const AnfNodePtr &anf_node);
  void SetNodeDef(const std::string &node_def);
  void SetExtInfo(const std::string &ext_info);
  void SetNodeName(const std::string &node_name);
  void SetCustSo(const std::string &cust_so);

  /**
   *  @brief Build AICPU Engine kernel structure, and allocate device memory for offline task generate
   *  @return SUCCESS
   *  @return FAIL
   *
   */
  void CreateCpuKernelInfo(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 protected:
  std::string args_;
  std::string ext_info_;
  std::string node_name_;
  std::string node_so_;
  bool cust_kernel_{false};
  std::string node_def_str_;
  // Because the ~DynamicAicpuKernelMod() is after ResetDevice, and ResetDevice has the function to free mem,
  // so it is no rtFree of ext_info_addr_dev_ in ~DynamicAicpuKernelMod()
  void *ext_info_addr_dev_ = nullptr;

 private:
  std::vector<int64_t> input_list_;
  std::vector<int64_t> output_list_;
};

using AicpuOpKernelModPtr = std::shared_ptr<AicpuOpKernelMod>;
using AicputOpKernelModPtrList = std::vector<AicpuOpKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_MOD_H_
