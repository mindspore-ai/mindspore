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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <string>
#include "backend/kernel_compiler/ascend_kernel_mod.h"
#include "backend/kernel_compiler/aicpu/aicpu_util.h"
namespace mindspore {
namespace kernel {
class AicpuOpKernelMod : public AscendKernelMod {
 public:
  AicpuOpKernelMod();
  ~AicpuOpKernelMod() override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;
  device::DynamicKernelPtr GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) override;

  void SetInputList(const std::vector<int64_t> &inputList);
  void SetOutputList(const std::vector<int64_t> &outputList);
  void SetAnfNode(const AnfNodePtr &anf_node);
  void SetNodeDef(const std::string &nodeDef);
  void SetExtInfo(const std::string &ext_info);
  void SetNodeName(const std::string &node_name);

  /**
   *  @brief Build AICPU Engine kernel structure, and allocate device memory for offline task generate
   *  @return SUCCESS
   *  @return FAIL
   *
   */
  void CreateCpuKernelInfo(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  void SetInputSizeList(const std::vector<size_t> &size_list);
  void SetOutputSizeList(const std::vector<size_t> &size_list);
  void SetWorkspaceSizeList(const std::vector<size_t> &size_list);
  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;

 private:
  std::string args_;
  std::string node_def_str_;
  std::string node_name_;
  std::string node_so_;
  std::string ext_info_;
  std::vector<int64_t> inputList_;
  std::vector<int64_t> outputList_;
  AnfNodePtr anf_node_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

using AicpuOpKernelModPtr = std::shared_ptr<AicpuOpKernelMod>;
using AicputOpKernelModPtrList = std::vector<AicpuOpKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_AICPU_KERNEL_MOD_H_
