/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_DYNAMIC_TBE_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_DYNAMIC_TBE_KERNEL_MOD_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_mod.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"
#include "include/backend/device_address.h"
#include "ir/tensor.h"
#include "register/op_tiling.h"
namespace mindspore {
namespace kernel {
class DynamicTbeKernelMod : public TbeKernelMod {
 public:
  explicit DynamicTbeKernelMod(const KernelPackPtr &kernel_pack) : TbeKernelMod(kernel_pack) {}  // maybe delete later
  DynamicTbeKernelMod(KernelPackPtr kernel_pack, const AnfNodePtr &anf_node_ptr);
  ~DynamicTbeKernelMod() override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 protected:
  void SyncData() override;

  // Called only by atomic clean ops
  void InitAtomicOps(const optiling::utils::OpRunInfo &op_info);

 private:
  void InferShapeRecursive();
  void InferShapeForNopNode(AnfNodePtr *input_node);
  std::string ParseCompileJson(const CNodePtr &cnode) const;
  void InitTilingDataPtr();
  void CopyTilingToDevice(void *stream_ptr);
  void GenFuncStub();

  std::string tiling_data_;
  // Because the ~DynamicTbeKernelMod() is after ResetDevice, and ResetDevice has the function to free mem,
  // so it is no rtFree of tiling_data_ptr_ in ~DynamicTbeKernelMod()
  void *tiling_data_ptr_ = nullptr;
  uint64_t tiling_key_{0};
  void *handle_ = nullptr;
  void *func_stub_ = nullptr;
  std::string op_compile_info_{};
  bool need_skip_execute_ = false;
};

using DynamicTbeKernelModPtr = std::shared_ptr<DynamicTbeKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_MOD_H_
