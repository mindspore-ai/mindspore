/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "runtime/pynative/op_runtime_info.h"
#include "transform/acl_ir/acl_convert.h"
#include "transform/acl_ir/op_api_exec.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using TensorParams = transform::TensorParams;
using aclOpExecutor = transform::aclOpExecutor;
using CallBackFunc = std::function<void()>;

class AclnnKernelMod : public AscendKernelMod {
 public:
  AclnnKernelMod() {}
  explicit AclnnKernelMod(const AnfNodePtr &anf_node_ptr) : AscendKernelMod(anf_node_ptr) {}
  ~AclnnKernelMod() = default;
  virtual bool Init(const AnfNodePtr &anf_node);
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr);
  void SetInputsInfo(const std::vector<TypeId> &type_ids, const ShapeArray &shapes);
  void SetOutputsInfo(const std::vector<TypeId> &type_ids, const ShapeArray &shapes);
  virtual int Resize(
    const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>());
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &, uint32_t) override;

  void ParseGenExecutor(const std::tuple<uint64_t, aclOpExecutor *, CallBackFunc> &args);
  bool IsNeedRetrieveOutputShape() override { return false; }

 protected:
  std::vector<TensorParams> input_params_;
  std::vector<TensorParams> output_params_;

  aclOpExecutor *executor_{nullptr};
  CallBackFunc after_launch_func_{nullptr};
};

using AclnnKernelModPtr = std::shared_ptr<AclnnKernelMod>;
using AclnnKernelModPtrList = std::vector<AclnnKernelModPtr>;
#define MS_ACLLNN_KERNEL_FACTORY_REG(NAME, DERIVE_CLASS) MS_KERNEL_FACTORY_REG(AclnnKernelMod, NAME, DERIVE_CLASS)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
