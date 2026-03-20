/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_CUSTOM_V2_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_CUSTOM_V2_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include <set>
#include <tuple>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/custom_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "kernel/ascend/acl_ir/custom/custom_op_api_exec.h"
#include "kernel/ascend/acl_ir/custom/custom_aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace custom {

using ExecutorTuple = std::tuple<uint64_t, aclOpExecutor *, std::function<void()>, uint64_t, bool>;

class CustomV2AclnnKernelMod : public AclnnKernelMod {
 public:
  explicit CustomV2AclnnKernelMod(std::string op_type) : AclnnKernelMod(std::move(op_type)) {}
  ~CustomV2AclnnKernelMod();
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  std::vector<void *> converted_params_;
  std::vector<CustomSupportType> input_output_types_;
  std::vector<int64_t> inputs_int_value_;
  std::vector<float> inputs_float_value_;
  std::vector<char> inputs_bool_value_;
  std::vector<double> inputs_double_value_;
  std::vector<std::string> inputs_string_value_;
  std::vector<aclDataType> inputs_type_value_;
  CacheTuple GenCustomExecutorForResize(const std::vector<std::vector<KernelTensor *>> &inputs,
                                        const std::vector<std::vector<KernelTensor *>> &outputs);
  void GetWorkspaceForResize(const std::vector<std::vector<KernelTensor *>> &inputs,
                             const std::vector<std::vector<KernelTensor *>> &outputs);
  void RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace,
             const std::vector<std::vector<KernelTensor *>> &inputs,
             const std::vector<std::vector<KernelTensor *>> &outputs);
  std::pair<aclOpExecutor *, std::function<void()>> GetExecutor(
    const std::vector<std::vector<KernelTensor *>> &inputs, const std::vector<std::vector<KernelTensor *>> &outputs);
  ExecutorTuple GenCustomExecutor(const std::vector<std::vector<KernelTensor *>> &inputs,
                                  const std::vector<std::vector<KernelTensor *>> &outputs);

  bool CallGetWorkSpaceSize(const std::vector<std::vector<KernelTensor *>> &inputs,
                            const std::vector<std::vector<KernelTensor *>> &outputs, uint64_t *workspace_size_addr,
                            aclOpExecutor **executor_addr, void *get_workspace_size_func);
  std::vector<std::vector<void *>> GetTensorAddress(const std::vector<std::vector<KernelTensor *>> &inputs,
                                                    const std::vector<std::vector<KernelTensor *>> &outputs);
  void UpdateTensorForLaunch(const std::vector<std::vector<KernelTensor *>> &inputs,
                             const std::vector<std::vector<KernelTensor *>> &outputs, const ProcessCache &cache);
  std::vector<void *> ConvertTypes(const std::vector<std::vector<KernelTensor *>> &inputs, size_t offset = 0);
  std::vector<std::vector<KernelTensor *>> GetCustomInputs(const std::vector<KernelTensor *> &inputs);
  std::vector<std::vector<KernelTensor *>> GetCustomOutputs(const std::vector<KernelTensor *> &outputs);
  void GetCustomInputTypes();
};

}  // namespace custom
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_CUSTOM_V2_ACLNN_KERNEL_MOD_H_
