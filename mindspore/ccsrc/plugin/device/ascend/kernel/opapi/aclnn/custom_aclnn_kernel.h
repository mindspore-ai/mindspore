/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
template <size_t N>
class CustomAclnnKernelMod : public AclnnKernelMod {
 public:
  explicit CustomAclnnKernelMod(std::string op_type) : AclnnKernelMod(std::move(op_type)) {}
  ~CustomAclnnKernelMod() = default;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    const auto &res_tuple = this->GetKernelTuple<N>(inputs, outputs);
    std::apply(
      [this](const auto &... args) {
        hash_id_ = transform::CalcOpApiHash(op_type_, args...);
        if (cache_hash_.count(hash_id_) == 0) {
          const bool use_huge_pages = false;
          auto return_value = GEN_EXECUTOR_CUST(op_type_, use_huge_pages, args...);
          UpdateWorkspace(return_value);
        } else {
          auto return_value = GEN_EXECUTOR_BOOST(op_type_, hash_id_, args...);
          UpdateWorkspace(return_value);
        }
      },
      res_tuple);
  }
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    ParseGenExecutor(GenExecutor(inputs, outputs));
    RunOp(stream_ptr, workspace);
    return true;
  }

 private:
  template <typename... Ts>
  auto GenExecutor(const std::vector<Ts> &... vecs) {
    const auto &op_type = this->op_type_;
    const auto &hash_id = this->hash_id_;
    const auto &res_tuple = this->GetKernelTuple<N>(vecs...);
    auto executor_info = std::apply(
      [&op_type, &hash_id](const auto &... args) { return GEN_EXECUTOR_BOOST(op_type, hash_id, args...); }, res_tuple);
    return executor_info;
  }

  void RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace) {
    if (workspace_size_list_.empty()) {
      RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor_, stream_ptr, release_func_);
    } else {
      if (workspace.empty()) {
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
      }
      auto workspace_tensor = workspace[0];
      if (workspace_tensor->size() != workspace_size_list_[0]) {
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
      }
      RUN_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor_, stream_ptr,
                       release_func_);
    }
  }
};

inline std::shared_ptr<AclnnKernelMod> GetCustomAclNNKernelMod(const AnfNodePtr &anf_node) {
  auto primitive = GetCNodePrimitive(anf_node);
  auto op_type = GetValue<std::string>(primitive->GetAttr("reg_op_name"));
  auto arg_num = AnfUtils::GetInputTensorNum(anf_node) + AnfUtils::GetOutputTensorNum(anf_node);
  MS_LOG(INFO) << "Kernel " << anf_node->fullname_with_scope() << " is a custom op, op type : " << op_type
               << ", arg num : " << arg_num;
  switch (arg_num) {
    case 1:
      return std::make_shared<CustomAclnnKernelMod<1>>(op_type);
    case 2:
      return std::make_shared<CustomAclnnKernelMod<2>>(op_type);
    case 3:
      return std::make_shared<CustomAclnnKernelMod<3>>(op_type);
    case 4:
      return std::make_shared<CustomAclnnKernelMod<4>>(op_type);
    case 5:
      return std::make_shared<CustomAclnnKernelMod<5>>(op_type);
    case 6:
      return std::make_shared<CustomAclnnKernelMod<6>>(op_type);
    case 7:
      return std::make_shared<CustomAclnnKernelMod<7>>(op_type);
    case 8:
      return std::make_shared<CustomAclnnKernelMod<8>>(op_type);
    case 9:
      return std::make_shared<CustomAclnnKernelMod<9>>(op_type);
    case 10:
      return std::make_shared<CustomAclnnKernelMod<10>>(op_type);
    case 11:
      return std::make_shared<CustomAclnnKernelMod<11>>(op_type);
    case 12:
      return std::make_shared<CustomAclnnKernelMod<12>>(op_type);
    default:
      MS_LOG(ERROR) << "Aclnn custom only support arg nums between 0 and 12, but get: " << arg_num;
  }
  return nullptr;
}

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_ACLNN_KERNEL_MOD_H_
