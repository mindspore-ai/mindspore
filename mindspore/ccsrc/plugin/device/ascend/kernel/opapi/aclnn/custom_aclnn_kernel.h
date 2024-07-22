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
constexpr size_t kTensorNum1 = 1;
constexpr size_t kTensorNum2 = 2;
constexpr size_t kTensorNum3 = 3;
constexpr size_t kTensorNum4 = 4;
constexpr size_t kTensorNum5 = 5;
constexpr size_t kTensorNum6 = 6;
constexpr size_t kTensorNum7 = 7;
constexpr size_t kTensorNum8 = 8;
constexpr size_t kTensorNum9 = 9;
constexpr size_t kTensorNum10 = 10;
constexpr size_t kTensorNum11 = 11;
constexpr size_t kTensorNum12 = 12;

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
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  template <typename... Ts>
  auto GenExecutor(const std::vector<Ts> &... vecs) {
    const auto &op_type = this->op_type_;
    const auto &hash_id = this->hash_id_;
    const auto &res_tuple = this->GetKernelTuple<N>(vecs...);
    auto executor_info = std::apply(
      [&op_type, &hash_id](const auto &... args) { return GEN_EXECUTOR_BOOST(op_type, hash_id, args...); }, res_tuple);
    return executor_info;
  }
};

inline std::shared_ptr<AclnnKernelMod> GetCustomAclNNKernelMod(const AnfNodePtr &anf_node) {
  auto primitive = GetCNodePrimitive(anf_node);
  auto op_type = GetValue<std::string>(primitive->GetAttr("reg_op_name"));
  auto arg_num = AnfUtils::GetInputTensorNum(anf_node) + AnfUtils::GetOutputTensorNum(anf_node);
  MS_LOG(INFO) << "Kernel " << anf_node->fullname_with_scope() << " is a custom op, op type : " << op_type
               << ", arg num : " << arg_num;
  switch (arg_num) {
    case kTensorNum1:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum1>>(op_type);
    case kTensorNum2:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum2>>(op_type);
    case kTensorNum3:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum3>>(op_type);
    case kTensorNum4:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum4>>(op_type);
    case kTensorNum5:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum5>>(op_type);
    case kTensorNum6:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum6>>(op_type);
    case kTensorNum7:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum7>>(op_type);
    case kTensorNum8:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum8>>(op_type);
    case kTensorNum9:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum9>>(op_type);
    case kTensorNum10:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum10>>(op_type);
    case kTensorNum11:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum11>>(op_type);
    case kTensorNum12:
      return std::make_shared<CustomAclnnKernelMod<kTensorNum12>>(op_type);
    default:
      MS_LOG(ERROR) << "Aclnn custom only support arg nums between 0 and 12, but get: " << arg_num;
  }
  return nullptr;
}

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CUSTOM_ACLNN_KERNEL_MOD_H_
