/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRANSPOSE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRANSPOSE_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <utility>
#include <unordered_map>
#include <memory>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/transpose.h"

namespace mindspore {
namespace kernel {
class TransposeFwdCpuKernelMod : public NativeCpuKernelMod {
 public:
  TransposeFwdCpuKernelMod() = default;
  ~TransposeFwdCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void ParallelRun(const T *input_addr, T *output_addr, size_t count) const;
  template <typename T>
  ErrorCodeCommonEnum DoTranspose(const T *in_data, T *out_data) const;
  template <typename T>
  void TransposeDim2(const T *in_data, T *out_data) const;
  template <typename T>
  void TransposeDim3(const T *in_data, T *out_data) const;
  template <typename T>
  void TransposeDim4(const T *in_data, T *out_data) const;
  template <typename T>
  void TransposeDim5(const T *in_data, T *out_data) const;
  template <typename T>
  void TransposeDim6(const T *in_data, T *out_data) const;
  template <typename T>
  void TransposeDim7(const T *in_data, T *out_data) const;
  template <typename T>
  void TransposeDims(const T *in_data, T *out_data, int64_t task_id, int64_t thread_num) const;

  void CheckPermValue();

  template <typename T>
  void InitPerm(const std::vector<kernel::AddressPtr> &inputs);

  std::vector<int64_t> input_shape_;
  std::vector<int64_t> perm_shape_;
  std::vector<int64_t> output_shape_;
  TypeId dtype_{kTypeUnknown};
  TypeId perm_type_{kNumberTypeInt64};
  std::vector<int64_t> perm_;
  size_t num_axes_{0};
  size_t data_num_{0};
  size_t output_size_{1};
  std::vector<int64_t> strides_;
  std::vector<int64_t> out_strides_;
  std::vector<size_t> tanspose_index_;
  bool got_perm_value_{false};

  using TypeKernel =
    std::function<void(TransposeFwdCpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, TypeKernel>> launch_list_;
  TypeKernel launch_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRANSPOSE_CPU_KERNEL_H_
