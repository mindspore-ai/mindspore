/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BATCH_NORM_GRAD_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BATCH_NORM_GRAD_GRAD_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include <string>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class BatchNormGradGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  BatchNormGradGradCpuKernelMod() = default;
  ~BatchNormGradGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void TrainingComputeNHWC(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &workspace,
                           const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void InferenceComputeNHWC(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> &workspace,
                            const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void TrainingComputeNCHW(const std::vector<kernel::AddressPtr> &inputs,
                           const std::vector<kernel::AddressPtr> &workspace,
                           const std::vector<kernel::AddressPtr> &outputs);
  template <typename T>
  void InferenceComputeNCHW(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> &workspace,
                            const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void TrainingNHWCCalculateDx(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &workspace,
                               const std::vector<kernel::AddressPtr> &outputs, float *x_hat, float *inv_std);

  template <typename T>
  void TrainingNHWCCalculateDdy(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs, float *x_hat, float *inv_std);

  template <typename T>
  void TrainingNHWCCalculateDscale(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &workspace,
                                   const std::vector<kernel::AddressPtr> &outputs, float *x_hat, float *inv_std);

  template <typename T>
  void TrainingNCHWCalculateDx(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &workspace,
                               const std::vector<kernel::AddressPtr> &outputs, float *x_hat, float *inv_std);

  template <typename T>
  void TrainingNCHWCalculateDdy(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs, float *x_hat, float *inv_std);

  template <typename T>
  void TrainingNCHWCalculateDscale(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &workspace,
                                   const std::vector<kernel::AddressPtr> &outputs, float *x_hat, float *inv_std);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  using BatchNormGradGradFunc =
    std::function<bool(BatchNormGradGradCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, BatchNormGradGradFunc>> func_list_;
  BatchNormGradGradFunc kernel_func_;

  float epsilon_;
  bool is_training_;
  std::string data_format_;
  int x_num_;
  int C_num_;
  int N_num_;
  int CHW_num_;
  int NHW_num_;
  int HW_num_;
  float M_;
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> scale_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BATCH_NORM_GRAD_GRAD_CPU_KERNEL_H_
