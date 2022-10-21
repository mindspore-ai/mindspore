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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/grad/fractional_max_pool_grad_with_fixed_ksize.h"

namespace mindspore {
namespace kernel {
class FractionalMaxPoolGradWithFixedKsizeCPUKernelMod : public NativeCpuKernelMod {
 public:
  FractionalMaxPoolGradWithFixedKsizeCPUKernelMod() = default;
  ~FractionalMaxPoolGradWithFixedKsizeCPUKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename backprop_t>
  bool GradComputeTemplate(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const;
  template <typename backprop_t>
  void FractionalMaxPoolGradWithFixedKsizeCompute(backprop_t *out_backpropForPlane, int64_t *argmaxForPlane,
                                                  backprop_t *outputForPlane) const;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> out_backprop_shape_;
  std::vector<int64_t> argmax_shape_;
  std::string data_format_{"NCHW"};
  TypeId out_backprop_type_;
  int64_t input_n_;
  int64_t input_c_;
  int64_t input_h_;
  int64_t input_w_;
  int64_t out_backprop_h_;
  int64_t out_backprop_w_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_CPU_KERNEL_H_
