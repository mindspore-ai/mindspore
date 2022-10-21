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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/fractional_max_pool_with_fixed_ksize.h"

namespace mindspore {
namespace kernel {
class FractionalMaxPoolWithFixedKsizeCPUKernelMod : public NativeCpuKernelMod {
 public:
  FractionalMaxPoolWithFixedKsizeCPUKernelMod() = default;
  ~FractionalMaxPoolWithFixedKsizeCPUKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename scalar_t>
  bool DoComputeWithRandomSamplesType(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                                      TypeId random_samples_type) const;
  template <typename scalar_t, typename random_sample_t>
  bool ComputeTemplate(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const;
  template <typename scalar_t, typename random_sample_t>
  void FractionalMaxPoolWithFixedKsizeCompute(scalar_t *inputForPlane, random_sample_t *random_samplesForPlane,
                                              scalar_t *outputForPlane, int64_t *argmaxForPlane) const;
  template <typename random_sample_t>
  std::vector<int> GenerateIntervals(random_sample_t sample, int input_size, int output_size, int kernel_size) const;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> random_samples_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> ksize_;
  std::string data_format_{"NCHW"};
  TypeId input_type_;
  TypeId random_samples_type_;
  TypeId argmax_type_;
  int64_t input_n_;
  int64_t input_c_;
  int64_t input_h_;
  int64_t input_w_;
  int64_t ksize_h_;
  int64_t ksize_w_;
  int64_t output_h_;
  int64_t output_w_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL_WITH_FIXED_KSIZE_CPU_KERNEL_H_
