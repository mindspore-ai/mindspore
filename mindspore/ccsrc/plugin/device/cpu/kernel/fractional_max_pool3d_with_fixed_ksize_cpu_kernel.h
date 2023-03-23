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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL3D_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL3D_WITH_FIXED_KSIZE_CPU_KERNEL_H_
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/fractional_max_pool3d_with_fixed_ksize.h"

namespace mindspore {
namespace kernel {
class FractionalMaxPool3DWithFixedKsizeCPUKernelMod : public NativeCpuKernelMod {
 public:
  FractionalMaxPool3DWithFixedKsizeCPUKernelMod() = default;
  ~FractionalMaxPool3DWithFixedKsizeCPUKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename scalar_t, typename random_sample_t, typename argmax_t>
  bool ComputeTemplate(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename scalar_t, typename random_sample_t, typename argmax_t>
  bool FractionalMaxPool3DWithFixedKsizeCompute(scalar_t *inputForPlane, random_sample_t *random_samplesForPlane,
                                                argmax_t *argmaxForPlane, scalar_t *outputForPlane, int64_t outputD,
                                                int64_t outputH, int64_t outputW, int64_t kernelsizeD,
                                                int64_t kernelsizeH, int64_t kernelsizeW, int64_t inputC,
                                                int64_t inputD, int64_t inputH, int64_t inputW);
  template <typename scalar_t, typename random_sample_t>
  bool DoComputeWithArgmaxType(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                               TypeId argmax_type);
  template <typename scalar_t>
  bool DoComputeWithRandomSamplesType(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                                      TypeId random_samples_type);
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> random_samples_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> ksize_;
  std::string data_format_;
  TypeId input_type_;
  TypeId random_samples_type_;
  TypeId argmax_type_;
  int64_t inputN_;
  int64_t inputC_;
  int64_t inputD_;
  int64_t inputH_;
  int64_t inputW_;
  int64_t outputD_;
  int64_t outputH_;
  int64_t outputW_;
  int64_t kernelsizeD_;
  int64_t kernelsizeH_;
  int64_t kernelsizeW_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_FRACTIONAL_MAX_POOL3D_WITH_FIXED_KSIZE_CPU_KERNEL_H_
