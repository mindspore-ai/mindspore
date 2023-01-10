/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CDIST_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CDIST_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <functional>
#include <map>
#include "mindspore/core/ops/cdist.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
namespace mindspore {
namespace kernel {
class CdistCpuKernelMod : public NativeCpuKernelMod {
 public:
  CdistCpuKernelMod() {}
  ~CdistCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool LaunchKernel(int64_t start, int64_t end);

  void InitFunc(float p);

  using DistFunc = std::function<void(const float *a, const float *b, float *c, int64_t m, float p)>;
  DistFunc dist_func_;

  int64_t batch_;
  int64_t r0_;
  int64_t m_;
  int64_t r1_;
  float p_ = 2;
  void *in_data0_;
  void *in_data1_;
  void *out_data_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CDIST_CPU_KERNEL_H_
