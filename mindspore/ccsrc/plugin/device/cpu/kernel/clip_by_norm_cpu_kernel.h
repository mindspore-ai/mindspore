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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CLIP_BY_NORM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CLIP_BY_NORM_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <functional>
#include "mindspore/core/ops/clip_by_norm.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ClipByNormCpuKernelMod : public NativeCpuKernelMod {
 public:
  ClipByNormCpuKernelMod() = default;
  ~ClipByNormCpuKernelMod() override = default;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &,
             const std::vector<KernelTensorPtr> &, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  // Init function
  void ResetResource();
  void InitIOShape(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);
  void InitAxisAndEpsilon(const ops::ClipByNormPtr &prim);
  void InitSizeLists();
  // Launch function
  template <typename T, typename S>
  void LaunchFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                  const std::vector<AddressPtr> &outputs);
  // Run `l2_norm(x)` calculation
  template <typename T>
  void L2NormLaunch(const T *x_addr, float *l2_norm_output_addr, size_t l2_norm_output_size);
  // Run `x / l2_norm(x)` calculation
  template <typename T>
  void DivLaunch(const T *x_addr, const float *l2_norm_output_addr, float *div_output_addr, size_t div_output_size);
  // Run `max(x, (x / l2_norm(x)) * clip_norm)` calculation
  // The output data type is float
  template <typename T, typename S>
  void ClipNormMulAndCmpLaunch(const T *x_addr, const float *div_output_addr, const S *clip_norm_addr,
                               float *output_addr, size_t output_size);
  // Basic variables
  float epsilon_{0.000001f};
  size_t x_dim_{0};
  std::pair<TypeId, TypeId> data_type_{kNumberTypeFloat32, kNumberTypeFloat32};
  std::vector<size_t> axis_;
  ShapeVector x_shape_;
  ShapeVector clip_norm_shape_;
  ShapeVector l2_norm_output_shape_;
  ShapeVector output_shape_;
  size_t stride_ = 1;
  std::vector<size_t> l2_norm_index_;
  std::vector<size_t> div_index1_;
  std::vector<size_t> div_index2_;
  std::vector<size_t> mul_index1_;
  std::vector<size_t> mul_index2_;
  ParallelSearchInfo parallel_search_info_div_;
  ParallelSearchInfo parallel_search_info_mul_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CLIP_BY_NORM_CPU_KERNEL_H_
