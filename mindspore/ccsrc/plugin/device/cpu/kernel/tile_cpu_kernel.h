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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_CPU_KERNEL_H_

#include <complex>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/base/tile_base.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class TileCpuKernelMod : public NativeCpuKernelMod {
 public:
  TileCpuKernelMod() = default;
  ~TileCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
      KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex64)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeComplex64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex128)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeComplex128),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
      KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeComplex64),
      KernelAttr()
        .AddInputAttr(kNumberTypeComplex128)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeComplex128),
      KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
      KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
      KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
      KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
      KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
      KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
      KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
      KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
      KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
      KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128)};
    return support_list;
  }

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  void TileMultipleCompute(void);

  ShapeVector x_shape_;
  ShapeVector y_shape_;
  ShapeVector multiple_shape;
  size_t input_num;
  size_t output_num;
  std::vector<int> multiples_;
  TypeId dtype_{kTypeUnknown};
  TypeId multiple_dtype_{kTypeUnknown};

  using TypeKernel = std::function<void(TileCpuKernelMod *, const std::vector<AddressPtr> &inputs,
                                        const std::vector<AddressPtr> &outputs)>;
  std::unordered_map<TypeId, TypeKernel> launch_map_;
  TypeKernel launch_func_;
  TileParameter tile_parameter_;
  bool one_dim_tile_{false};
  size_t input_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_CPU_KERNEL_H_
