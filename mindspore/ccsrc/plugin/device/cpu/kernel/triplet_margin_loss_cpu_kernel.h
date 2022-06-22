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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIPLET_MARGIN_LOSS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIPLET_MARGIN_LOSS_CPU_KERNEL_H_
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <map>
#include <complex>
#include <iostream>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class TripletMarginLossCPUKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  TripletMarginLossCPUKernelMod() = default;
  ~TripletMarginLossCPUKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt8)
                                                     .AddInputAttr(kNumberTypeInt8)
                                                     .AddInputAttr(kNumberTypeInt8)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt8)
                                                     .AddInputAttr(kNumberTypeUInt8)
                                                     .AddInputAttr(kNumberTypeUInt8)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt16)
                                                     .AddInputAttr(kNumberTypeUInt16)
                                                     .AddInputAttr(kNumberTypeUInt16)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt32)
                                                     .AddInputAttr(kNumberTypeUInt32)
                                                     .AddInputAttr(kNumberTypeUInt32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeUInt64)
                                                     .AddInputAttr(kNumberTypeUInt64)
                                                     .AddInputAttr(kNumberTypeUInt64)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeComplex64)
                                                     .AddInputAttr(kNumberTypeComplex64)
                                                     .AddInputAttr(kNumberTypeComplex64)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeComplex128)
                                                     .AddInputAttr(kNumberTypeComplex128)
                                                     .AddInputAttr(kNumberTypeComplex128)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void TripletMarginLossCompute_realtype(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void TripletMarginLossCompute_complextype(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &outputs);

  template <typename T>
  void realtype_broadcast_task(size_t start, size_t end, float *output_reduction_none_data,
                               const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void realtype_nobroadcast_task(size_t start, size_t end, float *output_reduction_none_data,
                                 const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void complextype_broadcast_task(size_t start, size_t end, float *output_reduction_none_data,
                                  const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void complextype_nobroadcast_task(size_t start, size_t end, float *output_reduction_none_data,
                                    const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void realtype_nobroadcast_compute(float *output_reduction_none_data, const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void realtype_broadcast_compute(float *output_reduction_none_data, const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void complextype_nobroadcast_compute(float *output_reduction_none_data, const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void complextype_broadcast_compute(float *output_reduction_none_data, const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void realtype_swap(size_t start, std::vector<T> &positive_broadcast, std::vector<T> &negative_broadcast,
                     std::vector<float> &calculate_swap, size_t j, size_t k, float &calc_swap_sum,
                     const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  template <typename T>
  void complextype_swap(size_t start, std::vector<T> &positive_broadcast, std::vector<T> &negative_broadcast,
                        std::vector<T> &calculate_swap, size_t j, size_t k, float &calc_swap_sum,
                        const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  void CheckParam(const CNodePtr &kernel_node);
  int64_t p = 2;
  bool swap = false;
  float eps = 1e-6;
  std::string reduction = MEAN;
  size_t input_num = 1;
  size_t output_num = 1;
  size_t kParallelDataNum = 1;
  TypeId dtype_0{kTypeUnknown};
  TypeId dtype_1{kTypeUnknown};
  TypeId dtype_2{kTypeUnknown};
  TypeId dtype_3{kTypeUnknown};
  ShapeVector x_shape;
  ShapeVector positive_shape;
  ShapeVector negative_shape;
  ShapeVector broadcast_shape;
  ShapeVector x_reshape_vector;
  ShapeVector positive_reshape_vector;
  ShapeVector negative_reshape_vector;
  size_t numelements = 1;
  size_t data_num = 1;
  size_t data_num_each_batch = 1;
  size_t index = 1;
  size_t batch_size = 1;
  size_t once_compute_size = 1;
  bool broadcast = false;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIPLET_MARGIN_LOSS_CPU_KERNEL_H_
