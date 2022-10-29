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
constexpr int kParallel = 28;
constexpr int kParallelunit = 1024;
constexpr int kPInit = 2;
constexpr auto kEps = 1e-6;
constexpr int kInitParam = 1;
constexpr auto kInputNumber = 4;
constexpr auto kOutputNumber = 1;
constexpr auto kNumber1 = 1;
class TripletMarginLossCPUKernelMod : public NativeCpuKernelMod {
 public:
  TripletMarginLossCPUKernelMod() = default;
  ~TripletMarginLossCPUKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

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

  int64_t p_{kPInit};
  bool swap_{false};
  float eps_{kEps};
  std::string reduction_{MEAN};
  size_t kParallelDataNum_{kParallel * kParallelunit};
  TypeId dtype_0_{kTypeUnknown};
  ShapeVector x_shape_;
  ShapeVector positive_shape_;
  ShapeVector negative_shape_;
  ShapeVector broadcast_shape_;
  ShapeVector x_reshape_vector_;
  ShapeVector positive_reshape_vector_;
  ShapeVector negative_reshape_vector_;
  size_t numelements_{kInitParam};
  size_t data_num_{kInitParam};
  size_t data_num_each_batch_{kInitParam};
  size_t index_{kInitParam};
  size_t batch_size_{kInitParam};
  size_t once_compute_size_{kInitParam};
  bool broadcast_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIPLET_MARGIN_LOSS_CPU_KERNEL_H_
