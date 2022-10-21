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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_PARAMETERIZED_TRUNCATED_NORMAL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_PARAMETERIZED_TRUNCATED_NORMAL_CPU_KERNEL_H_

#include <vector>
#include <random>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
namespace mindspore {
namespace kernel {
class ParameterizedTruncatedNormalCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  ParameterizedTruncatedNormalCpuKernelMod() = default;
  ~ParameterizedTruncatedNormalCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  T GetBatchSizeCheckDims(const std::vector<AddressPtr> &inputs);
  template <typename T>
  void GenerateCase1(const int64_t size, const T norm_min, const T norm_max, const T stddev, const T mean,
                     T **output_ptr);
  template <typename T>
  void GenerateCase2(const int64_t size, const T norm_min, const T norm_max, const T stddev, const T mean,
                     T **output_ptr);
  template <typename T>
  void GenerateCase3(const int64_t size, const T norm_min, const T norm_max, const T stddev, const T mean,
                     T **output_ptr);
  template <typename T>
  void Generate(const int64_t size, const T mean, T stddev, T minval, T maxval, T **output_ptr);
  template <typename T_shape, typename T>
  bool BatchGenerate(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> input_means_shape_;
  std::vector<int64_t> input_stdevs_shape_;
  std::vector<int64_t> input_min_shape_;
  std::vector<int64_t> input_max_shape_;
  std::vector<int64_t> out_shape_;
  TypeId output_type_;
  TypeId input_type_;
  TypeId input_means_type_;
  TypeId input_stdevs_type_;
  TypeId input_min_type_;
  TypeId input_max_type_;
  std::default_random_engine rng;
  int64_t seed_{0};
  int64_t seed2_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PARAMETERIZED_TRUNCATED_NORMAL_CPU_KERNEL_H_
