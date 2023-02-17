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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RAGGED_TENSOR_TO_TENSOR_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RAGGED_TENSOR_TO_TENSOR_CPU_KERNEL_H_

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "ir/tensor.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
using std::string;
using std::vector;

namespace mindspore {
namespace kernel {
#define TYPE1_flat Eigen::TensorMap<Eigen::Tensor<TYPE1, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
#define TYPE2_flat Eigen::TensorMap<Eigen::Tensor<TYPE2, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>

class RaggedTensorToTensorCpuKernelMod : public NativeCpuKernelMod {
 public:
  RaggedTensorToTensorCpuKernelMod() = default;
  ~RaggedTensorToTensorCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
    return support_list;
  }

 private:
  template <typename TYPE1, typename TYPE2>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename TYPE1>
  bool CalculateOutputIndexValueRowID(const std::vector<TYPE1_flat> &value_rowids,
                                      const vector<TYPE1> &parent_output_index, TYPE1 output_index_multiplier,
                                      TYPE1 output_size, vector<TYPE1> *result);

  template <typename TYPE1>
  bool CalculateOutputIndexRowSplit(const std::vector<TYPE1_flat> &row_split, const vector<TYPE1> &parent_output_index,
                                    TYPE1 output_index_multiplier, TYPE1 output_size, vector<TYPE1> *result);

  template <typename TYPE1, typename TYPE2>
  bool SetOutput(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs,
                 const vector<TYPE1> &output_index);

  int GetRaggedRank(std::vector<std::string> types);

  template <typename TYPE1>
  void GetFirstDimension(const std::vector<kernel::AddressPtr> &inputs, TYPE1 *first_dim);

  TypeId shape_dtype_{kTypeUnknown};
  TypeId values_dtype_{kTypeUnknown};
  std::vector<int64_t> values_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> default_values_shape_;
  std::vector<std::string> row_partition_types_;
  std::vector<std::vector<int64_t>> row_partition_shape_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RMSPROP_CPU_KERNEL_H_
