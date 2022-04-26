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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_DETERMINANT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_DETERMINANT_GPU_KERNEL_H_
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MatrixDeterminantGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixDeterminantGpuKernelMod() { ResetResource(); }
  ~MatrixDeterminantGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
              const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

  void ResetResource() noexcept {
    is_null_input_ = false;
    cuda_stream_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();

    input_size_list_.emplace_back(input_elements_ * unit_size_);

    // The workspace for device middle lu output size.
    size_t middle_output_size = input_elements_ * unit_size_;
    // The workspace for device batch lu device address.
    size_t batch_lu_address_size = batch_size_ * sizeof(void *);
    // The workspace for device lu pivot size.
    size_t pivot_size = batch_size_ * m_ * sizeof(int);
    // The workspace for device return info.
    size_t info_size = batch_size_ * sizeof(int);
    // The workspace for device transpose row <--> col.
    // The input will flatten into (batch_size, m, m) for matrix_determinant.
    constexpr size_t three_dims = 3;
    size_t device_input_shape_size = three_dims * sizeof(size_t);
    size_t device_input_axis_size = three_dims * sizeof(size_t);
    workspace_size_list_.emplace_back(middle_output_size);
    workspace_size_list_.emplace_back(batch_lu_address_size);
    workspace_size_list_.emplace_back(pivot_size);
    workspace_size_list_.emplace_back(info_size);
    workspace_size_list_.emplace_back(device_input_shape_size);
    workspace_size_list_.emplace_back(device_input_axis_size);

    output_size_list_.emplace_back(output_elements_ * unit_size_);
    // For LogMatrixDetermine, there are two outputs.
    if (is_sign_log_determinant_) {
      output_size_list_.emplace_back(output_elements_ * unit_size_);
    }
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using MatrixDeterminantFunc =
    std::function<bool(MatrixDeterminantGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

 private:
  bool is_null_input_{false};
  bool is_sign_log_determinant_{false};
  size_t unit_size_{1};
  size_t m_{1};
  size_t batch_size_{1};
  size_t input_elements_{};
  size_t output_elements_{};
  void *cuda_stream_{nullptr};
  cublasHandle_t cublas_handle_{nullptr};
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<KernelTensorPtr> outputs_ = {};
  MatrixDeterminantFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, MatrixDeterminantFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MATRIX_DETERMINANT_GPU_KERNEL_H_
