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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/triangle_matrix_copy_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kCholeskyInputsNum = 1;
constexpr size_t kInputIndex = 0;
constexpr size_t kCholeskyOutputsNum = 1;
constexpr size_t kOutputIndex = 0;
constexpr size_t kRowIndex = 2;
constexpr size_t kColIndex = 1;

template <typename T>
class CholeskySolveGpuKernelMod : public NativeGpuKernelMod {
 public:
  using pointer = T *;
  CholeskySolveGpuKernelMod() = default;
  ~CholeskySolveGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    constexpr size_t input_num = 1;
    constexpr size_t output_num = 1;
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
    kernel_name_ = base_operator->GetPrim()->name();
    if (base_operator->HasAttr("upper")) {
      upper_ = GetValue<bool>(base_operator->GetAttr("upper"));
    }
    // Gpu input is col major default, so need to change row major.
    // In order to speedup it, just change lower to upper, because of cholesky input a is triangle matrix
    // when input b_col is not equal to one, maybe need a normal transpose op inplace.
    if (upper_) {
      uplo_ = CUBLAS_FILL_MODE_LOWER;
    } else {
      uplo_ = CUBLAS_FILL_MODE_UPPER;
    }
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();

    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
      return ret;
    }
    ResetResource();
    auto in_a_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
    auto in_b_shape = LongVecToSizeVec(inputs[kIndex1]->GetShapeVector());
    (void)InitDim(in_a_shape, in_b_shape);
    return KRET_OK;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                  "cholesky solve cusolverDnSetStream failed");
    auto input_a_addr = GetDeviceAddress<T>(inputs, kDim0);
    auto input_b_addr = GetDeviceAddress<T>(inputs, kDim1);
    auto output_addr = GetDeviceAddress<T>(outputs, kDim0);
    auto d_a_array_addr = GetDeviceAddress<pointer>(workspace, kDim0);
    auto d_b_array_addr = GetDeviceAddress<pointer>(workspace, kDim1);
    auto d_info_array_addr = GetDeviceAddress<int>(workspace, kDim2);
    for (size_t i = 0; i < outer_batch_; i++) {
      h_a_array_[i] = input_a_addr + i * lda_ * m_;
      h_b_array_[i] = input_b_addr + i * ldb_ * nrhs_;
    }
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(d_a_array_addr, h_a_array_.data(), sizeof(pointer) * outer_batch_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cuda memcopy Fail");
    CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
      cudaMemcpyAsync(d_b_array_addr, h_b_array_.data(), sizeof(pointer) * outer_batch_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cuda memcopy Fail");
    //  Only support rhs = 1
    if constexpr (std::is_same_v<T, float>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnSpotrsBatched(handle_, uplo_, m_, nrhs_, d_a_array_addr, lda_, d_b_array_addr, ldb_,
                                d_info_array_addr, outer_batch_),
        "cusolver cholesky solve batched Fail");
    } else if constexpr (std::is_same_v<T, double>) {
      CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
        cusolverDnDpotrsBatched(handle_, uplo_, m_, nrhs_, d_a_array_addr, lda_, d_b_array_addr, ldb_,
                                d_info_array_addr, outer_batch_),
        "cusolver cholesky solve batched Fail");
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the data type only should be float or double, right now.";
    }
    size_t output_elements = outputs.at(kDim0)->size / unit_size_;
    // Copy results from written input's matrix to output's matrix.
    MatrixCopy(input_b_addr, output_addr, output_elements, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  void ResetResource() {
    input_size_list_.clear();
    workspace_size_list_.clear();
    output_size_list_.clear();
    h_b_array_.clear();
    h_a_array_.clear();
  }

 protected:
  void InitSizeLists() {
    size_t input_size = outer_batch_ * m_ * lda_ * unit_size_;
    input_size_list_.emplace_back(input_size);
    input_size = outer_batch_ * nrhs_ * ldb_ * unit_size_;
    input_size_list_.emplace_back(input_size);

    size_t workspace_size = outer_batch_ * sizeof(pointer);
    workspace_size_list_.emplace_back(workspace_size);
    workspace_size_list_.emplace_back(workspace_size);
    workspace_size = outer_batch_ * sizeof(int);
    workspace_size_list_.emplace_back(workspace_size);

    size_t output_size = outer_batch_ * m_ * unit_size_;
    output_size_list_.push_back(output_size);
  }

 private:
  void InitDim(const std::vector<size_t> &in_a_shape, const std::vector<size_t> &in_b_shape) {
    constexpr size_t min_dim = 1;
    if (in_a_shape.size() <= min_dim) {
      MS_LOG_EXCEPTION << kernel_name_ << " input a shape dim is " << in_a_shape.size() << " which is invalid.";
    }
    cho_row_ = in_a_shape.at(in_a_shape.size() - kRowIndex);
    cho_col_ = in_a_shape.at(in_a_shape.size() - kColIndex);
    outer_batch_ = min_dim;
    for (int batch = 0; batch < static_cast<int>(in_a_shape.size() - kRowIndex); ++batch) {
      outer_batch_ *= in_a_shape.at(batch);
    }
    if (cho_row_ != cho_col_) {
      MS_LOG_EXCEPTION << kernel_name_ << " input shape is invalid. "
                       << "Cholesky expects a square matrix. but input a shape is: " << cho_row_ << ", " << cho_col_;
    }
    const bool is_right_equal_left = in_a_shape.size() == in_b_shape.size();
    size_t b_row;
    if (is_right_equal_left) {
      b_row = in_b_shape.at(in_b_shape.size() - kRowIndex);
    } else {
      b_row = in_b_shape.back();
    }
    if (cho_row_ != b_row) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', right hand matrix should be equal to left matrix";
    }
    m_ = SizeToInt(cho_row_);
    lda_ = m_;
    ldb_ = m_;
    h_a_array_.resize(outer_batch_);
    h_b_array_.resize(outer_batch_);
    InitSizeLists();
  }
  size_t cho_row_{0};
  size_t cho_col_{0};
  size_t unit_size_{sizeof(T)};
  size_t nrhs_{1};
  size_t outer_batch_{0};
  size_t m_{0};
  size_t lda_{0};
  size_t ldb_{0};
  cusolverDnHandle_t handle_{nullptr};
  cublasFillMode_t uplo_ = CUBLAS_FILL_MODE_UPPER;
  std::vector<pointer> h_a_array_;
  std::vector<pointer> h_b_array_;
  bool upper_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_CHOLESKY_SOLVE_GPU_KERNEL_H_
