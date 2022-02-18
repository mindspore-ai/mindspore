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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LU_SOLVE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LU_SOLVE_GPU_KERNEL_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "utils/convert_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/transpose_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class LuSolveGpuKernelMod : public NativeGpuKernelMod {
 public:
  LuSolveGpuKernelMod() = default;
  ~LuSolveGpuKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCusolverDnHandle();
    auto input_a_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto input_b_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = (CHECK_SHAPE_NULL(input_a_shape, kernel_name_, " lu solve input a") &&
                      (CHECK_SHAPE_NULL(input_b_shape, kernel_name_, " lu solve input b")));
    if (is_null_input_) {
      MS_LOG(EXCEPTION) << "For 'LuSolveGpuKernelMod', input shape is null, please your input.";
    }
    if (!InitInputSize(kernel_node)) {
      MS_LOG(EXCEPTION) << "For 'LuSolveGpuKernelMod', input shape init failed.";
    }
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    CHECK_CUSOLVER_RET_WITH_ERROR(cusolverDnSetStream(handle_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                  "LuSolve kernel cusolverDnSetStream failed");
    T *batch_input_a_addr = GetDeviceAddress<T>(inputs, kDim0);
    T *batch_input_b_addr = GetDeviceAddress<T>(inputs, kDim1);
    T *batch_output_addr = GetDeviceAddress<T>(outputs, kDim0);

    int *info_output_addr = GetDeviceAddress<int>(workspace, kDim0);
    size_t *dev_transpose_a_shape = GetDeviceAddress<size_t>(workspace, kDim1);
    size_t *dev_transpose_a_axis = GetDeviceAddress<size_t>(workspace, kDim2);
    size_t *dev_transpose_b_shape = GetDeviceAddress<size_t>(workspace, kDim3);
    size_t *dev_transpose_b_axis = GetDeviceAddress<size_t>(workspace, kDim4);

    constexpr size_t shape_2d = 2;
    size_t host_transpose_a_shape[shape_2d] = {a_row_, a_col_};
    size_t host_transpose_a_axis[shape_2d] = {1, 0};
    size_t host_transpose_b_shape[shape_2d] = {b_row_, b_col_};
    size_t host_transpose_b_axis[shape_2d] = {1, 0};

    T *dev_transpose_a_work = GetDeviceAddress<T>(workspace, kDim5);
    T *dev_transpose_b_work = GetDeviceAddress<T>(workspace, kDim6);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_transpose_a_axis, host_transpose_a_axis, shape_2d * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "memcpy input a axis workspace failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_transpose_b_axis, host_transpose_b_axis, shape_2d * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "memcpy input b axis workspace failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_transpose_a_shape, host_transpose_a_shape, shape_2d * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "memcpy input a shape workspace failed");

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_transpose_b_shape, host_transpose_b_shape, shape_2d * sizeof(size_t),
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "memcpy input b shape workspace failed");

    // actually output's shape equals to input b's shape.
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(batch_output_addr, batch_input_b_addr, outer_batch_ * a_col_ * b_col_ * unit_size_,
                      cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed in LuSolveGpuKernelMod::Launch.");

    for (size_t batch = 0; batch < outer_batch_; ++batch) {
      T *output_addr = batch_output_addr + batch * a_col_ * b_col_;
      T *input_a_addr = batch_input_a_addr + batch * a_row_ * a_col_;

      CalTranspose(a_row_ * a_col_, input_a_addr, dev_transpose_a_shape, dev_transpose_a_axis, shape_2d,
                   dev_transpose_a_work, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalTranspose(a_col_ * b_col_, output_addr, dev_transpose_b_shape, dev_transpose_b_axis, shape_2d,
                   dev_transpose_b_work, reinterpret_cast<cudaStream_t>(stream_ptr));

      if constexpr (std::is_same_v<T, float>) {
        CHECK_CUSOLVER_RET_WITH_EXCEPT(kernel_node_,
                                       cusolverDnSgetrs(handle_, CUBLAS_OP_N, m_, 1, dev_transpose_a_work, lda_, NULL,
                                                        dev_transpose_b_work, ldb_, info_output_addr),
                                       "cusolver lu fail");
      } else if constexpr (std::is_same_v<T, double>) {
        CHECK_CUSOLVER_RET_WITH_EXCEPT(kernel_node_,
                                       cusolverDnDgetrs(handle_, CUBLAS_OP_N, m_, 1, dev_transpose_a_work, lda_, NULL,
                                                        dev_transpose_b_work, ldb_, info_output_addr),
                                       "cusolver lu fail");
      } else {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the data type only should be float or double, right now.";
      }
      CalTranspose(a_col_ * b_col_, dev_transpose_b_work, dev_transpose_b_shape, dev_transpose_b_axis, shape_2d,
                   output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

 private:
  bool InitInputSize(const CNodePtr &kernel_node) {
    auto input_a_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto input_b_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    constexpr size_t input_min_dim = 1;
    if (input_a_shape.size() <= input_min_dim || input_b_shape.size() <= input_min_dim) {
      MS_LOG_EXCEPTION << kernel_name_ << " LuSolveGpuKernelMod input shape size is " << input_a_shape.size()
                       << " which is invalid.";
    }
    constexpr size_t input_reverse_row_dim = 2;
    a_row_ = input_a_shape.at(input_a_shape.size() - input_reverse_row_dim);
    a_col_ = input_a_shape.at(input_a_shape.size() - 1);
    if (a_row_ != a_col_) {
      MS_LOG_EXCEPTION << kernel_name_ << "LuSolveGpuKernelMod input a is not square matrix, please check : " << a_row_
                       << " vs " << a_col_;
    }

    b_row_ = input_b_shape.at(input_b_shape.size() - input_reverse_row_dim);
    b_col_ = input_b_shape.at(input_b_shape.size() - 1);

    if (a_row_ != b_row_) {
      MS_LOG_EXCEPTION << kernel_name_ << " LuSolveGpuKernelMod input a's row " << a_row_
                       << " is not equal to input b's row " << b_row_ << " which is invalid.";
    }

    outer_batch_ = 1;
    for (int batch = 0; batch < static_cast<int>(input_a_shape.size() - input_reverse_row_dim); ++batch) {
      outer_batch_ *= input_b_shape.at(batch);
    }
    // set matrix row or col to be lead dimension
    m_ = SizeToInt(a_row_);
    n_ = SizeToInt(a_col_);
    lda_ = m_;
    ldb_ = n_;
    InitSizeLists();
    return true;
  }

  void InitSizeLists() override {
    size_t input_a_size = outer_batch_ * a_row_ * b_col_ * unit_size_;
    size_t input_b_size = outer_batch_ * b_row_ * b_col_ * unit_size_;
    input_size_list_.emplace_back(input_a_size);
    input_size_list_.emplace_back(input_b_size);

    // for ax = b --> output x shape [outer_batch_, a_col, b_col]
    size_t output_size = outer_batch_ * a_col_ * b_col_ * unit_size_;
    output_size_list_.emplace_back(output_size);

    // a device addr to place lu solve return code.
    workspace_size_list_.emplace_back(sizeof(int));

    // transpose 2d matrix scalar args workspace
    constexpr size_t shape_2d = 2;
    workspace_size_list_.emplace_back(shape_2d * sizeof(size_t));
    workspace_size_list_.emplace_back(shape_2d * sizeof(size_t));
    workspace_size_list_.emplace_back(shape_2d * sizeof(size_t));
    workspace_size_list_.emplace_back(shape_2d * sizeof(size_t));
    // transpose workspace
    workspace_size_list_.emplace_back(a_row_ * a_col_ * unit_size_);
    workspace_size_list_.emplace_back(b_row_ * b_col_ * unit_size_);
  }

  size_t unit_size_{sizeof(T)};
  size_t outer_batch_{0};
  size_t a_row_{0};
  size_t a_col_{0};
  size_t b_row_{0};
  size_t b_col_{0};
  size_t m_{0};
  size_t n_{0};
  size_t lda_{0};
  size_t ldb_{0};
  cusolverDnHandle_t handle_{nullptr};
  bool is_null_input_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_LU_SOLVE_GPU_KERNEL_H_
