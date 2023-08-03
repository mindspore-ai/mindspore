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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NONMAXSUPPRESSIONV3_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NONMAXSUPPRESSIONV3_HELPER_H_
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include "mindspore/core/ops/non_max_suppression_v3.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/cuda_class_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/non_max_suppressionv3_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"

namespace mindspore {
namespace cukernel {
class NonMaxSuppressionV3Attr : public GpuKernelAttrBase {
 public:
  NonMaxSuppressionV3Attr() = default;
  ~NonMaxSuppressionV3Attr() override = default;
};

template <typename T, typename M, typename S>
class NonMaxSuppressionV3HelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit NonMaxSuppressionV3HelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    num_input = 0;
    u_num = 0;
    post_output_size_ = 0;
  }

  virtual ~NonMaxSuppressionV3HelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr int64_t kzero = 0;
    constexpr int64_t kone = 1;
    constexpr int64_t ktwo = 2;
    constexpr int64_t kthree = 3;
    constexpr int64_t kfour = 4;
    constexpr int64_t kfourbytes = 32;
    ResetResource();
    std::vector<std::vector<int64_t>> input_shapes_1;
    std::vector<std::vector<int64_t>> input_shapes_2;
    std::vector<std::vector<int64_t>> input_shapes_3;
    input_shapes_1.emplace_back(input_shapes[kzero]);
    input_shapes_1.emplace_back(input_shapes[kone]);
    input_shapes_2.emplace_back(input_shapes[ktwo]);
    input_shapes_3.emplace_back(input_shapes[kthree]);
    input_shapes_3.emplace_back(input_shapes[kfour]);
    int inp_flag_1 = CalShapesSizeInBytes<T>(input_shapes_1, ktwo, kernel_name_, "input_shapes_1", &input_size_list_);
    if (inp_flag_1 == -1) {
      return inp_flag_1;
    }
    int inp_flag_2 = CalShapesSizeInBytes<S>(input_shapes_2, kone, kernel_name_, "input_shapes_2", &input_size_list_);
    if (inp_flag_2 == -1) {
      return inp_flag_2;
    }
    int inp_flag_3 = CalShapesSizeInBytes<M>(input_shapes_3, ktwo, kernel_name_, "input_shapes_3", &input_size_list_);
    if (inp_flag_3 == -1) {
      return inp_flag_3;
    }
    output_size_list_.emplace_back(input_shapes[kzero][0] * sizeof(int));
    num_input = input_shapes[kzero][0];
    u_num = (num_input + kfourbytes - 1) / kfourbytes;
    work_size_list_.emplace_back(num_input * sizeof(int));                   // index buff
    work_size_list_.emplace_back(num_input * u_num * sizeof(unsigned int));  // sel mask
    work_size_list_.emplace_back(num_input * sizeof(bool));                  // box mask
    work_size_list_.emplace_back(sizeof(int));                               // count
    work_size_list_.emplace_back(sizeof(int));                               // num_keep
    is_null_input_ = (inp_flag_1 == 1 || inp_flag_2 == 1 || inp_flag_3 == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    constexpr int64_t kzero = 0;
    constexpr int64_t kone = 1;
    constexpr int64_t ktwo = 2;
    constexpr int64_t kthree = 3;
    constexpr int64_t kfour = 4;
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    T *scores = nullptr;
    S *max_output_size_ = nullptr;
    M *iou_threshold_ = nullptr;
    M *score_threshold_ = nullptr;
    int *output_ptr = nullptr;
    int *index_buff = nullptr;
    unsigned int *sel_mask = nullptr;
    bool *sel_boxes = nullptr;
    int *count = nullptr;
    int *num_keep = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kzero, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kone, kernel_name_, &scores);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(input_ptrs, ktwo, kernel_name_, &max_output_size_);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<M>(input_ptrs, kthree, kernel_name_, &iou_threshold_);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<M>(input_ptrs, kfour, kernel_name_, &score_threshold_);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(work_ptrs, kzero, kernel_name_, &index_buff);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<unsigned int>(work_ptrs, kone, kernel_name_, &sel_mask);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<bool>(work_ptrs, ktwo, kernel_name_, &sel_boxes);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(work_ptrs, kthree, kernel_name_, &count);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(work_ptrs, kfour, kernel_name_, &num_keep);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(output_ptrs, kzero, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
    M iou_host = 0.0;
    cudaMemcpy(&iou_host, iou_threshold_, sizeof(M), cudaMemcpyDeviceToHost);
    float iou = static_cast<float>(iou_host);
    if (iou > 1 || iou < 0) {
      MS_EXCEPTION(ValueError) << "For NonMaxSuppressionV3, iou_threshold must be in [0, 1], but got " << iou;
      return -1;
    }
    S max_host = 0;
    cudaMemcpy(&max_host, max_output_size_, sizeof(S), cudaMemcpyDeviceToHost);
    int max = static_cast<int32_t>(max_host);
    if (max < 0) {
      max_host = 0;
    }
    M score_host = 0.0;
    cudaMemcpy(&score_host, score_threshold_, sizeof(M), cudaMemcpyDeviceToHost);
    const int b_size = 4;
    auto status =
      DoNms(num_input, count, num_keep, scores, input_ptr, iou_host, score_host, index_buff, max_host, b_size, sel_mask,
            sel_boxes, output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream), &post_output_size_);
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  TensorInfo GetOutputTensorInfo() override {
    TensorInfo dyn_out;
    dyn_out.shapes.push_back({{post_output_size_}});
    return dyn_out;
  }

 private:
  std::vector<int64_t> input_shape_;
  bool is_null_input_;
  int num_input;
  int u_num;
  float iou_threshold;
  int post_output_size_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  //  MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NONMAXSUPPRESSIONV3_GPU_KERNEL_H_
