/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MULTILABEL_MARGIN_LOSS_GRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MULTILABEL_MARGIN_LOSS_GRAD_HELPER_H_
#include <malloc.h>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/multilabel_margin_loss_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
class MultilabelMarginLossGradAttr : public GpuKernelAttrBase {
 public:
  MultilabelMarginLossGradAttr() = default;
  ~MultilabelMarginLossGradAttr() override = default;
  int64_t reduction;
};

template <typename T>
class MultilabelMarginLossGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit MultilabelMarginLossGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    batch_size_ = 1;
    class_num_ = 0;
    is_null_input_ = false;
  }

  virtual ~MultilabelMarginLossGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    constexpr size_t INPUT_NUM_1 = 2;
    constexpr size_t INPUT_NUM_2 = 2;
    constexpr size_t OUTPUT_NUM = 1;
    constexpr int64_t kInputGradIndex = 0;
    constexpr int64_t kInputXIndex = 1;
    constexpr int64_t kInputTargetIndex = 2;
    constexpr int64_t kInputIsTargetIndex = 3;
    y_grad_shape_ = input_shapes[kInputGradIndex];
    x_shape_ = input_shapes[kInputXIndex];
    target_shape_ = input_shapes[kInputTargetIndex];
    is_target_shape_ = input_shapes[kInputIsTargetIndex];
    std::vector<std::vector<int64_t>> input_shapes_1, input_shapes_2;
    input_shapes_1.emplace_back(y_grad_shape_);
    input_shapes_1.emplace_back(x_shape_);
    input_shapes_2.emplace_back(target_shape_);
    input_shapes_2.emplace_back(is_target_shape_);

    int inp_flag1 =
      CalShapesSizeInBytes<T>(input_shapes_1, INPUT_NUM_1, kernel_name_, "input_shapes_1", &input_size_list_);
    if (inp_flag1 == -1) {
      return inp_flag1;
    }
    int inp_flag2 =
      CalShapesSizeInBytes<int>(input_shapes_2, INPUT_NUM_2, kernel_name_, "input_shapes_2", &input_size_list_);
    if (inp_flag2 == -1) {
      return inp_flag2;
    }
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag1 == 1 || inp_flag2 == 1 || out_flag == 1);

    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    const int64_t kInputGradIndex = 0;
    const int64_t kInputXIndex = 1;
    const int64_t kInputTargetIndex = 2;
    const int64_t kInputIsTargetIndex = 3;
    const int64_t kOutputGradIndex = 0;
    T *y_grad_ptr = nullptr;
    T *x_ptr = nullptr;
    int *target_ptr = nullptr;
    int *is_target_ptr = nullptr;
    T *x_grad_ptr = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kInputGradIndex, kernel_name_, &y_grad_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kInputXIndex, kernel_name_, &x_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(input_ptrs, kInputTargetIndex, kernel_name_, &target_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(input_ptrs, kInputIsTargetIndex, kernel_name_, &is_target_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, kOutputGradIndex, kernel_name_, &x_grad_ptr);
    if (flag != 0) {
      return flag;
    }
    reduction_ = attr_ptr_->reduction;
    cudaMemsetAsync(x_grad_ptr, 0, sizeof(T) * batch_size_ * class_num_, reinterpret_cast<cudaStream_t>(cuda_stream));
    CalMultilabelMarginLossGrad(y_grad_ptr, x_ptr, target_ptr, is_target_ptr, batch_size_, class_num_, reduction_,
                                x_grad_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<MultilabelMarginLossGradAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    reduction_ = attr_ptr_->reduction;
    int64_t dims = static_cast<int64_t>(x_shape_.size());
    int64_t min_dim = 1, max_dim = 2;
    if (dims < min_dim || dims > max_dim) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dim of input should be 0 or 1 or 2 but got " << dims;
      return -1;
    } else if (dims <= 1) {
      batch_size_ = 1;
      class_num_ = dims == 0 ? 1 : x_shape_[0];
    } else {
      batch_size_ = x_shape_[0];
      class_num_ = x_shape_[1];
    }
    return 0;
  }

 private:
  int64_t reduction_;
  int batch_size_;
  int class_num_;
  std::shared_ptr<MultilabelMarginLossGradAttr> attr_ptr_;
  std::vector<int64_t> y_grad_shape_;
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> target_shape_;
  std::vector<int64_t> is_target_shape_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MULTILABEL_MARGIN_LOSS_GRAD_HELPER_H_
