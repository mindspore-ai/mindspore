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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MULTILABEL_MARGIN_LOSS_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MULTILABEL_MARGIN_LOSS_HELPER_H_
#include <malloc.h>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/multilabel_margin_loss_impl.cuh"

namespace mindspore {
namespace cukernel {
class MultilabelMarginLossAttr : public GpuKernelAttrBase {
 public:
  MultilabelMarginLossAttr() = default;
  ~MultilabelMarginLossAttr() override = default;
  int64_t reduction;
};

template <typename T>
class MultilabelMarginLossHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit MultilabelMarginLossHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    batch_size_ = 1;
    class_num_ = 0;
    is_null_input_ = false;
  }

  virtual ~MultilabelMarginLossHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    constexpr size_t INPUT_NUM_1 = 1;
    constexpr size_t INPUT_NUM_2 = 1;
    constexpr size_t OUTPUT_NUM_1 = 1;
    constexpr size_t OUTPUT_NUM_2 = 1;
    constexpr size_t WORKSPACE_NUM = 1;
    input_shape_ = input_shapes[0];
    target_shape_ = input_shapes[1];
    std::vector<std::vector<int64_t>> input_shapes_1, input_shapes_2;
    std::vector<std::vector<int64_t>> output_shapes_1, output_shapes_2;
    std::vector<std::vector<int64_t>> workspace_shapes_T;
    input_shapes_1.emplace_back(input_shape_);
    input_shapes_2.emplace_back(target_shape_);
    output_shapes_1.emplace_back(output_shapes[0]);
    output_shapes_2.emplace_back(output_shapes[1]);
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
    int out_flag1 =
      CalShapesSizeInBytes<T>(output_shapes_1, OUTPUT_NUM_1, kernel_name_, "output_shapes_1", &output_size_list_);
    if (out_flag1 == -1) {
      return out_flag1;
    }
    int out_flag2 =
      CalShapesSizeInBytes<int>(output_shapes_2, OUTPUT_NUM_2, kernel_name_, "output_shapes_2", &output_size_list_);
    if (out_flag2 == -1) {
      return out_flag2;
    }
    is_null_input_ = (inp_flag1 == 1 || inp_flag2 == 1 || out_flag1 == 1 || out_flag2 == 1);

    int WorkplaceBytes = input_shape_.size() == 1 ? 1 : input_shape_[0];
    std::vector<int64_t> workplace_shape;
    workplace_shape.push_back(WorkplaceBytes);
    workspace_shapes_T.emplace_back(workplace_shape);
    int work_flag =
      CalShapesSizeInBytes<T>(workspace_shapes_T, WORKSPACE_NUM, kernel_name_, "workspace_shapes_T", &work_size_list_);
    if (work_flag == -1) {
      return work_flag;
    }
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    int *target_ptr = nullptr;
    T *output_ptr = nullptr;
    int *is_target_ptr = nullptr;
    T *output_tmp_ptr = nullptr;
    int kBatchSize = batch_size_;
    std::vector<T> output_tmp_host(kBatchSize);
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(input_ptrs, 1, kernel_name_, &target_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<int>(output_ptrs, 1, kernel_name_, &is_target_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(work_ptrs, 0, kernel_name_, &output_tmp_ptr);
    if (flag != 0) {
      return flag;
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpy(output_tmp_ptr, output_tmp_host.data(), batch_size_ * sizeof(T), cudaMemcpyHostToDevice),
      "cudaMemcpy workspace failed");

    reduction_ = attr_ptr_->reduction;

    CalMultilabelMarginLoss(input_ptr, target_ptr, is_target_ptr, batch_size_, class_num_, reduction_, output_ptr,
                            output_tmp_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<MultilabelMarginLossAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    reduction_ = attr_ptr_->reduction;
    int64_t dims = static_cast<int64_t>(input_shape_.size());
    int64_t min_dim = 1, max_dim = 2;
    if (dims < min_dim || dims > max_dim) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dim of input should be 0 or 1 or 2 but got " << dims;
      return -1;
    } else if (dims <= 1) {
      batch_size_ = 1;
      class_num_ = dims == 0 ? 1 : input_shape_[0];
    } else {
      batch_size_ = input_shape_[0];
      class_num_ = input_shape_[1];
    }
    int64_t tgt_dims = static_cast<int64_t>(target_shape_.size());
    if (tgt_dims != dims) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "got the inconsistent target size: [" << tgt_dims
                    << "] for input of size: [" << dims << "]";
      return -1;
    }
    return 0;
  }

 private:
  int64_t reduction_;
  int batch_size_;
  int class_num_;
  std::shared_ptr<MultilabelMarginLossAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> target_shape_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_MULTILABEL_MARGIN_LOSS_HELPER_H_
