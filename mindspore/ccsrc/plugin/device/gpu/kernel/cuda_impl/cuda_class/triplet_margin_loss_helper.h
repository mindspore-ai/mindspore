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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_TRIPLET_MARGIN_LOSS_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_TRIPLET_MARGIN_LOSS_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/triplet_margin_loss_impl.cuh"

namespace mindspore {
namespace cukernel {
class TripletMarginLossAttr : public GpuKernelAttrBase {
 public:
  TripletMarginLossAttr() = default;
  ~TripletMarginLossAttr() override = default;
  int64_t p;
  bool swap;
  float eps;
  std::string reduction;
};

template <typename T, typename S>
class TripletMarginLossHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit TripletMarginLossHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    shape_size_ = 0;
    need_broadcast_ = false;
    is_null_input_ = false;
    reduction_ = "mean";
    bound_ = 0;
  }

  virtual ~TripletMarginLossHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM_1 = 3;
    constexpr size_t OUTPUT_NUM = 1;
    constexpr int64_t kzero = 0;
    constexpr int64_t kone = 1;
    constexpr int64_t ktwo = 2;
    constexpr int64_t kthree = 3;
    ResetResource();
    dst_shape_.clear();
    std::vector<std::vector<int64_t>> input_shapes_1;
    input_shapes_1.emplace_back(input_shapes[kzero]);
    input_shapes_1.emplace_back(input_shapes[kone]);
    input_shapes_1.emplace_back(input_shapes[ktwo]);
    int inp_flag =
      CalShapesSizeInBytes<T>(input_shapes_1, INPUT_NUM_1, kernel_name_, "input_shapes_1", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    input_size_list_.emplace_back(sizeof(float));
    int out_flag =
      CalShapesSizeInBytes<S>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    anchor_shape_ = input_shapes[kzero];
    positive_shape_ = input_shapes[kone];
    negative_shape_ = input_shapes[ktwo];
    size_t dim_x = anchor_shape_.size();
    size_t dim_positive = positive_shape_.size();
    size_t dim_negative = negative_shape_.size();
    shape_size_ = std::max(std::max(dim_x, dim_positive), dim_negative);
    std::reverse(anchor_shape_.begin(), anchor_shape_.end());
    std::reverse(positive_shape_.begin(), positive_shape_.end());
    std::reverse(negative_shape_.begin(), negative_shape_.end());
    if (dim_x < shape_size_) anchor_shape_.resize(shape_size_, kone);
    if (dim_positive < shape_size_) positive_shape_.resize(shape_size_, kone);
    if (dim_negative < shape_size_) negative_shape_.resize(shape_size_, kone);
    std::reverse(anchor_shape_.begin(), anchor_shape_.end());
    std::reverse(positive_shape_.begin(), positive_shape_.end());
    std::reverse(negative_shape_.begin(), negative_shape_.end());
    if (anchor_shape_ != positive_shape_ || anchor_shape_ != negative_shape_ || positive_shape_ != negative_shape_) {
      need_broadcast_ = true;
    }
    int64_t tem_shape_size = 0;
    for (size_t i = 0; i < shape_size_; i++) {
      tem_shape_size++;
      dst_shape_.push_back((int64_t)std::max(std::max(anchor_shape_[i], positive_shape_[i]), negative_shape_[i]));
    }
    std::vector<std::vector<int64_t>> workspace_shapes_sizet;
    std::vector<std::vector<int64_t>> workspace_shapes_S;
    constexpr size_t WORKSPACE_SIZET_NUM = 1;
    constexpr size_t WORKSPACE_S_NUM = 1;
    std::vector<int64_t> shape_shape;
    tem_shape_size *= 4;  // store 4 shapes
    shape_shape.push_back(tem_shape_size);
    workspace_shapes_sizet.emplace_back(shape_shape);
    swap_ = attr_ptr_->swap;
    std::vector<int64_t> tem_output_shape(dst_shape_);
    tem_output_shape.erase(tem_output_shape.begin() + 1);
    if (swap_) {
      tem_output_shape.insert(tem_output_shape.begin(), kthree);
    } else {
      tem_output_shape.insert(tem_output_shape.begin(), ktwo);
    }
    workspace_shapes_S.emplace_back(tem_output_shape);
    int work_flag = CalShapesSizeInBytes<int64_t>(workspace_shapes_sizet, WORKSPACE_SIZET_NUM, kernel_name_,
                                                  "workspace_shapes", &work_size_list_);
    if (work_flag == -1) {
      return work_flag;
    }
    work_flag = CalShapesSizeInBytes<float>(workspace_shapes_S, WORKSPACE_S_NUM, kernel_name_, "workspace_shapes",
                                            &work_size_list_);
    if (work_flag == -1) {
      return work_flag;
    }
    size_t workspace_boundlist = kthree * sizeof(size_t);
    work_size_list_.emplace_back(workspace_boundlist);
    if (need_broadcast_) {
      std::vector<std::vector<int64_t>> workspace_shapes_T;
      constexpr size_t WORKSPACE_T_NUM = 1;
      workspace_shapes_T.emplace_back(dst_shape_);
      workspace_shapes_T[0].insert(workspace_shapes_T[0].begin(), kthree);
      work_flag = CalShapesSizeInBytes<T>(workspace_shapes_T, WORKSPACE_T_NUM, kernel_name_, "workspace_shapes",
                                          &work_size_list_);
      if (work_flag == -1) {
        return work_flag;
      }
    }
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    const int64_t kzero = 0;
    const int64_t kone = 1;
    const int64_t ktwo = 2;
    const int64_t kthree = 3;
    const int64_t kfour = 4;
    if (is_null_input_) {
      return 0;
    }
    bound_list_[kzero] = ChooseBound(anchor_shape_[kone], positive_shape_[kone], dst_shape_[kone]);
    bound_list_[kone] = ChooseBound(anchor_shape_[kone], negative_shape_[kone], dst_shape_[kone]);
    bound_list_[ktwo] = ChooseBound(positive_shape_[kone], negative_shape_[kone], dst_shape_[kone]);
    bound_ = dst_shape_[kone];

    size_t outer_size = dst_shape_[kzero];
    size_t inner_size = 1;
    for (size_t i = 2; i < shape_size_; i++) {
      inner_size *= dst_shape_[i];
    }
    T *anchor_ptr = nullptr;
    T *positive_ptr = nullptr;
    T *negative_ptr = nullptr;
    float *margin_ptr = nullptr;
    S *output_ptr = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kzero, kernel_name_, &anchor_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, kone, kernel_name_, &positive_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<T>(input_ptrs, ktwo, kernel_name_, &negative_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<float>(input_ptrs, kthree, kernel_name_, &margin_ptr);
    if (flag != 0) {
      return flag;
    }
    flag = GetDeviceAddress<S>(output_ptrs, kzero, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }
    std::vector<int64_t> input_shapes;
    input_shapes.insert(input_shapes.end(), anchor_shape_.begin(), anchor_shape_.end());
    input_shapes.insert(input_shapes.end(), positive_shape_.begin(), positive_shape_.end());
    input_shapes.insert(input_shapes.end(), negative_shape_.begin(), negative_shape_.end());
    input_shapes.insert(input_shapes.end(), dst_shape_.begin(), dst_shape_.end());
    int64_t *anchor_shape_ptr = nullptr;
    int64_t *dst_shape_ptr = nullptr;
    float *tem_output_ptr = nullptr;
    size_t *bound_list_ptr = nullptr;
    T *anchor_broadcast_ptr = anchor_ptr;
    T *positive_broadcast_ptr = positive_ptr;
    T *negative_broadcast_ptr = positive_ptr;
    flag = GetDeviceAddress<int64_t>(work_ptrs, kzero, kernel_name_, &anchor_shape_ptr);
    if (flag != 0) {
      return flag;
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(anchor_shape_ptr, &input_shapes[kzero], shape_size_ * sizeof(int64_t) * kfour,
                      cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpyAsync workspace failed");

    dst_shape_ptr = anchor_shape_ptr + kthree * shape_size_;

    flag = GetDeviceAddress<float>(work_ptrs, kone, kernel_name_, &tem_output_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<size_t>(work_ptrs, ktwo, kernel_name_, &bound_list_ptr);
    if (flag != 0) {
      return flag;
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(bound_list_ptr, &bound_list_[kzero], sizeof(size_t) * kthree, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "cudaMemcpyAsync workspace failed");

    if (need_broadcast_) {
      flag = GetDeviceAddress<T>(work_ptrs, kthree, kernel_name_, &anchor_broadcast_ptr);
      if (flag != 0) {
        return flag;
      }
      positive_broadcast_ptr = anchor_broadcast_ptr + bound_ * outer_size * inner_size;
      negative_broadcast_ptr = positive_broadcast_ptr + bound_ * outer_size * inner_size;
    }
    auto status = CalTripletMarginLoss(anchor_ptr, positive_ptr, negative_ptr, anchor_broadcast_ptr,
                                       positive_broadcast_ptr, negative_broadcast_ptr, output_ptr, tem_output_ptr,
                                       anchor_shape_ptr, dst_shape_ptr, outer_size, inner_size, bound_list_ptr, bound_,
                                       shape_size_, margin_ptr, attr_ptr_->p, attr_ptr_->eps, reduction_, swap_,
                                       need_broadcast_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<TripletMarginLossAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    std::string reduction_list = "[mean,none,sum]";
    reduction_ = attr_ptr_->reduction;
    if (reduction_ != "mean" && reduction_ != "none" && reduction_ != "sum") {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'reduciton' should be in " << reduction_list
                    << "but got:" << reduction_;
      return -1;
    }
    return 0;
  }

  size_t ChooseBound(size_t src_bound_first, size_t src_bound_second, size_t dst_bound) {
    if (src_bound_first == 1 && src_bound_second == 1 && dst_bound != 1) {
      return 1;
    }
    return dst_bound;
  }

 private:
  std::shared_ptr<TripletMarginLossAttr> attr_ptr_;
  std::vector<int64_t> anchor_shape_, positive_shape_, negative_shape_, dst_shape_;
  size_t shape_size_;
  size_t bound_list_[3];
  size_t bound_;
  bool need_broadcast_;
  bool swap_;
  bool is_null_input_;
  std::string reduction_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_TRIPLET_MARGIN_LOSS_HELPER_H_
