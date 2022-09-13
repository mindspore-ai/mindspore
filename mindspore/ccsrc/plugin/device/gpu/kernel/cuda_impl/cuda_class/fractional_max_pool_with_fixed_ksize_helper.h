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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_max_pool_with_fixed_ksize_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t kDimSize2 = 2;
constexpr size_t kDimSize3 = 3;
constexpr size_t kDimSize4 = 4;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kkernelsizeIndexH = 0;
constexpr size_t kkernelsizeIndexW = 1;
constexpr size_t kOutputshapeIndexH = 0;
constexpr size_t kOutputshapeIndexW = 1;
constexpr size_t kFormatNCHWIndexN = 0;
constexpr size_t kFormatNCHWIndexC = 1;
constexpr size_t kFormatNCHWIndexH = 2;
constexpr size_t kFormatNCHWIndexW = 3;

class FractionalMaxPoolWithFixedKsizeAttr : public GpuKernelAttrBase {
 public:
  FractionalMaxPoolWithFixedKsizeAttr() = default;
  ~FractionalMaxPoolWithFixedKsizeAttr() override = default;
  std::vector<int64_t> ksize;
  std::vector<int64_t> output_shape;
  std::string data_format;
};

template <typename T, typename S, typename G>
class FractionalMaxPoolWithFixedKsizeHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit FractionalMaxPoolWithFixedKsizeHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~FractionalMaxPoolWithFixedKsizeHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();

    input_shape_ = input_shapes[kInputIndex0];
    random_samples_shape_ = input_shapes[kInputIndex1];
    output_shape_ = output_shapes[kOutputIndex0];
    argmax_shape_ = output_shapes[kOutputIndex1];

    data_format_ = attr_ptr_->data_format;
    if (data_format_ != "NCHW") {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', data_format must be NCHW, but got " << data_format_ << ".";
      return -1;
    }
    inputN_ = input_shape_[kFormatNCHWIndexN];
    inputC_ = input_shape_[kFormatNCHWIndexC];
    inputH_ = input_shape_[kFormatNCHWIndexH];
    inputW_ = input_shape_[kFormatNCHWIndexW];
    int inp_flag = 0;
    size_t cur_size_T = sizeof(T);
    for (const auto &val : input_shape_) {
      cur_size_T *= val;
    }
    if (cur_size_T == 0 && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(cur_size_T);

    size_t cur_size_S = sizeof(S);
    for (const auto &val : random_samples_shape_) {
      cur_size_S *= val;
    }
    if (cur_size_S == 0 && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(cur_size_S);

    int out_flag = 0;
    cur_size_T = sizeof(T);
    for (const auto &val : output_shape_) {
      cur_size_T *= val;
    }

    if (cur_size_T == 0 && out_flag == 0) {
      out_flag = 1;
    }
    output_size_list_.emplace_back(cur_size_T);

    size_t cur_size_G = sizeof(G);
    for (const auto &val : argmax_shape_) {
      cur_size_G *= val;
    }
    if (cur_size_G == 0 && out_flag == 0) {
      out_flag = 1;
    }
    output_size_list_.emplace_back(cur_size_G);

    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    S *random_samples_ptr = nullptr;
    T *output_ptr = nullptr;
    G *argmax_ptr = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kInputIndex0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<S>(input_ptrs, kInputIndex1, kernel_name_, &random_samples_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, kOutputIndex0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<G>(output_ptrs, kOutputIndex1, kernel_name_, &argmax_ptr);
    if (flag != 0) {
      return flag;
    }

    int64_t dims = static_cast<int64_t>(output_shape_.size());
    int64_t outer_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      outer_size *= output_shape_[i];
    }

    CalFractionalmaxpoolwithfixedksize(input_ptr, random_samples_ptr, output_ptr, argmax_ptr, outputH_, outputW_,
                                       inputN_, inputC_, inputH_, inputW_, kernelsizeH_, kernelsizeW_, outer_size,
                                       device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<FractionalMaxPoolWithFixedKsizeAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    ksize_ = attr_ptr_->ksize;
    output_shape_attr_ = attr_ptr_->output_shape;
    kernelsizeH_ = ksize_[kkernelsizeIndexH];
    kernelsizeW_ = ksize_[kkernelsizeIndexW];
    outputH_ = output_shape_attr_[kOutputshapeIndexH];
    outputW_ = output_shape_attr_[kOutputshapeIndexW];
    size_t input_num_dims = input_shape_.size();
    size_t random_samples_dims = random_samples_shape_.size();
    size_t output_shape_dims = output_shape_attr_.size();
    size_t ksize_dims = ksize_.size();
    if (input_num_dims != kDimSize4) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'input_x' must be equal to 4, but got "
                    << input_num_dims << ".";
      return -1;
    }
    if (random_samples_dims != kDimSize3) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'random_samples' must be equal to 3, but got "
                    << random_samples_dims << ".";
      return -1;
    }
    if (output_shape_dims != kDimSize2) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'output_shape' must be equal to 2, but got "
                    << output_shape_dims << ".";
      return -1;
    }
    if (ksize_dims != kDimSize2) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'ksize' must be equal to 2, but got "
                    << ksize_dims << ".";
      return -1;
    }
    if (random_samples_shape_[kDimSize2] != kDimSize2) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', expected the third dimension of 'random_samples' must be 2, but got "
                    << random_samples_shape_[kDimSize2] << ".";
      return -1;
    }
    return 0;
  }

 private:
  std::vector<int64_t> ksize_;
  std::vector<int64_t> output_shape_attr_;
  std::string data_format_;
  int64_t outputH_{1};
  int64_t outputW_{1};
  int64_t inputN_{1};
  int64_t inputC_{1};
  int64_t inputH_{1};
  int64_t inputW_{1};
  int64_t kernelsizeH_{1};
  int64_t kernelsizeW_{1};
  std::shared_ptr<FractionalMaxPoolWithFixedKsizeAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> random_samples_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> argmax_shape_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALMAXPOOLWITHFIXEDKSIZE_HELPER_H_
