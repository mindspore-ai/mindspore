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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALPOOL_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALPOOL_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_pool_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_pool_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t kInputShapeIndexH = 1;
constexpr size_t kInputShapeIndexW = 2;
constexpr size_t kOutputShapeIndexH = 1;
constexpr size_t kOutputShapeIndexW = 2;
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputRowPoolingSequenceIndex = 1;
constexpr size_t kOutputColPoolingSequenceIndex = 2;
constexpr size_t kOutputIndex = 0;

// max grad
constexpr size_t kOrigInputIndex = 0;
constexpr size_t kOrigOutputIndex = 1;
constexpr size_t kOutBackpropIndex = 2;
constexpr size_t kInputRowPoolingSequenceIndex = 3;
constexpr size_t kInputColPoolingSequenceIndex = 4;

// avg grad
constexpr size_t kAvgGradOutBackpropIndex = 1;
constexpr size_t kAvgGradInputRowPoolingSequenceIndex = 2;
constexpr size_t kAvgGradInputColPoolingSequenceIndex = 3;

class FractionalPoolAttr : public GpuKernelAttrBase {
 public:
  FractionalPoolAttr() = default;
  ~FractionalPoolAttr() override = default;
  std::vector<float> pooling_ratio;
  bool pseudo_random;
  bool overlapping;
  bool deterministic;
  int64_t seed;
  int64_t seed2;
};

template <typename T>
class FractionalPoolHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit FractionalPoolHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_fractional_input_ = false;
  }

  virtual ~FractionalPoolHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 1;
    ResetResource();
    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    input_shape_ = input_shapes[0];
    output_shape_ = output_shapes[0];

    size_t cur_size_T = sizeof(T);
    for (const auto &val : output_shape_) {
      cur_size_T *= val;
    }
    int out_flag = 0;
    if (cur_size_T == 0 && out_flag == 0) {
      out_flag = 1;
    }
    output_size_list_.emplace_back(cur_size_T);

    size_t cur_size = sizeof(int64_t);
    row_pooling_shape_ = output_shapes[kOutputRowPoolingSequenceIndex];
    col_pooling_shape_ = output_shapes[kOutputColPoolingSequenceIndex];
    if ((row_pooling_shape_[0] == 0 || col_pooling_shape_[0] == 0) && out_flag == 0) {
      out_flag = 1;
    }
    output_size_list_.emplace_back(cur_size * row_pooling_shape_[0]);
    output_size_list_.emplace_back(cur_size * col_pooling_shape_[0]);

    is_null_fractional_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int GeneratePoolingSequence(int64_t *cum_seq, size_t input_length, size_t output_length, bool pseudo_random,
                              int seed) {
    if (output_length == 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the GeneratePoolingSequence length should be greater than 0 , but got " << output_length;
      return -1;
    }
    std::vector<int64_t> diff;
    if (input_length % output_length == 0) {
      diff = std::vector<int64_t>(output_length, input_length / output_length);
    } else {
      if (pseudo_random) {
        std::vector<int64_t> cum_seq_tmp(output_length + 1, 0);
        std::vector<int64_t> diff_(output_length, 0);

        // generate random number u which is in (0,1)
        double alpha = static_cast<double>(input_length) / output_length;
        int k = input_length / output_length;
        double u_max1 = (k + 2) / alpha - 1;
        double u_max2 = (input_length + 1 - k) / alpha - (output_length - 1);
        double max_u = std::min(u_max1, u_max2);
        std::default_random_engine random(seed);
        std::uniform_real_distribution<double> dis2(0.0, 1.0);
        const double u = dis2(random) * max_u;
        cum_seq_tmp[0] = 1;
        cum_seq_tmp[output_length] = input_length + 1;
        for (size_t i = 1; i < output_length; ++i) {
          cum_seq_tmp[i] = static_cast<int>(ceil(alpha * (i + u)));
        }
        for (size_t i = 0; i < output_length; ++i) {
          diff_[i] = cum_seq_tmp[i + 1] - cum_seq_tmp[i];
        }
        diff = diff_;
      } else {
        int k = input_length / output_length;
        size_t num_random_spot = input_length % output_length;
        std::vector<int64_t> diff_(output_length, k);
        for (size_t i = 0; i < num_random_spot; ++i) {
          diff_[i] += 1;
        }
        std::shuffle(diff_.begin(), diff_.end(), std::default_random_engine(seed));
        diff = diff_;
      }
    }
    int k = input_length / output_length;
    for (size_t i = 0; i < output_length; i++) {
      if (diff[i] < k || diff[i] > k + 1) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the GeneratePoolingSequence diff[" << i
                      << " should be in the range [-" << k << "," << (k + 1) << "), but got " << diff[i];
        return -1;
      }
    }
    cum_seq[0] = 0;
    for (size_t i = 1; i < output_length + 1; ++i) {
      cum_seq[i] = cum_seq[i - 1] + diff[i - 1];
    }
    return 0;
  }

  int InitSeed(int seed_, int seed2_, bool deterministic_) {
    std::random_device rd;
    std::mt19937 generator(rd());
    int seed = seed_;
    int seed2 = seed2_;
    if (deterministic_) {
      // If both seeds are not set when deterministic is true, force set seeds.
      if ((seed_ == 0) && (seed2_ == 0)) {
        seed = generator();
        seed2 = generator();
      }
    } else {
      if ((seed_ != 0) || (seed2_ != 0)) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', both 'seed' and 'seed2' must be 0 if 'deterministic' is false, but got " << seed_
                      << " and " << seed2_ << ".";
        return -1;
      }
    }
    if (seed_ == 0 && seed2_ != 0) {
      seed = seed2;
    }
    return seed;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_fractional_input_) {
      return 0;
    }
    T *input_ptr = nullptr;
    T *output_ptr = nullptr;

    int64_t *row_pooling_sequence = nullptr;
    int64_t *col_pooling_sequence = nullptr;

    int flag = GetDeviceAddress<T>(input_ptrs, kInputIndex, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, kOutputIndex, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<int64_t>(output_ptrs, kOutputRowPoolingSequenceIndex, kernel_name_, &row_pooling_sequence);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<int64_t>(output_ptrs, kOutputColPoolingSequenceIndex, kernel_name_, &col_pooling_sequence);
    if (flag != 0) {
      return flag;
    }

    int seed = InitSeed(seed_, seed2_, deterministic_);
    // Generate pooling sequence.
    std::vector<int64_t> height_cum_seq(output_shape_[kOutputShapeIndexH] + 1);
    std::vector<int64_t> width_cum_seq(output_shape_[kOutputShapeIndexW] + 1);
    flag = GeneratePoolingSequence(height_cum_seq.data(), input_shape_[kInputShapeIndexH],
                                   output_shape_[kOutputShapeIndexH], pseudo_random_, seed);
    if (flag != 0) {
      return flag;
    }
    flag = GeneratePoolingSequence(width_cum_seq.data(), input_shape_[kInputShapeIndexW],
                                   output_shape_[kOutputShapeIndexW], pseudo_random_, seed);
    if (flag != 0) {
      return flag;
    }

    auto cuda_ret = cudaMemcpy(row_pooling_sequence, height_cum_seq.data(),
                               sizeof(int64_t) * (output_shape_[kOutputShapeIndexH] + 1), cudaMemcpyHostToDevice);
    if (cuda_ret != 0) {
      MS_LOG(ERROR) << "copy mem failed,ret " << cudaGetErrorName(cuda_ret);
      return -1;
    }
    cuda_ret = cudaMemcpy(col_pooling_sequence, width_cum_seq.data(),
                          sizeof(int64_t) * (output_shape_[kOutputShapeIndexW] + 1), cudaMemcpyHostToDevice);
    if (cuda_ret != 0) {
      MS_LOG(ERROR) << "copy mem failed,ret " << cudaGetErrorName(cuda_ret);
      return -1;
    }

    int64_t dims = static_cast<int64_t>(output_shape_.size());
    int64_t outer_size = 1;
    for (int64_t i = dims - 1; i >= 0; i--) {
      outer_size *= output_shape_[i];
    }
    if (kernel_name_.find("FractionalMaxPool") != std::string::npos) {
      CalFractionalmaxpool(input_ptr, output_ptr, row_pooling_sequence, col_pooling_sequence, input_shape_,
                           output_shape_, overlapping_, outer_size, device_id_,
                           reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      CalFractionalavgpool(input_ptr, output_ptr, row_pooling_sequence, col_pooling_sequence, input_shape_,
                           output_shape_, overlapping_, outer_size, device_id_,
                           reinterpret_cast<cudaStream_t>(cuda_stream));
    }
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<FractionalPoolAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    pooling_ratio_ = attr_ptr_->pooling_ratio;
    pseudo_random_ = attr_ptr_->pseudo_random;
    overlapping_ = attr_ptr_->overlapping;
    deterministic_ = attr_ptr_->deterministic;
    seed_ = attr_ptr_->seed;
    seed2_ = attr_ptr_->seed2;
    return 0;
  }

 private:
  std::vector<float> pooling_ratio_;
  bool pseudo_random_{false};
  bool overlapping_{false};
  bool deterministic_{false};
  int64_t seed_{0};
  int64_t seed2_{0};
  std::shared_ptr<FractionalPoolAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> row_pooling_shape_;
  std::vector<int64_t> col_pooling_shape_;
  bool is_null_fractional_input_;
};

class FractionalPoolGradAttr : public GpuKernelAttrBase {
 public:
  FractionalPoolGradAttr() = default;
  ~FractionalPoolGradAttr() override = default;
  bool overlapping;
};

template <typename T>
class FractionalPoolGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit FractionalPoolGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_fractional_grad_input_ = false;
    is_max_pooling_grad_ = (kernel_name_.find("FractionalMaxPoolGrad") != std::string::npos);
  }

  int IsNullInput(size_t cur_size, int inp_flag) {
    if (cur_size == 0 && inp_flag == 0) {
      return 1;
    } else {
      return inp_flag;
    }
  }

  virtual ~FractionalPoolGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int inp_flag = 0;
    size_t row_pooling_index =
      is_max_pooling_grad_ ? kInputRowPoolingSequenceIndex : kAvgGradInputRowPoolingSequenceIndex;
    size_t col_pooling_index =
      is_max_pooling_grad_ ? kInputColPoolingSequenceIndex : kAvgGradInputColPoolingSequenceIndex;
    size_t input3_index = is_max_pooling_grad_ ? kOutBackpropIndex : kAvgGradOutBackpropIndex;
    size_t cur_size_T = sizeof(T);
    if (is_max_pooling_grad_) {
      orig_input_shape_ = input_shapes[kOrigInputIndex];
      for (const auto &val : orig_input_shape_) {
        cur_size_T *= val;
      }
      inp_flag = IsNullInput(cur_size_T, inp_flag);
      input_size_list_.emplace_back(cur_size_T);

      orig_output_shape_ = input_shapes[kOrigOutputIndex];
      cur_size_T = sizeof(T);
      for (const auto &val : orig_output_shape_) {
        cur_size_T *= val;
      }
      inp_flag = IsNullInput(cur_size_T, inp_flag);
      input_size_list_.emplace_back(cur_size_T);
    } else {
      orig_input_shape_ = input_shapes[kOrigInputIndex];
      size_t cur_size_input_shape = sizeof(int64_t);
      inp_flag = IsNullInput(orig_input_shape_[0], inp_flag);
      input_size_list_.emplace_back(cur_size_input_shape * orig_input_shape_[0]);
    }

    out_backprop_shape_ = input_shapes[input3_index];
    cur_size_T = sizeof(T);
    for (const auto &val : out_backprop_shape_) {
      cur_size_T *= val;
    }
    inp_flag = IsNullInput(cur_size_T, inp_flag);
    input_size_list_.emplace_back(cur_size_T);
    row_pooling_shape_ = input_shapes[row_pooling_index];
    col_pooling_shape_ = input_shapes[col_pooling_index];
    size_t cur_size = sizeof(int64_t);
    if ((row_pooling_shape_[0] == 0 || col_pooling_shape_[0] == 0) && inp_flag == 0) {
      inp_flag = 1;
    }
    input_size_list_.emplace_back(cur_size * row_pooling_shape_[0]);
    input_size_list_.emplace_back(cur_size * col_pooling_shape_[0]);

    output_shape_ = output_shapes[kOutputIndex];
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }

    is_null_fractional_grad_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_fractional_grad_input_) {
      return 0;
    }

    if (is_max_pooling_grad_) {
      T *orig_input_ptr = nullptr;
      T *orig_output_ptr = nullptr;
      T *out_backprop_ptr = nullptr;

      int64_t *row_pooling_sequence = nullptr;
      int64_t *col_pooling_sequence = nullptr;

      T *output_ptr = nullptr;

      int flag = GetDeviceAddress<T>(input_ptrs, kOrigInputIndex, kernel_name_, &orig_input_ptr);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<T>(input_ptrs, kOrigOutputIndex, kernel_name_, &orig_output_ptr);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<T>(input_ptrs, kOutBackpropIndex, kernel_name_, &out_backprop_ptr);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<int64_t>(input_ptrs, kInputRowPoolingSequenceIndex, kernel_name_, &row_pooling_sequence);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<int64_t>(input_ptrs, kInputColPoolingSequenceIndex, kernel_name_, &col_pooling_sequence);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<T>(output_ptrs, kOutputIndex, kernel_name_, &output_ptr);
      if (flag != 0) {
        return flag;
      }

      int64_t dims = static_cast<int64_t>(output_shape_.size());

      int64_t backprop_size = 1;
      for (int64_t i = dims - 1; i >= 0; i--) {
        backprop_size *= out_backprop_shape_[i];
      }

      int64_t outer_size = 1;
      for (int64_t i = dims - 1; i >= 0; i--) {
        outer_size *= output_shape_[i];
      }
      CalFractionalmaxpoolgrad(orig_input_ptr, orig_output_ptr, out_backprop_ptr, row_pooling_sequence,
                               col_pooling_sequence, output_ptr, out_backprop_shape_, output_shape_, overlapping_,
                               backprop_size, outer_size, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      int64_t *orig_input_ptr = nullptr;
      T *out_backprop_ptr = nullptr;

      int64_t *row_pooling_sequence = nullptr;
      int64_t *col_pooling_sequence = nullptr;

      T *output_ptr = nullptr;

      int flag = GetDeviceAddress<int64_t>(input_ptrs, kOrigInputIndex, kernel_name_, &orig_input_ptr);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<T>(input_ptrs, kAvgGradOutBackpropIndex, kernel_name_, &out_backprop_ptr);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<int64_t>(input_ptrs, kAvgGradInputRowPoolingSequenceIndex, kernel_name_,
                                       &row_pooling_sequence);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<int64_t>(input_ptrs, kAvgGradInputColPoolingSequenceIndex, kernel_name_,
                                       &col_pooling_sequence);
      if (flag != 0) {
        return flag;
      }

      flag = GetDeviceAddress<T>(output_ptrs, kOutputIndex, kernel_name_, &output_ptr);
      if (flag != 0) {
        return flag;
      }

      int64_t dims = static_cast<int64_t>(output_shape_.size());

      int64_t backprop_size = 1;
      for (int64_t i = dims - 1; i >= 0; i--) {
        backprop_size *= out_backprop_shape_[i];
      }

      int64_t outer_size = 1;
      for (int64_t i = dims - 1; i >= 0; i--) {
        outer_size *= output_shape_[i];
      }
      CalFractionalavgpoolgrad(orig_input_ptr, out_backprop_ptr, row_pooling_sequence, col_pooling_sequence, output_ptr,
                               out_backprop_shape_, output_shape_, overlapping_, backprop_size, outer_size, device_id_,
                               reinterpret_cast<cudaStream_t>(cuda_stream));
    }

    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<FractionalPoolGradAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    overlapping_ = attr_ptr_->overlapping;
    return 0;
  }

 private:
  bool overlapping_{false};
  std::shared_ptr<FractionalPoolGradAttr> attr_ptr_;
  std::vector<int64_t> orig_input_shape_;
  std::vector<int64_t> orig_output_shape_;
  std::vector<int64_t> out_backprop_shape_;
  std::vector<int64_t> row_pooling_shape_;
  std::vector<int64_t> col_pooling_shape_;
  std::vector<int64_t> output_shape_;
  bool is_null_fractional_grad_input_;
  bool is_max_pooling_grad_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_FRACTIONALPOOL_HELPER_H_
