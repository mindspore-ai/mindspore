/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_CANDIDATE_SAMPLER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_CANDIDATE_SAMPLER_GPU_KERNEL_H_

#include <cmath>
#include <set>
#include <vector>
#include <random>
#include <limits>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/uniform_candidate_sampler_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class UniformCandidateSamplerGpuKernel : public GpuKernel {
 public:
  UniformCandidateSamplerGpuKernel()
      : num_true_(0),
        num_sampled_(0),
        unique_(false),
        range_max_(0),
        input_size_(0),
        remove_accidental_hits_(false),
        is_null_input_(false) {}
  ~UniformCandidateSamplerGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    if (init_seed_ == 0 && cur_seed_ == 0) {
      // Update current seed.
      cur_seed_ = time(NULL);
      generator_.seed(cur_seed_);
    }
    VARIABLE_NOT_USED(workspaces);
    T *sampled_candidates = GetDeviceAddress<T>(outputs, 0);
    S *true_expected_count = GetDeviceAddress<S>(outputs, 1);
    S *sampled_expected_count = GetDeviceAddress<S>(outputs, 2);
    if (remove_accidental_hits_) {
      T *input = GetDeviceAddress<T>(inputs, 0);
      array_input_ = std::vector<T>(input_size_, 0);
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(&array_input_[0], input, input_size_ * sizeof(T),
                                                 cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync sampled_candidates failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaDeviceSynchronize(), "cudaDeviceSyncFailed");
      for (const auto item : array_input_) {
        set_input_.insert(item);
      }
    }
    int64_t counter = Sampling();
    S prob = Probability();
    size_t sampled_candidates_size = num_sampled_ * sizeof(T);
    S value = ApproximateExpectedCount(prob, num_sampled_, counter);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(sampled_candidates, &sampled_candidates_[0], sampled_candidates_size,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync sampled_candidates failed");
    CalUniformCandidateSampler(static_cast<int64_t>(input_size_), num_sampled_, value, true_expected_count,
                               sampled_expected_count, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but UniformCandidateSampler needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 3) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but UniformCandidateSampler has 3 outputs.";
      return false;
    }
    // getting attrs
    num_true_ = GetAttr<int64_t>(kernel_node, "num_true");
    num_sampled_ = GetAttr<int64_t>(kernel_node, "num_sampled");
    unique_ = GetAttr<bool>(kernel_node, "unique");
    range_max_ = GetAttr<int64_t>(kernel_node, "range_max");
    remove_accidental_hits_ = GetAttr<bool>(kernel_node, "remove_accidental_hits");
    init_seed_ = GetAttr<int64_t>(kernel_node, "seed");
    if (init_seed_ == 0) {
      cur_seed_ = time(NULL);
      generator_.seed(cur_seed_);
    } else {
      generator_.seed(init_seed_);
    }

    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'UniformCandidateSamplerGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    if (input_shape.size() != 2) {
      MS_LOG(ERROR) << "Input is " << input_shape.size() << "-D, but UniformCandidateSampler supports only 2-D inputs.";
      return false;
    }
    input_size_ = input_shape[0] * input_shape[1];
    if (num_sampled_ + static_cast<int64_t>(input_size_) > range_max_) {
      remove_accidental_hits_ = false;
    }
    InitSizeLists();
    return true;
  }

  void ReleaseResource() override {
    // Reset current seed.
    if (init_seed_ == 0 && cur_seed_ != 0) {
      cur_seed_ = 0;
    }

    if (init_seed_ != 0) {
      generator_.seed(init_seed_);
    }
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(num_sampled_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(S));
    output_size_list_.push_back(num_sampled_ * sizeof(S));
  }

  int64_t Sampling() {
    int64_t counter = 0;
    T tmp;
    int64_t picked;
    std::set<T> set_container;
    // pick between [0, range_max_-1]
    T range;
    if (range_max_ > static_cast<int64_t>(std::numeric_limits<T>::max())) {
      MS_LOG(EXCEPTION) << "range_max_ failed to cast";
    }
    range = static_cast<T>(range_max_);
    std::uniform_int_distribution<T> distribution(0, range - 1);
    sampled_candidates_.clear();
    if (unique_) {
      picked = 0;
      while (picked < num_sampled_) {
        tmp = distribution(generator_);
        counter++;
        if ((set_container.find(tmp) == set_container.end()) &&
            ((!remove_accidental_hits_) || set_input_.find(tmp) == set_input_.end())) {
          set_container.insert(tmp);
          sampled_candidates_.push_back(tmp);
          picked++;
        }
      }
    } else {
      for (int64_t i = 0; i < num_sampled_; i++) {
        sampled_candidates_.push_back(distribution(generator_));
      }
      counter = num_sampled_;
    }
    return counter;
  }

  S Probability() {
    S range;
    if (range_max_ > static_cast<int64_t>(std::numeric_limits<S>::max())) {
      MS_LOG(EXCEPTION) << "range_max_ failed to cast";
    }
    range = static_cast<S>(range_max_);
    MS_EXCEPTION_IF_ZERO("range", range);
    return static_cast<S>(1.0f / range);
  }

  S ApproximateExpectedCount(S p, int64_t sampled_size, int64_t counter) {
    if (sampled_size == counter) return p * sampled_size;
    return -std::expm1(counter * std::log1p(-p));
  }

 private:
  int64_t num_true_;
  int64_t num_sampled_;
  bool unique_;
  int64_t range_max_;
  size_t input_size_;
  bool remove_accidental_hits_;
  bool is_null_input_;
  std::vector<T> array_input_;
  std::set<T> set_input_;
  std::default_random_engine generator_;
  std::vector<T> sampled_candidates_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int64_t init_seed_{0};
  int64_t cur_seed_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_UNIFORM_CANDIDATE_SAMPLER_GPU_KERNEL_H_
