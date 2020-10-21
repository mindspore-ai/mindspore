/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UNIFORM_SAMPLER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UNIFORM_SAMPLER_GPU_KERNEL_H_

#include <cmath>
#include <set>
#include <vector>
#include <random>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/uniform_sampler_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class UniformSamplerGpuKernel : public GpuKernel {
 public:
  UniformSamplerGpuKernel() : num_true_(0), num_sampled_(0), unique_(false), range_max_(0), input_size_(0) {}
  ~UniformSamplerGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspaces);
    T *sampled_candidates = GetDeviceAddress<T>(outputs, 0);
    S *true_expected_count = GetDeviceAddress<S>(outputs, 1);
    S *sampled_expected_count = GetDeviceAddress<S>(outputs, 2);
    int counter = Sampling();
    float prob = Probability();
    size_t sampled_candidates_size = num_sampled_ * sizeof(T);
    S value = ApproximateExpectedCount(prob, num_sampled_, counter);
    CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(sampled_candidates, &sampled_candidates_[0], sampled_candidates_size,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync sampled_candidates failed");
    CalUniformSampler(static_cast<int>(input_size_), num_sampled_, value, true_expected_count, sampled_expected_count,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but UniformSampler needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 3) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but UniformSampler has 3 outputs.";
      return false;
    }
    // getting attrs
    num_true_ = GetAttr<int>(kernel_node, "num_true");
    num_sampled_ = GetAttr<int>(kernel_node, "num_sampled");
    unique_ = GetAttr<bool>(kernel_node, "unique");
    range_max_ = GetAttr<int>(kernel_node, "range_max");
    int seed = GetAttr<int>(kernel_node, "seed");
    if (seed == 0) seed = time(NULL);
    generator_.seed(seed);
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    if (input_shape.size() != 2) {
      MS_LOG(ERROR) << "Input is " << input_shape.size() << "-D, but UniformSampler supports only 2-D inputs.";
      return false;
    }
    input_size_ = input_shape[0] * input_shape[1];
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(num_sampled_ * sizeof(T));
    output_size_list_.push_back(input_size_ * sizeof(S));
    output_size_list_.push_back(num_sampled_ * sizeof(S));
  }

  int Sampling() {
    int counter = 0;
    int tmp;
    int picked;
    std::set<int> set_container;
    // pick between [0, range_max_-1]
    std::uniform_int_distribution<int> distribution(0, range_max_ - 1);
    sampled_candidates_.clear();
    if (unique_) {
      picked = 0;
      while (picked < num_sampled_) {
        tmp = distribution(generator_);
        counter++;
        if (set_container.find(tmp) == set_container.end()) {
          set_container.insert(tmp);
          sampled_candidates_.push_back(tmp);
          picked++;
        }
      }
    } else {
      for (int i = 0; i < num_sampled_; i++) {
        sampled_candidates_.push_back(distribution(generator_));
      }
      counter = num_sampled_;
    }
    return counter;
  }

  S Probability() { return static_cast<S>(1.0f / range_max_); }

  S ApproximateExpectedCount(S p, int sampled_size, int counter) {
    if (sampled_size == counter) return p * sampled_size;
    return -std::expm1(counter * std::log1p(-p));
  }

 private:
  int num_true_;
  int num_sampled_;
  bool unique_;
  int range_max_;
  size_t input_size_;
  std::default_random_engine generator_;
  std::vector<int> sampled_candidates_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_UNIFORM_SAMPLER_GPU_KERNEL_H_
