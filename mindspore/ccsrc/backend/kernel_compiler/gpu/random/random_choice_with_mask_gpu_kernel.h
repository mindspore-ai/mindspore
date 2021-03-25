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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_

#include <vector>
#include <chrono>
#include <random>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/random_choice_with_mask_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class RandomChoiceWithMaskGpuKernel : public GpuKernel {
 public:
  RandomChoiceWithMaskGpuKernel()
      : input_shape_size_(0), seed_(0), seed2_(0), input_size_(1), count_(0), ceil_power2_(0) {}
  ~RandomChoiceWithMaskGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input = GetDeviceAddress<T>(inputs, 0);
    S *output_index = GetDeviceAddress<S>(outputs, 0);
    T *output_mask = GetDeviceAddress<T>(outputs, 1);
    int seedc = 0;
    if (seed2_ != 0) {
      seedc = seed2_;
    } else if (seed_ != 0) {
      seedc = seed_;
    } else {
      seedc = generator_();
    }
    if (count_ > kSmallK || input_shape_size_ > 1) {
      S *index_buff = GetDeviceAddress<S>(workspaces, 0);
      S *mask_buff = GetDeviceAddress<S>(workspaces, 1);
      S *rank_buff = GetDeviceAddress<S>(workspaces, 2);
      S *Tnum_buff = GetDeviceAddress<S>(workspaces, 3);
      S *tmp_buff = GetDeviceAddress<S>(workspaces, 4);
      void *States = GetDeviceAddress<void *>(workspaces, 5);
      curandState *devStates = reinterpret_cast<curandState *>(States);
      CalRandomChoiceWithMask(input_size_, input_shape_size_, input_shape_5D_[0], input_shape_5D_[1],
                              input_shape_5D_[2], input_shape_5D_[3], input_shape_5D_[4], seedc, count_, input,
                              output_index, output_mask, index_buff, mask_buff, rank_buff, Tnum_buff, tmp_buff,
                              devStates, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CalRandomChoiceWithMaskSmall<float, S, T>(input_size_, seedc, count_, input, output_index, output_mask,
                                                reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    uint32_t time_interval = std::chrono::system_clock::now().time_since_epoch().count();
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but RandomChoiceWithMask needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 2) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but RandomChoiceWithMask has 2 outputs.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_shape_size_ = input_shape.size();
    if (input_shape_size_ < 1 || input_shape_size_ > MAX_DIMENSION) {
      MS_LOG(ERROR) << "Input is " << input_shape_size_
                    << "-D, but RandomChoiceWithMask supports only 1-D to 5-D inputs.";
      return false;
    }
    // convert size_t to int
    for (auto i = 0; i < input_shape_size_; i++) {
      input_shape_5D_.push_back(input_shape[i]);
    }
    // convert shape to 5D
    while (input_shape_5D_.size() != MAX_DIMENSION) {
      input_shape_5D_.insert(input_shape_5D_.begin(), 1);
    }
    // init seedc
    seed_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "seed"));
    seed2_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "seed2"));
    generator_.seed(time_interval);
    // init memory
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    count_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "count"));
    // upper ceiling for input for ceil_power2
    if (count_ > kSmallK || input_shape_size_ > 1) {
      ceil_power2_ = RcwmRoundUpPower2(input_size_);
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(count_ * input_shape_size_ * sizeof(S));
    output_size_list_.push_back(count_ * sizeof(T));
    if (count_ > kSmallK || input_shape_size_ > 1) {
      workspace_size_list_.push_back(input_size_ * input_shape_size_ * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(S));
      int blocknum = std::ceil(static_cast<float>(ceil_power2_) / BLOCKSIZE);
      workspace_size_list_.push_back(blocknum * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(S));
      workspace_size_list_.push_back(ceil_power2_ * sizeof(curandState));
    }
  }

 private:
  const int kSmallK = 2048;
  int input_shape_size_;
  int seed_;
  int seed2_;
  int input_size_;
  int count_;
  int ceil_power2_;
  std::mt19937 generator_;
  std::vector<int> input_shape_5D_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CHOICE_WITH_MASK_GPU_KERNEL_H_
