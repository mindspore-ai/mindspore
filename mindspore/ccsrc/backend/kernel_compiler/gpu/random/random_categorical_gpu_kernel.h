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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CATEGORICAL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CATEGORICAL_GPU_KERNEL_H_

#include <vector>
#include <memory>
#include <random>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/random_categorical.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename G, typename S>
class RandomCategoricalGpuKernel : public GpuKernel {
 public:
  RandomCategoricalGpuKernel() : is_null_input_(false), batch_size_(0), num_classes_(0), num_samples_(0), seed_(0) {}
  ~RandomCategoricalGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *logits_addr = GetDeviceAddress<T>(inputs, 0);
    S *output_addr = GetDeviceAddress<S>(outputs, 0);

    std::unique_ptr<double *[]> host_cdf;
    host_cdf = std::make_unique<double *[]>(batch_size_);
    for (size_t i = 0; i < batch_size_; i++) {
      host_cdf[i] = GetDeviceAddress<double>(workspaces, i);
    }
    double **dev_cdf = GetDeviceAddress<double *>(workspaces, batch_size_);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_cdf,  // NOLINT
                                               host_cdf.get(), sizeof(double *) * batch_size_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Random_categorica cudaMemcpyAsync dev_cdf failed");

    std::unique_ptr<double *[]> host_rand;
    host_rand = std::make_unique<double *[]>(batch_size_);
    for (size_t i = 0; i < batch_size_; i++) {
      host_rand[i] = GetDeviceAddress<double>(workspaces, batch_size_ + 1 + i);
    }

    double **dev_rand = GetDeviceAddress<double *>(workspaces, batch_size_ * 2 + 1);
    for (size_t i = 0; i < batch_size_; i++) {
      std::unique_ptr<double[]> host_1d_rand;
      host_1d_rand = std::make_unique<double[]>(num_samples_);

      std::default_random_engine rng(static_cast<G>(seed_));
      std::uniform_real_distribution<> dist(0, 1);
      for (size_t j = 0; j < num_samples_; j++) {
        host_1d_rand[j] = dist(rng);
      }
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(host_rand[i],  // NOLINT
                                                 host_1d_rand.get(), sizeof(double) * num_samples_,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Random_categorica cudaMemcpyAsync host_1d_rand failed");
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(dev_rand,  // NOLINT
                                               host_rand.get(), sizeof(double *) * batch_size_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Random_categorica cudaMemcpyAsync dev_rand failed");

    GetCdfKernel(logits_addr, dev_cdf, batch_size_, num_classes_, reinterpret_cast<cudaStream_t>(stream_ptr));
    RandomCategoricalKernel(num_samples_, dev_rand, dev_cdf, batch_size_, num_classes_, output_addr,
                            reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but RandomCategorical needs 3 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but RandomCategorical has 1 output.";
      return false;
    }
    auto logits_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(logits_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'RandomCategoricalGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    if (logits_shape.size() != 2) {
      MS_LOG(ERROR) << "logits's dims is " << logits_shape.size() << ", but it should be only 2-D.";
      return false;
    }
    batch_size_ = logits_shape[0];
    num_classes_ = logits_shape[1];

    num_samples_ = LongToSize(GetAttr<int64_t>(kernel_node, "num_samples"));
    seed_ = GetAttr<int64_t>(kernel_node, "seed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {}
  void InitSizeLists() override {
    // init memory
    input_size_list_.push_back(sizeof(T) * batch_size_ * num_classes_);
    input_size_list_.push_back(sizeof(G));
    input_size_list_.push_back(sizeof(G));
    output_size_list_.push_back(sizeof(S) * batch_size_ * num_samples_);

    for (size_t i = 0; i < batch_size_; i++) {
      workspace_size_list_.push_back(sizeof(double) * num_classes_);
    }
    workspace_size_list_.push_back(sizeof(double *) * batch_size_);

    for (size_t i = 0; i < batch_size_; i++) {
      workspace_size_list_.push_back(sizeof(double) * num_samples_);
    }
    workspace_size_list_.push_back(sizeof(double *) * batch_size_);
  }

 private:
  bool is_null_input_;
  size_t batch_size_;
  size_t num_classes_;
  size_t num_samples_;
  int64_t seed_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CATEGORICAL_GPU_KERNEL_H_
