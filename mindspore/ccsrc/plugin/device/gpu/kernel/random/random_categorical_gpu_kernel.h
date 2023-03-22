/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_categorical.cuh"
#include "mindspore/core/ops/random_categorical.h"

namespace mindspore {
namespace kernel {
template <typename T, typename G, typename S>
class RandomCategoricalGpuKernelMod : public NativeGpuKernelMod {
 public:
  RandomCategoricalGpuKernelMod() : is_null_input_(false), batch_size_(0), num_classes_(0), num_samples_(0), seed_(0) {}
  ~RandomCategoricalGpuKernelMod() override = default;

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
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
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
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(host_rand[i],  // NOLINT
                        host_1d_rand.get(), sizeof(double) * num_samples_, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "Random_categorica cudaMemcpyAsync host_1d_rand failed");
    }
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(dev_rand,  // NOLINT
                      host_rand.get(), sizeof(double *) * batch_size_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "Random_categorica cudaMemcpyAsync dev_rand failed");

    GetCdfKernel(logits_addr, dev_cdf, batch_size_, num_classes_, reinterpret_cast<cudaStream_t>(stream_ptr));
    RandomCategoricalKernel(num_samples_, dev_rand, dev_cdf, batch_size_, num_classes_, output_addr,
                            reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
    if (ret != KRET_OK) {
      return ret;
    }
    auto kernel_ptr = std::dynamic_pointer_cast<ops::RandomCategorical>(base_operator);
    MS_EXCEPTION_IF_NULL(kernel_ptr);

    size_t input_num = inputs.size();
    const size_t kRandomCategoricalInputSize = 3;
    if (input_num != kRandomCategoricalInputSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 3, but got " << input_num;
    }
    size_t output_num = outputs.size();
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    num_samples_ = kernel_ptr->get_num_sample();
    seed_ = kernel_ptr->get_seed();

    auto logits_shape = inputs[0]->GetShapeVector();
    const size_t kLogitsShapeSize = 2;
    if (logits_shape.size() != kLogitsShapeSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of logits should be 2, but got "
                        << logits_shape.size();
    }
    batch_size_ = LongToSizeClipNeg(logits_shape[0]);
    num_classes_ = LongToSizeClipNeg(logits_shape[1]);
    InitSizeLists();
    return ret;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    if (kernel_name_ != prim::kPrimRandomCategorical->name()) {
      MS_LOG(ERROR) << "For 'RandomCategorical', the kernel name must be 'RandomCategorical', but got " << kernel_name_;
      return false;
    }
    return true;
  }

 protected:
  void InitResource() override {}
  void InitSizeLists() {
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
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_RANDOM_RANDOM_CATEGORICAL_GPU_KERNEL_H_
