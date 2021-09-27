/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT3D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT3D_GPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/dropout3d_impl.cuh"
#include "include/curand.h"

namespace mindspore {
namespace kernel {
template <typename T>
class Dropout3DGpuFwdKernel : public GpuKernel {
 public:
  Dropout3DGpuFwdKernel() { ResetResource(); }
  ~Dropout3DGpuFwdKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    bool *mask_addr = GetDeviceAddress<bool>(outputs, 1);
    float *rand_f = GetDeviceAddress<float>(workspace, 0);

    if (!states_init_) {
      CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT),
                                   "Failed to create generator");
      CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(curand_generator_, time(NULL)),
                                   "Failed to SetPseudoRandomGeneratorSeed");
      MS_EXCEPTION_IF_NULL(curand_generator_);
      states_init_ = true;
    }

    CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(curand_generator_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "Failed to set stream for generator");
    // curandGen only supports float or double.
    // generate random float for every channel
    CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateUniform(curand_generator_, rand_f, num_chan_),
                                 "Failed to generate uniform");

    Dropout3DForward(input_addr, mask_addr, output_addr, rand_f, num_count_, keep_prob_, num_per_chan_,
                     reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but Dropout3DGpuFwdKernel needs 1.";
    }

    std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "Dropout3DGpuKernel input is null.";
      InitSizeLists();
      return true;
    }

    CheckTensorSize({input_shape});

    size_t dims = input_shape.size();
    if (dims != 5) {
      MS_LOG(EXCEPTION) << "Input dims " << dims << "not supported. Must be in NCDHW format.";
    }

    // get N and C values from 5 dim input tensor
    n_ = input_shape[0];
    c_ = input_shape[1];

    num_count_ = 1;
    for (size_t i = 0; i < dims; i++) {
      num_count_ *= input_shape[i];
    }

    num_chan_ = n_ * c_;
    MS_EXCEPTION_IF_ZERO("num channel", num_chan_);
    num_per_chan_ = num_count_ / num_chan_;  // number of elements per channel

    keep_prob_ = GetAttr<float>(kernel_node, "keep_prob");
    if ((keep_prob_ < 0.0) || (keep_prob_ > 1.0)) {
      MS_LOG(EXCEPTION) << "keep_prob is out of range [0.0, 1.0]";
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    is_null_input_ = false;
    num_count_ = 0;
    keep_prob_ = 0.0;
    states_init_ = false;
    curand_generator_ = nullptr;
    n_ = 0;
    c_ = 0;
    num_chan_ = 0;
    num_per_chan_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t input_size = num_count_ * sizeof(T);
    size_t mask_size = num_count_ * sizeof(bool);
    input_size_list_.push_back(input_size);
    output_size_list_.push_back(input_size);  // output size: the same as input size
    output_size_list_.push_back(mask_size);

    size_t workspace_size = num_chan_ * sizeof(float);  // rand_f for curandGen
    workspace_size_list_.push_back(workspace_size);
  }

 private:
  cudnnHandle_t cudnn_handle_;
  curandGenerator_t curand_generator_;
  bool is_null_input_;
  bool states_init_;
  size_t num_count_;
  size_t n_;
  size_t c_;
  size_t num_chan_;
  size_t num_per_chan_;
  float keep_prob_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_DROPOUT3D_GPU_KERNEL_H_
