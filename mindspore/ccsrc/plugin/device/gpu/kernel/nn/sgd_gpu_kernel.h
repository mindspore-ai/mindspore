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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_SGD_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_SGD_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sgd_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "ops/sgd.h"

namespace mindspore {
namespace kernel {
template <typename T>
class SGDGpuKernelMod : public NativeGpuKernelMod {
 public:
  SGDGpuKernelMod() : size_(1), dampening_(0.0), weight_decay_(0.0), nesterov_(false), is_null_input_(false) {}
  ~SGDGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream) override {
    if (is_null_input_) {
      return true;
    }
    T *param = GetDeviceAddress<T>(inputs, 0);
    T *grad = GetDeviceAddress<T>(inputs, 1);
    T *lr = GetDeviceAddress<T>(inputs, 2);
    T *accum = GetDeviceAddress<T>(inputs, 3);
    T *momentum = GetDeviceAddress<T>(inputs, 4);
    T *stat = GetDeviceAddress<T>(inputs, 5);
    T *output_param = GetDeviceAddress<T>(outputs, 0);

    SGD(size_, dampening_, weight_decay_, nesterov_, lr, momentum, grad, param, accum, stat,
        reinterpret_cast<cudaStream_t>(stream));
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(output_param, param, sizeof(T) * size_, cudaMemcpyDeviceToDevice,
                                                       reinterpret_cast<cudaStream_t>(stream)),
                                       kernel_name_ + " SGD cudaMemcpyAsync params to outputs failed");
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    if (inputs.empty() || outputs.empty()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
      return false;
    }
    constexpr size_t kSGDInputsNum = 6;
    if (inputs.size() != kSGDInputsNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSGDInputsNum << ", but got "
                    << inputs.size();
      return false;
    }
    constexpr size_t kSGDOutputsNum = 1;
    if (outputs.size() != kSGDOutputsNum) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size must be " << kSGDOutputsNum << ", but got "
                    << outputs.size();
      return false;
    }
    auto sgd_op = std::make_shared<ops::SGD>(base_operator->GetPrim());

    dampening_ = sgd_op->get_dampening();
    weight_decay_ = sgd_op->get_weight_decay();
    nesterov_ = sgd_op->get_nesterov();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    int ret = KernelMod::Resize(base_operator, inputs, outputs);
    if (ret != 0) {
      return ret;
    }
    auto input_shape = Convert2SizeTClipNeg(inputs[0]->GetShapeVector());
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "parameters");
    if (is_null_input_) {
      InitSizeLists();
      return KRET_OK;
    }
    for (auto &dim : input_shape) {
      size_ *= dim;
    }
    InitSizeLists();
    return KRET_OK;
  }

 protected:
  void InitSizeLists() {
    input_size_list_.clear();
    output_size_list_.clear();

    size_t input_size = size_ * sizeof(T);
    input_size_list_.push_back(input_size);  // parameter
    input_size_list_.push_back(input_size);  // gradient
    input_size_list_.push_back(sizeof(T));   // lr
    input_size_list_.push_back(input_size);  // accum
    input_size_list_.push_back(sizeof(T));   // momentum
    input_size_list_.push_back(input_size);  // stat
    output_size_list_.push_back(input_size);
  }

 private:
  size_t size_;
  float dampening_;
  float weight_decay_;
  bool nesterov_;
  bool is_null_input_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_SGD_KERNEL_H_
