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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RMSPROP_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RMSPROP_KERNEL_H_

#include <vector>
#include <functional>
#include <map>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/rmsprop_impl.cuh"
#include "mindspore/core/ops/apply_rms_prop.h"

namespace mindspore {
namespace kernel {
constexpr size_t kCenteredRMSPropInputsNum = 9;
constexpr size_t kRMSPropInputsNum = 8;
constexpr auto kApplyRMSProp = "ApplyRMSProp";
constexpr auto kApplyCenteredRMSProp = "ApplyCenteredRMSProp";
constexpr auto kNumberZero = 0;
constexpr auto kNumberOne = 1;
constexpr auto kNumberTwo = 2;
constexpr auto kNumberThree = 3;
constexpr auto kNumberFour = 4;
constexpr auto kNumberFive = 5;
constexpr auto kNumberSix = 6;
constexpr auto kNumberSeven = 7;
constexpr auto kNumberEight = 8;
template <typename T>
class RMSPropGpuKernelMod : public NativeGpuKernelMod {
 public:
  RMSPropGpuKernelMod() : size_(1), use_center_(false), is_null_input_(false) {}
  ~RMSPropGpuKernelMod() override = default;
  int CheckShapeEqual(std::vector<int64_t> size_a, std::vector<int64_t> size_b, const char *name_a,
                      const char *name_b) {
    if (!IsSameShape(size_a, size_b)) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of '" << name_a
                    << "' must be the same as the shape of '" << name_b << "', but got the shape of '" << name_b
                    << "': " << Vector2Str(size_b) << " and the shape of '" << name_a << "': " << Vector2Str(size_a);
      return KRET_RESIZE_FAILED;
    }
    return KRET_OK;
  }
  int CalElements(std::vector<int64_t> var_shape, std::vector<int64_t> lr_shape, int ret) {
    if (batch_rank_ == 0) {
      if (lr_shape.size() != 0 && lr_shape.size() != 1) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', the shape size of 'lr' must be 0 or 1, but got the shape of 'lr': " << Vector2Str(lr_shape)
                      << " and 'batch_rank': " << batch_rank_;
      }
    } else {
      if (batch_rank_ < 0 || lr_shape.size() != static_cast<size_t>(batch_rank_)) {
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', the shape size of 'lr' must be equal to 'batch_rank', "
                         "but got the shape of 'lr': "
                      << Vector2Str(lr_shape) << " and 'batch_rank': " << batch_rank_;
        return KRET_RESIZE_FAILED;
      }
    }

    if (!lr_shape.empty()) {
      batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), int64_t(1), std::multiplies<int64_t>());
    }

    if (batch_size_ > 0) {
      input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), int64_t(1), std::multiplies<int64_t>());
      input_elements_ = input_elements_ / batch_size_;
      return ret;
    } else {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
      return KRET_RESIZE_FAILED;
    }
  }
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
    int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
    if (ret != 0) {
      return ret;
    }

    if (!use_center_) {
      CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRMSPropInputsNum, kernel_name_);
      std::vector<int64_t> var_shape = inputs[kNumberZero]->GetShapeVector();
      std::vector<int64_t> mean_square_shape = inputs[kNumberOne]->GetShapeVector();
      std::vector<int64_t> moment_shape = inputs[kNumberTwo]->GetShapeVector();
      std::vector<int64_t> lr_shape = inputs[kNumberThree]->GetShapeVector();
      std::vector<int64_t> grad_shape = inputs[kNumberFour]->GetShapeVector();

      CheckShapeEqual(var_shape, mean_square_shape, "var", "mean_square");
      CheckShapeEqual(var_shape, moment_shape, "var", "moment");
      CheckShapeEqual(var_shape, grad_shape, "var", "grad");
      return CalElements(var_shape, lr_shape, ret);
    } else {
      CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCenteredRMSPropInputsNum, kernel_name_);
      std::vector<int64_t> var_shape = inputs[kNumberZero]->GetShapeVector();
      std::vector<int64_t> mean_gradients_shape = inputs[kNumberOne]->GetShapeVector();
      std::vector<int64_t> mean_square_shape = inputs[kNumberTwo]->GetShapeVector();
      std::vector<int64_t> moment_shape = inputs[kNumberThree]->GetShapeVector();
      std::vector<int64_t> grad_shape = inputs[kNumberFour]->GetShapeVector();
      std::vector<int64_t> lr_shape = inputs[kNumberFive]->GetShapeVector();

      CheckShapeEqual(var_shape, mean_gradients_shape, "var", "mean_gradients");
      CheckShapeEqual(var_shape, mean_square_shape, "var", "mean_square");
      CheckShapeEqual(var_shape, moment_shape, "var", "moment");
      CheckShapeEqual(var_shape, grad_shape, "var", "grad");
      return CalElements(var_shape, lr_shape, ret);
    }
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream) override {
    if (is_null_input_) {
      return true;
    }
    if (!use_center_) {
      T *variable = GetDeviceAddress<T>(inputs, kNumberZero);
      T *mean_square = GetDeviceAddress<T>(inputs, kNumberOne);
      T *moment = GetDeviceAddress<T>(inputs, kNumberTwo);
      T *learning_rate = GetDeviceAddress<T>(inputs, kNumberThree);
      T *gradients = GetDeviceAddress<T>(inputs, kNumberFour);
      T *decay = GetDeviceAddress<T>(inputs, kNumberFive);
      T *momentum = GetDeviceAddress<T>(inputs, kNumberSix);
      T *epsilon = GetDeviceAddress<T>(inputs, kNumberSeven);

      RmsProp(batch_size_, input_elements_, learning_rate, decay, momentum, epsilon, variable, mean_square, moment,
              gradients, size_, reinterpret_cast<cudaStream_t>(stream));
    } else {
      T *variable = GetDeviceAddress<T>(inputs, 0);
      T *mean_gradients = GetDeviceAddress<T>(inputs, 1);
      T *mean_square = GetDeviceAddress<T>(inputs, 2);
      T *moment = GetDeviceAddress<T>(inputs, 3);
      T *gradients = GetDeviceAddress<T>(inputs, 4);
      T *learning_rate = GetDeviceAddress<T>(inputs, 5);
      T *decay = GetDeviceAddress<T>(inputs, 6);
      T *momentum = GetDeviceAddress<T>(inputs, 7);
      T *epsilon = GetDeviceAddress<T>(inputs, 8);

      RmsPropCenter(batch_size_, input_elements_, learning_rate, decay, momentum, epsilon, variable, mean_gradients,
                    mean_square, moment, gradients, size_, reinterpret_cast<cudaStream_t>(stream));
    }
    return true;
  }
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    auto node_name = base_operator->name();
    batch_rank_ = base_operator->get_batch_rank();
    if (node_name == "ApplyCenteredRMSProp") {
      use_center_ = true;
    }
    auto input_shape = inputs[0]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, node_name, "var");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    size_ = SizeOf(input_shape);
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() {
    size_t input_size = size_ * sizeof(T);
    if (!use_center_) {
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(input_size);
      output_size_list_.push_back(input_size);
    } else {
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(input_size);
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      input_size_list_.push_back(sizeof(T));
      output_size_list_.push_back(input_size);
    }
  }

 private:
  size_t size_;
  bool use_center_;
  bool is_null_input_;
  int64_t batch_size_{1};
  int64_t batch_rank_{0};
  int64_t input_elements_;
};
}  // namespace kernel
}  // namespace mindspore

#endif
