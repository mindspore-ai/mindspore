/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class BCEWithLogitsLossKernelMod : public NativeGpuKernelMod {
 public:
  BCEWithLogitsLossKernelMod() = default;
  ~BCEWithLogitsLossKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  void InitWorkSpaceSizeLists() {
    workspace_size_list_.push_back(input_shape_.size() * sizeof(size_t));
    workspace_size_list_.push_back(weight_shape_.size() * sizeof(size_t));
    workspace_size_list_.push_back(pos_weight_shape_.size() * sizeof(size_t));
    // extra space for holding extra array shape of input, for broadcasted
    // weight and pos_weight
    workspace_size_list_.push_back(input_size_ * type_id_size_);
  }

  bool NeedBroadcast(ShapeVector *shape, const ShapeVector &result_shape) {
    // result_shape is larger that shape
    // and shape is able to broadcasted to result_shape
    if (shape->size() < result_shape.size()) {
      size_t fill_size = result_shape.size() - shape->size();
      (void)shape->insert(shape->begin(), fill_size, 1);
      return true;
    }
    for (size_t i = 0; i < result_shape.size(); i++) {
      if (shape->at(i) != result_shape[i]) {
        return true;
      }
    }
    return false;
  }

  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using BCEWithLogitsLossLaunchFunc =
    std::function<bool(BCEWithLogitsLossKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  std::string kernel_name_{};
  BCEWithLogitsLossLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, BCEWithLogitsLossLaunchFunc>> func_list_;
  size_t input_size_;
  size_t weight_size_;
  size_t pos_weight_size_;
  bool weight_need_broadcast_;
  bool pos_weight_need_broadcast_;
  ShapeVector input_shape_;
  ShapeVector weight_shape_;
  ShapeVector pos_weight_shape_;
  size_t type_id_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_
