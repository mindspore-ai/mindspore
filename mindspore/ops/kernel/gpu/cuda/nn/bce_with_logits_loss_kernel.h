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

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class BCEWithLogitsLossKernelMod : public NativeGpuKernelMod {
 public:
  BCEWithLogitsLossKernelMod() = default;
  ~BCEWithLogitsLossKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
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
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using BCEWithLogitsLossLaunchFunc = std::function<bool(
    BCEWithLogitsLossKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
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
  size_t weight_workspace_index_ = 0;
  size_t pos_weight_workspace_index_ = 0;
  size_t output_tmp_index_ = 0;
  size_t reduce_workspace_index_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H_
