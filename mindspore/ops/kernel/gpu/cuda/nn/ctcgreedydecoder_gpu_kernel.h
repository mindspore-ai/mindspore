/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CTCGREEDYDECODER_CTCGREEDYDECODER_KERNEL_H_
#define MINDSPORE_CTCGREEDYDECODER_CTCGREEDYDECODER_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include "mindspore/ops/infer/ctc_greedy_decoder.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
namespace mindspore {
namespace kernel {
class CTCGreedyDecoderGpuKernelMod : public NativeGpuKernelMod {
 public:
  CTCGreedyDecoderGpuKernelMod() = default;
  ~CTCGreedyDecoderGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using CTCGreedyDecoderFunc = std::function<bool(
    CTCGreedyDecoderGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;
  static std::vector<std::pair<KernelAttr, CTCGreedyDecoderFunc>> func_list_;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  CTCGreedyDecoderFunc kernel_func_;

 private:
  std::vector<int64_t> inputs_x_shape_;
  std::vector<int64_t> sequence_shape_;
  size_t data_unit_size_;
  size_t batch_size_;
  size_t max_time_;
  int bound_;
  bool merge_repeated_{true};
  bool is_null_input_;
  int64_t element_cnt_;
  void *stream_ptr_;
  void ResetResource();
  void InitSizeLists();
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CTCGREEDYDECODER_CTCGREEDYDECODER_KERNEL_H_
