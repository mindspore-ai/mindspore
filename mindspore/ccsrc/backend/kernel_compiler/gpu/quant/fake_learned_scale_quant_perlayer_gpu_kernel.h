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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PERLAYER_GPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PERLAYER_GPUKERNEL_H_

#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class FakeLearnedScaleQuantPerLayerGpuKernel : public GpuKernel {
 public:
  FakeLearnedScaleQuantPerLayerGpuKernel();
  ~FakeLearnedScaleQuantPerLayerGpuKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel) override;

 protected:
  void InitSizeLists() override;

 private:
  size_t input_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  int quant_num_;
  int global_step_;
  int quant_delay_;
  bool training_;
  bool neg_trunc_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PERLAYER_GPUKERNEL_H_
