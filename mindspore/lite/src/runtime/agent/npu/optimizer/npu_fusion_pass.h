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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_FUSION_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_FUSION_PASS_H_
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/agent/npu/optimizer/npu_base_pass.h"
namespace mindspore::lite {
class NPUFusionPass : public NPUBasePass {
 public:
  explicit NPUFusionPass(std::vector<kernel::LiteKernel *> *dst_kernels) {
    kernels = dst_kernels;
    name_ = "NPUFusionPass";
  }

  ~NPUFusionPass() override = default;

  int Run() override;

 protected:
  void UpdatePreKernels(kernel::LiteKernel *kernel);
  void UpdatePostKernels(kernel::LiteKernel *kernel);
  void RemoveAndFreeKernel(kernel::LiteKernel *cur_kernel);
  void UpdateKernel(kernel::LiteKernel *kernel);
  int CommonFusion(kernel::LiteKernel *kernel);
  int ConcatFusion(kernel::LiteKernel *kernel);
  int FormatFusion(kernel::LiteKernel *kernel);
  int SplitFusion(kernel::LiteKernel *kernel);
  int PadFusion(kernel::LiteKernel *kernel);
  int StridedSliceFusion(kernel::LiteKernel *kernel);

 private:
  std::vector<kernel::LiteKernel *> *kernels;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_FUSION_PASS_H_
