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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_PASS_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_PASS_UTILS_H_
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include "src/lite_kernel.h"
namespace mindspore::lite {
extern std::unordered_map<schema::PrimitiveType, std::set<int>> nodes2const_index;
class NPUPassUtils {
 public:
  static kernel::LiteKernel *CreateNchw2NhwcKernel(const std::vector<Tensor *> &in_tensors,
                                                   const std::vector<Tensor *> &out_tensors, const InnerContext *ctx,
                                                   const std::string &name);

  static kernel::LiteKernel *CreateNhwc2NchwKernel(const std::vector<Tensor *> &in_tensors,
                                                   const std::vector<Tensor *> &out_tensors, const InnerContext *ctx,
                                                   const std::string &name);

  static void UpdateKernel(kernel::LiteKernel *kernel, const std::vector<kernel::LiteKernel *> &in_kernels,
                           const std::vector<kernel::LiteKernel *> &out_kernels,
                           const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors);

  static void UpdateNH2NCTransNodePreKernel(kernel::LiteKernel *pre_kernel, kernel::LiteKernel *trans_kernel,
                                            kernel::LiteKernel *kernel);

  static void UpdateNC2NHTransNodePreKernel(kernel::LiteKernel *pre_kernel,
                                            const std::vector<kernel::LiteKernel *> &trans_kernels,
                                            const std::vector<kernel::LiteKernel *> &kernels);

  static void UpdateNH2NCTransNodePostKernel(kernel::LiteKernel *trans_kernel, kernel::LiteKernel *post_kernel);

  static void UpdateNC2NHTransNodePostKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                             kernel::LiteKernel *post_kernel);

  static void UpdateNC2NHPostKernelInTensors(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                             kernel::LiteKernel *post_kernel);

  static bool IsNhwc2Nchw(const kernel::LiteKernel *kernel);

  static bool IsNchw2Nhwc(const kernel::LiteKernel *kernel);
  static kernel::LiteKernel *KernelInputFromKernel(const kernel::LiteKernel *kernel, Tensor *in_tensor);
  static std::vector<Tensor *> GetNonConstInputs(kernel::LiteKernel *kernel);
  static bool Scale4dCase(const kernel::LiteKernel *kernel);
  static void AssistDataNHWC2NCHW(int *data, size_t unit_size);
  static int MaskDataNHWC2NCHW(int mask);
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_PASS_UTILS_H_
