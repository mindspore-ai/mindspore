/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_ADD_TRANSFORM_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_ADD_TRANSFORM_PASS_H_
#include <vector>
#include "src/lite_kernel.h"
#include "src/ops/primitive_c.h"
namespace mindspore::lite {
class NPUAddTransformPass {
 public:
  int Run(const InnerContext *context, std::vector<kernel::LiteKernel *> *all_kernels,
          std::vector<Tensor *> *all_tensors);

 private:
  int UpdateNH2NCTransNodePreKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                    kernel::LiteKernel *after_kernel);

  int UpdateNC2NHTransNodeAfterKernel(kernel::LiteKernel *kernel, kernel::LiteKernel *trans_kernel,
                                      kernel::LiteKernel *next_kernel);

  int InsertNode(const InnerContext *context, std::vector<kernel::LiteKernel *>::iterator it,
                 std::vector<kernel::LiteKernel *> *all_kernels, std::vector<Tensor *> *all_tensors);

 private:
  int total = 0;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_ADD_TRANSFORM_PASS_H_
