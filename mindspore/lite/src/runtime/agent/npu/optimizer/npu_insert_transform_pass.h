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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_INSERT_TRANSFORM_PASS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_INSERT_TRANSFORM_PASS_H_
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/agent/npu/optimizer/npu_base_pass.h"

namespace mindspore::lite {
class NPUInsertTransformPass : public NPUBasePass {
 public:
  explicit NPUInsertTransformPass(const InnerContext *context, std::vector<kernel::LiteKernel *> *all_kernels,
                                  std::vector<Tensor *> *all_tensors) {
    context_ = context;
    all_kernels_ = all_kernels;
    all_tensors_ = all_tensors;
    name_ = "NPUInsertTransformPass";
  }

  int Run() override;

 private:
  int GetInsertState(kernel::LiteKernel *kernel);
  int InsertPreNodes(kernel::LiteKernel *kernel, std::vector<kernel::LiteKernel *> *trans_kernels);

  int InsertPostNodes(kernel::LiteKernel *kernel, std::vector<kernel::LiteKernel *> *trans_kernels);

  int InsertNode(kernel::LiteKernel *kernel, kernel::LiteKernel *post_kernel, size_t post_input_index,
                 std::vector<kernel::LiteKernel *> *trans_kernels);
  int InsertForInputTensor(kernel::LiteKernel *kernel, size_t in_tensor_index, kernel::LiteKernel *pre_kernel,
                           std::vector<kernel::LiteKernel *> *trans_kernels);

  int InsertForOutputTensor(kernel::LiteKernel *kernel, kernel::LiteKernel *post_kernel, size_t post_in_tensor_index,
                            std::vector<kernel::LiteKernel *> *trans_kernels);

 private:
  int total = 0;
  const InnerContext *context_;
  std::vector<kernel::LiteKernel *> *all_kernels_;
  std::vector<Tensor *> *all_tensors_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_INSERT_TRANSFORM_PASS_H_
