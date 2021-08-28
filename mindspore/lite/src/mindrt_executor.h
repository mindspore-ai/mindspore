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

#ifndef MINDSPORE_LITE_SRC_MINDRT_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_MINDRT_EXECUTOR_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include "src/runtime/inner_allocator.h"
#include "src/lite_kernel.h"
#include "src/lite_mindrt.h"
#include "src/executor.h"
#include "include/lite_session.h"

namespace mindspore::lite {
class MindrtExecutor : public Executor {
 public:
  explicit MindrtExecutor(std::unordered_map<Tensor *, Tensor *> *output_map) : output_tensor_map_(output_map) {}
  virtual ~MindrtExecutor() { MindrtTerminate(op_actors_); }

  int Prepare(const std::vector<kernel::LiteKernel *> &kernels, const std::vector<Tensor *> &inputs,
              const std::vector<Tensor *> &outputs, const lite::InnerContext *ctx) override;

  int Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
          const std::vector<kernel::LiteKernel *> &kernels, const KernelCallBack &before = nullptr,
          const KernelCallBack &after = nullptr) override;

  int Resize(const std::vector<mindspore::tensor::MSTensor *> &inputs,
             const std::vector<std::vector<int>> &dims) override;

 private:
  void TransferGraphOutput();
  void FreeOutputTensor();

 protected:
  int PrepareInputData(const std::vector<kernel::LiteKernel *> &kernels, const std::vector<Tensor *> &inputs);
  int PrepareOutputData(const std::vector<kernel::LiteKernel *> &kernels, const std::vector<Tensor *> &outputs);
  std::vector<std::shared_ptr<LiteOpActor>> op_actors_;
  std::vector<OpDataPtr<Tensor>> input_data_;
  std::vector<OpDataPtr<Tensor>> output_data_;
  std::unordered_map<Tensor *, Tensor *> *output_tensor_map_;
};

}  // namespace mindspore::lite
#endif
