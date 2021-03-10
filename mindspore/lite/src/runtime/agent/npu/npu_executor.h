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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_EXECUTOR_H_
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include "src/executor.h"
#include "include/errorcode.h"
#include "include/HiAiModelManagerService.h"
#ifdef SUPPORT_NPU
#include "src/runtime/agent/npu/npu_manager.h"
#endif

namespace mindspore::lite {
class NPUExecutor : public Executor {
 public:
  explicit NPUExecutor(const std::string &model_name, NPUManager *npu_manager = nullptr)
      : model_name_(model_name), npu_manager_(npu_manager) {}
  ~NPUExecutor() override;
  int Prepare(const std::vector<kernel::LiteKernel *> &kernels) override;

  int Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
          const std::vector<kernel::LiteKernel *> &out_kernels, const std::vector<kernel::LiteKernel *> &kernels,
          Allocator *allocator = nullptr, const KernelCallBack &before = nullptr,
          const KernelCallBack &after = nullptr);

 private:
  int GetIOTensorVec();

  int UpdateInputTensorVec(const std::vector<hiai::TensorDimension> &input_dimension);

  int UpdateOutputTensorVec(const std::vector<hiai::TensorDimension> &output_dimension);

 private:
  std::string model_name_;
  NPUManager *npu_manager_ = nullptr;
  std::shared_ptr<hiai::AiModelMngerClient> client_ = nullptr;
  std::vector<std::shared_ptr<hiai::AiTensor>> npu_input_tensors_;
  std::vector<std::shared_ptr<hiai::AiTensor>> npu_output_tensors_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_EXECUTOR_H_
