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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_EXECUTOR_H_
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "include/HiAiModelManagerService.h"
#include "src/delegate/npu/npu_manager.h"
#include "src/delegate/npu/op/npu_op.h"

namespace mindspore {
class NPUExecutor {
 public:
  explicit NPUExecutor(const std::string &model_name, NPUManager *npu_manager = nullptr)
      : model_name_(model_name), npu_manager_(npu_manager) {}
  ~NPUExecutor();
  int Prepare();

  int Run(const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors,
          const std::vector<NPUOp *> &in_ops);

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
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_EXECUTOR_H_
