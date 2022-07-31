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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_GRAPH_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_GRAPH_H_

#include <vector>
#include <queue>
#include <map>
#include <string>
#include <utility>
#include "proto/Model.pb.h"
#include "proto/NeuralNetwork.pb.h"
#include "include/api/kernel.h"
#include "src/litert/delegate/coreml/op/coreml_op.h"
#include "src/litert/delegate/coreml/coreml_executor_wrapper.h"

namespace mindspore::lite {
constexpr int kCoreMLVersion4 = 4;
class CoreMLGraph : public kernel::Kernel {
 public:
  CoreMLGraph(std::vector<CoreMLOp *> coreml_ops, const std::vector<mindspore::MSTensor> &inputs,
              const std::vector<mindspore::MSTensor> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr), coreml_ops_(std::move(coreml_ops)) {}

  ~CoreMLGraph() override;

  int Init();

  int Prepare() override { return RET_OK; }

  int Execute() override;

  int ReSize() override {
    MS_LOG(ERROR) << "CoreML does not support the resize function temporarily.";
    return RET_ERROR;
  }

  void set_input(mindspore::MSTensor in_tensor, int index) override;

  void set_output(mindspore::MSTensor out_tensor, int index) override;

  std::vector<CoreMLOp *> *GetOps() { return &coreml_ops_; }

  std::vector<mindspore::MSTensor *> *GetInsertTensors() { return &insert_tensors_; }

 protected:
  CoreML::Specification::Model *BuildMLModel();

  int SetMLModelInOut(CoreML::Specification::Model *model);

  std::string SaveMLModel();

  std::vector<CoreMLOp *> coreml_ops_{};
  std::vector<kernel::Kernel *> all_kernels_{};
  CoreML::Specification::Model *ml_model_ = nullptr;
  CoreMLExecutorWrapper *executor_wrapper_ = nullptr;
  std::vector<mindspore::MSTensor *> insert_tensors_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_GRAPH_H_
