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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_SUBGRAPH_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_SUBGRAPH_H_

#include <memory>
#include <vector>
#include <string>
#include "include/api/kernel.h"
#include "src/delegate/npu/npu_executor.h"

namespace mindspore {
class NPUSubGraph : public kernel::Kernel {
 public:
  NPUSubGraph(const std::vector<NPUOp *> &npu_ops, NPUManager *npu_manager)
      : npu_ops_(npu_ops), npu_manager_(npu_manager) {}

  ~NPUSubGraph() override;

  int Init();

  int Prepare() override;

  int Execute() override;

  int ReSize() override {
    MS_LOG(ERROR) << "NPU does not support the resize function temporarily.";
    return lite::RET_ERROR;
  }

  void set_input(mindspore::MSTensor in_tensor, int index) override;

  void set_output(mindspore::MSTensor out_tensor, int index) override;

  int GetGraphInOutOps();

  std::vector<NPUOp *> FindPreOps(NPUOp *cur_op);

 private:
  std::shared_ptr<domi::ModelBufferData> BuildIRModel();

  int BuildNPUInputOp();

  int BuildNPUOutputOp();

  int GetNPUOperators(const std::vector<NPUOp *> &ops);

  bool IsSubGraphInputTensor(mindspore::MSTensor input);

  std::string GetOMModelName();

  bool is_compiled_ = false;

  std::vector<ge::Operator> subgraph_input_ops_;

  std::vector<ge::Operator> subgraph_output_ops_;

  std::vector<mindspore::MSTensor> out_tensor_sorted_;

  std::vector<ge::Operator *> op_buffer_;

  std::vector<NPUOp *> npu_ops_{};
  // entry nodes in nodes
  std::vector<NPUOp *> in_ops_{};
  // exit nodes in nodes
  std::vector<NPUOp *> out_ops_{};

  NPUExecutor *executor_ = nullptr;

  NPUManager *npu_manager_ = nullptr;
};

}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_SUBGRAPH_H_
