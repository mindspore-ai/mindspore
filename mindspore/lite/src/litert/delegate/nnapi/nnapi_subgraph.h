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

#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_NNAPI_SUBGRAPH_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_NNAPI_SUBGRAPH_H_

#include <vector>
#include <utility>
#include <unordered_map>
#include "include/api/kernel.h"
#include "src/common/log_adapter.h"
#include "src/litert/delegate/nnapi/op/nnapi_op.h"
#include "src/litert/delegate/nnapi/nnapi_implementation.h"

namespace mindspore {
namespace lite {
class NNAPISubGraph : public kernel::Kernel {
 public:
  NNAPISubGraph(std::vector<NNAPIOp *> ops, const std::vector<mindspore::MSTensor> &inputs,
                const std::vector<mindspore::MSTensor> &outputs, const std::vector<ANeuralNetworksDevice *> devices,
                bool relax_fp32_to_fp16)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr),
        ops_(std::move(ops)),
        relax_fp32_to_fp16_(relax_fp32_to_fp16),
        devices_(std::move(devices)) {}

  ~NNAPISubGraph() override;

  int Init();

  int CreateNNAPIModel();

  int CompileNNAPIModel();

  int Prepare() override;

  int ReSize() override {
    MS_LOG(ERROR) << "NNAPI does not support the resize function temporarily.";
    return RET_ERROR;
  }

  int Execute() override;

 private:
  int PreProcess();

  std::vector<NNAPIOp *> ops_;
  std::vector<MSTensor> all_tensors_;
  std::vector<uint32_t> input_indices_;
  std::vector<uint32_t> output_indices_;

  bool relax_fp32_to_fp16_ = true;
  std::vector<ANeuralNetworksDevice *> devices_;
  ANeuralNetworksModel *nnapi_model_ = nullptr;
  ANeuralNetworksCompilation *nnapi_compilation_ = nullptr;
  ANeuralNetworksExecution *nnapi_execution_ = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_NNAPI_NNAPI_SUBGRAPH_H_
