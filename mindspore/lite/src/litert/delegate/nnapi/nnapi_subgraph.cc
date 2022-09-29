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

#include "src/litert/delegate/nnapi/nnapi_subgraph.h"
#include <algorithm>
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
NNAPISubGraph::~NNAPISubGraph() {
  if (nnapi_execution_ != nullptr) {
    nnapi_->ANeuralNetworksExecution_free(nnapi_execution_);
  }
  nnapi_->ANeuralNetworksCompilation_free(nnapi_compilation_);
  nnapi_->ANeuralNetworksModel_free(nnapi_model_);

  for (auto op : ops_) {
    if (op != nullptr) {
      delete op;
    }
  }
}

int NNAPISubGraph::Init() {
  for (auto op : ops_) {
    std::for_each(op->inputs().begin(), op->inputs().end(), [&](const MSTensor &tensor) {
      if (std::find(all_tensors_.begin(), all_tensors_.end(), tensor) == all_tensors_.end()) {
        all_tensors_.push_back(tensor);
      }
    });
    std::for_each(op->outputs().begin(), op->outputs().end(), [&](const MSTensor &tensor) {
      if (std::find(all_tensors_.begin(), all_tensors_.end(), tensor) == all_tensors_.end()) {
        all_tensors_.push_back(tensor);
      }
    });
  }

  for (auto input : inputs_) {
    auto itr = std::find(all_tensors_.begin(), all_tensors_.end(), input);
    MS_CHECK_TRUE_RET(itr != all_tensors_.end(), RET_ERROR);
    input_indices_.push_back(itr - all_tensors_.begin());
  }
  for (auto input : outputs_) {
    auto itr = std::find(all_tensors_.begin(), all_tensors_.end(), input);
    MS_CHECK_TRUE_RET(itr != all_tensors_.end(), RET_ERROR);
    output_indices_.push_back(itr - all_tensors_.begin());
  }

  quant_type_ = ops_.front()->get_quant_type();
  return RET_OK;
}

int NNAPISubGraph::CreateNNAPIModel() {
  // Convert data type of quantized model from int8 to uint8 to adapt the dsp and apu.
  if (quant_type_ == schema::QuantType_QUANT_ALL) {
    for (auto op : ops_) {
      auto ret = op->ConvertInOutQuantSymmToASymm();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Convert quant params failed for op: " << op->name();
        return RET_ERROR;
      }
    }
  }

  if (nnapi_->ANeuralNetworksModel_create(&nnapi_model_) != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Create NNAPI Model failed.";
    return RET_ERROR;
  }

  for (size_t idx = 0; idx < all_tensors_.size(); idx++) {
    auto tensor = all_tensors_.at(idx);
    if (AddNNAPIOperand(nnapi_model_, tensor, static_cast<int>(idx)) != RET_OK) {
      MS_LOG(ERROR) << "Add NNAPI operand failed.";
      return RET_ERROR;
    }
  }
  for (auto op : ops_) {
    if (op->AddOpToNNAPIModel(nnapi_model_, &all_tensors_) != RET_OK) {
      MS_LOG(ERROR) << "Add NNAPI Op to model failed: " << op->name();
      return RET_ERROR;
    }
  }
  if (nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs(
        nnapi_model_, static_cast<uint32_t>(input_indices_.size()), input_indices_.data(),
        static_cast<uint32_t>(output_indices_.size()), output_indices_.data()) != RET_OK) {
    MS_LOG(ERROR) << "Identify inputs and outputs failed for NNAPI model.";
    return RET_ERROR;
  }

  if (nnapi_->android_sdk_version > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16(nnapi_model_, relax_fp32_to_fp16_);
  }
  nnapi_->ANeuralNetworksModel_finish(nnapi_model_);
  return RET_OK;
}

int NNAPISubGraph::CompileNNAPIModel() {
  auto ret = 0;
  if (!devices_.empty()) {
    // available at API 28.
    ret = nnapi_->ANeuralNetworksCompilation_createForDevices(nnapi_model_, devices_.data(), devices_.size(),
                                                              &nnapi_compilation_);
  } else {
    ret = nnapi_->ANeuralNetworksCompilation_create(nnapi_model_, &nnapi_compilation_);
  }
  if (ret != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Create compilateion failed.";
    return RET_ERROR;
  }
  nnapi_->ANeuralNetworksCompilation_finish(nnapi_compilation_);
  return RET_OK;
}

int NNAPISubGraph::Prepare() {
  MS_ASSERT(nnapi_compilation_ != nullptr);
  nnapi_->ANeuralNetworksExecution_create(nnapi_compilation_, &nnapi_execution_);
  MS_CHECK_TRUE_RET(nnapi_execution_ != nullptr, RET_ERROR);
  return RET_OK;
}

int NNAPISubGraph::PreProcess() {
  if (nnapi_execution_ == nullptr) {
    nnapi_->ANeuralNetworksExecution_create(nnapi_compilation_, &nnapi_execution_);
    MS_CHECK_TRUE_RET(nnapi_execution_ != nullptr, RET_ERROR);
  }
  // set input data.
  for (int idx = 0; idx < static_cast<int>(input_indices_.size()); idx++) {
    MS_CHECK_TRUE_RET(idx < static_cast<int>(inputs_.size()), RET_ERROR);
    auto tensor = inputs_.at(idx);
    if (nnapi_->ANeuralNetworksExecution_setInput(nnapi_execution_, idx, nullptr, tensor.MutableData(),
                                                  tensor.DataSize()) != ANEURALNETWORKS_NO_ERROR) {
      MS_LOG(ERROR) << "Set input failed.";
      return RET_ERROR;
    }
  }
  for (int idx = 0; idx < static_cast<int>(output_indices_.size()); idx++) {
    MS_CHECK_TRUE_RET(idx < static_cast<int>(outputs_.size()), RET_ERROR);
    auto tensor = outputs_.at(idx);
    if (nnapi_->ANeuralNetworksExecution_setOutput(nnapi_execution_, idx, nullptr, tensor.MutableData(),
                                                   tensor.DataSize()) != ANEURALNETWORKS_NO_ERROR) {
      MS_LOG(ERROR) << "Set output failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NNAPISubGraph::Execute() {
  if (PreProcess() != RET_OK) {
    MS_LOG(ERROR) << "PreProcess failed.";
    return RET_ERROR;
  }
  auto ret = 0;
  if (nnapi_->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_3) {
    ret = nnapi_->ANeuralNetworksExecution_compute(nnapi_execution_);
  } else {
    ANeuralNetworksEvent *event;
    ret = nnapi_->ANeuralNetworksExecution_startCompute(nnapi_execution_, &event);
    if (ret != ANEURALNETWORKS_NO_ERROR) {
      MS_LOG(ERROR) << "Start to compute nnapi execution failed.";
      return RET_ERROR;
    }
    ret = nnapi_->ANeuralNetworksEvent_wait(event);
    nnapi_->ANeuralNetworksEvent_free(event);
  }
  if (ret != ANEURALNETWORKS_NO_ERROR) {
    MS_LOG(ERROR) << "Execute NNAPI model failed.";
    return RET_ERROR;
  }
  if (nnapi_->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_5) {
    nnapi_->ANeuralNetworksExecution_setReusable(nnapi_execution_, true);
  } else {
    nnapi_->ANeuralNetworksExecution_free(nnapi_execution_);
    nnapi_execution_ = nullptr;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
