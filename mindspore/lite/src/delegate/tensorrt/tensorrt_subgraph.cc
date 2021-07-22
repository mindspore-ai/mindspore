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

#include "src/delegate/tensorrt/tensorrt_subgraph.h"
#include <string>
#include <vector>
#include <set>
#include "src/delegate/delegate_utils.h"

namespace mindspore::lite {
TensorRTSubGraph::~TensorRTSubGraph() {
  if (network_ != nullptr) {
    network_->destroy();
    network_ = nullptr;
  }
  if (config_ != nullptr) {
    config_->destroy();
    config_ = nullptr;
  }
  if (context_ != nullptr) {
    context_->destroy();
    context_ = nullptr;
  }
  if (engine_ != nullptr) {
    engine_->destroy();
    engine_ = nullptr;
  }
  if (tensor_bindings_ != nullptr) {
    delete tensor_bindings_;
    tensor_bindings_ = nullptr;
  }
  for (auto op : all_ops_) {
    delete op;
  }
}

int TensorRTSubGraph::Init() {
  auto ret = GetGraphInOutOps(inputs_, outputs_, &in_ops_, &out_ops_, all_ops_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get NPU subgraph input and output ops failed.";
    return RET_ERROR;
  }
  runtime_ = TensorRTRuntime::GetInstance();
  ret = runtime_->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorRTRuntime init failed.";
    return RET_ERROR;
  }
  this->network_ = runtime_->GetBuilder()->createNetworkV2(
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  if (this->network_ == nullptr) {
    MS_LOG(ERROR) << "New network failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorRTSubGraph::BuildEngine() {
  this->config_ = runtime_->GetBuilder()->createBuilderConfig();
  if (this->config_ == nullptr) {
    MS_LOG(ERROR) << "create builder config failed.";
    return RET_ERROR;
  }
  engine_ = runtime_->GetBuilder()->buildEngineWithConfig(*this->network_, *this->config_);
  if (engine_ == nullptr) {
    MS_LOG(ERROR) << "Create engine failed in TensorRT network";
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorRTSubGraph::BuildTensorRTGraph() {
  MS_ASSERT(!all_ops_.empty());
  // Connect NetWork.
  int ret;
  for (auto cur_op : all_ops_) {
    for (auto in_tensor : cur_op->inputs()) {
      // Data From CPU
      if (IsSubGraphInputTensor(this->inputs(), in_tensor)) {
        auto cuda_dtype = ConvertDataType(in_tensor.DataType());
        if (static_cast<int>(cuda_dtype) == -1) {
          MS_LOG(ERROR) << "Unsupported input data type " << static_cast<int>(in_tensor.DataType());
          return RET_ERROR;
        }
        auto trt_tensor =
          this->network_->addInput(in_tensor.Name().c_str(), cuda_dtype, ConvertCudaDims(in_tensor.Shape()));
        cur_op->AddInnerInTensors(trt_tensor);
        continue;
      }

      auto trt_tensor = FindTensorRTInputs(cur_op, in_tensor);
      // weight tensor
      if (trt_tensor == nullptr) {
        if (trt_specific_weight_nodes_.find(cur_op->type()) == trt_specific_weight_nodes_.end()) {
          if (in_tensor == nullptr) {
            MS_LOG(ERROR) << "Weight Tensor is nullptr.";
            return RET_ERROR;
          }
          trt_tensor = lite::ConvertConstantTensor(this->network_, in_tensor);
          cur_op->AddInnerInTensors(trt_tensor);
        }
      } else {
        cur_op->AddInnerInTensors(trt_tensor);
      }
    }

    ret = cur_op->AddInnerOp(this->network_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Add op failed in TensorRT network";
      return RET_ERROR;
    }
  }

  // Mark NetWork Output Tensor.
  for (auto out_tensor : outputs_) {
    for (auto out_op : this->out_ops_) {
      for (size_t index = 0; index < out_op->outputs().size(); index++) {
        if (out_op->outputs()[index] == out_tensor) {
          out_op->GetInnerOutTensor()[index]->setName(out_tensor.Name().c_str());
          this->network_->markOutput(*out_op->GetInnerOutTensor()[index]);
        }
      }
    }
  }

  ret = BuildEngine();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create engine failed in TensorRT network";
    return ret;
  }
  return RET_OK;
}

int TensorRTSubGraph::Prepare() {
  if (runtime_->GetBatchSize() <= 0) {
    MS_LOG(ERROR) << "TensorRTSubGraph has invalid batch size.";
    return RET_ERROR;
  }
  if (this->engine_ == nullptr) {
    MS_LOG(ERROR) << "engine_ is null in this builder_";
    return RET_ERROR;
  }
  this->context_ = this->engine_->createExecutionContext();
  if (this->context_ == nullptr) {
    MS_LOG(ERROR) << "TensorRTSubGraph create context failed.";
    return RET_ERROR;
  }
  int binding_num = this->engine_->getNbBindings();
  tensor_bindings_ = new (std::nothrow) void *[binding_num];
  if (tensor_bindings_ == nullptr) {
    MS_LOG(ERROR) << "malloc tensor binding array failed.";
    return RET_ERROR;
  }

  for (auto tensor : inputs_) {
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor, tensor.DataSize());
    int index = this->engine_->getBindingIndex(tensor.Name().c_str());
    tensor_bindings_[index] = device_ptr;
    trt_in_tensor_name_.push_back(tensor.Name());
  }

  for (auto tensor : outputs_) {
    tensor.MutableData();
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor, tensor.DataSize());
    int index = this->engine_->getBindingIndex(tensor.Name().c_str());
    tensor_bindings_[index] = device_ptr;
    trt_out_tensor_name_.push_back(tensor.Name());
  }
  return RET_OK;
}

int TensorRTSubGraph::Execute() {
  for (size_t i = 0; i < inputs_.size(); i++) {
    runtime_->GetAllocator()->SyncMemInHostAndDevice(inputs_[i], trt_in_tensor_name_[i], true);
  }
  auto ret = this->context_->executeV2(tensor_bindings_);
  if (!ret) {
    MS_LOG(ERROR) << "TensorRT execute failed.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    if (outputs_[i].MutableData() == nullptr) {
      MS_LOG(ERROR) << "Malloc output tensor data failed.";
    }
    runtime_->GetAllocator()->SyncMemInHostAndDevice(outputs_[i], trt_out_tensor_name_[i], false);
  }
  return RET_OK;
}

nvinfer1::ITensor *TensorRTSubGraph::FindTensorRTInputs(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor) {
  for (auto input_op : cur_op->in_ops()) {
    for (size_t i = 0; i < input_op->outputs().size(); i++) {
      auto out_tensor = input_op->outputs().at(i);
      if (in_tensor == out_tensor) {
        return input_op->GetInnerOutTensor().at(i);
      }
    }
  }
  return nullptr;
}
}  // namespace mindspore::lite
