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
#include <cuda_runtime_api.h>
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
  if (trt_context_ != nullptr) {
    trt_context_->destroy();
    trt_context_ = nullptr;
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
  this->network_ = runtime_->GetBuilder()->createNetworkV2(
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  if (this->network_ == nullptr) {
    MS_LOG(ERROR) << "New network failed.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (inputs_[i].Shape().size() != DIMENSION_4D) {
      MS_LOG(WARNING) << "hw dims resize is unsupported.";
      input_hw_index_ = -1;
    }
  }
  if (SetDeviceConfig() != RET_OK) {
    MS_LOG(WARNING) << "set tensorrt config failed.";
  }
  profile_ = runtime_->GetBuilder()->createOptimizationProfile();
  return RET_OK;
}

int TensorRTSubGraph::BuildEngine() {
  // print all network ops
  if (this->config_->addOptimizationProfile(profile_) == -1) {
    MS_LOG(ERROR) << "addOptimizationProfile failed.";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "build engine for tensorrt network: " << this->network_->getName();
  for (int i = 0; i < this->network_->getNbLayers(); i++) {
    MS_LOG(DEBUG) << "tensorrt op: " << this->network_->getLayer(i)->getName();
  }
  MS_LOG(DEBUG) << "end of tensorrt network: " << this->network_->getName();

  this->engine_ = runtime_->GetBuilder()->buildEngineWithConfig(*this->network_, *this->config_);
  if (this->engine_ == nullptr) {
    MS_LOG(ERROR) << "Create engine failed in TensorRT network";
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorRTSubGraph::SetDeviceConfig() {
  this->config_ = runtime_->GetBuilder()->createBuilderConfig();
  if (this->config_ == nullptr) {
    MS_LOG(ERROR) << "create builder config failed.";
    return RET_ERROR;
  }
  // set fp16
  if (device_info_->GetEnableFP16() && runtime_->GetBuilder()->platformHasFastFp16()) {
    MS_LOG(INFO) << "set fp16 flag successfully for tensorrt.";
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    input_hw_index_ = -1;
  }

  // config setMaxWorkspaceSize to 32 MB for max limit
  config_->setMaxWorkspaceSize(32 * (1 << 20));
  return RET_OK;
}

bool TensorRTSubGraph::SupportFP16() {
  int deviceCnt = 0;

  cudaError ret = cudaGetDeviceCount(&deviceCnt);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaGetDeviceCount failed.";
    return false;
  }
  std::vector<std::string> supportFP16_versions{"5.3", "6.0", "6.2", "7.0", "7.2", "7.5", "8.0", "8.6"};
  cudaDeviceProp prop;
  std::string version;
  for (int dev = 0; dev < deviceCnt; dev++) {
    ret = cudaGetDeviceProperties(&prop, dev);
    if (ret != cudaSuccess) {
      MS_LOG(ERROR) << "cuDeviceGetAttribute failed.";
      return false;
    }
    version = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    if (std::find(supportFP16_versions.begin(), supportFP16_versions.end(), version) != supportFP16_versions.end()) {
      MS_LOG(INFO) << "cuda device version is: " << version << ", support FP16, set enable FP16 tag successful";
      return true;
    }
  }
  MS_LOG(WARNING) << "cuda device version is: " << version << ", don't support FP16, set enable FP16 tag failed";
  return false;
}

nvinfer1::ITensor *TensorRTSubGraph::SetTensorRTNetworkInput(const mindspore::MSTensor &in_tensor) {
  for (int i = 0; i < this->network_->getNbInputs(); i++) {
    if (in_tensor.Name().compare(this->network_->getInput(i)->getName()) == 0) {
      MS_LOG(INFO) << "input tensor is already added in network: " << in_tensor.Name();
      return this->network_->getInput(i);
    }
  }

  auto cuda_dtype = ConvertDataType(in_tensor.DataType());
  if (static_cast<int>(cuda_dtype) == -1) {
    MS_LOG(ERROR) << "Unsupported input data type " << static_cast<int>(in_tensor.DataType());
    return nullptr;
  }
  nvinfer1::Dims input_dims = ConvertCudaDims(in_tensor.Shape());
  if (runtime_->GetBatchSize() == 0) {
    runtime_->SetBatchSize(input_dims.d[0]);
    MS_LOG(INFO) << "batch size init as " << runtime_->GetBatchSize();
    input_dims.d[0] = -1;  // dynamic batch size with wildcard N, default batchsize is first dims
    input_batchsize_index_ = 0;
  } else {
    for (int n = 0; n < input_dims.nbDims; n++) {
      if (input_dims.d[n] == runtime_->GetBatchSize()) {
        // first dims equals to batchsize
        input_dims.d[n] = -1;
        input_batchsize_index_ = n;
        break;
      }
    }
  }
  // only support NHWC HW dim resize
  if (input_hw_index_ != -1) {
    MS_LOG(INFO) << "input tensor format is (NHWC:1, NCHW:0): " << in_tensor.format();
    input_hw_index_ = in_tensor.format() == Format::NHWC ? 1 : 2;  // NCHW is 2
    input_dims.d[input_hw_index_] = -1;
    input_dims.d[input_hw_index_ + 1] = -1;
  }
  // We do not need to check the return of setDimension and addOptimizationProfile here as all dims are explicitly set
  nvinfer1::Dims input_dims_min = ConvertCudaDims(in_tensor.Shape());
  input_dims_min.d[input_batchsize_index_] = 1;
  if (input_hw_index_ != -1) {
    input_dims_min.d[input_hw_index_] = 1;
    input_dims_min.d[input_hw_index_ + 1] = 1;
  }
  if (!profile_->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMIN, input_dims_min)) {
    MS_LOG(ERROR) << "setDimensions of kMIN failed.";
    return nullptr;
  }
  nvinfer1::Dims input_dims_opt = ConvertCudaDims(in_tensor.Shape());
  if (!profile_->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kOPT, input_dims_opt)) {
    MS_LOG(ERROR) << "setDimensions of kOPT failed.";
    return nullptr;
  }
  nvinfer1::Dims input_dims_max = ConvertCudaDims(in_tensor.Shape());
  // input_dims_max should be the same with input network dims
  if (!profile_->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMAX, input_dims_max)) {
    MS_LOG(ERROR) << "setDimensions of kMAX failed.";
    return nullptr;
  }

  return this->network_->addInput(in_tensor.Name().c_str(), cuda_dtype, input_dims);
}

int TensorRTSubGraph::BuildTensorRTGraph() {
  MS_ASSERT(!all_ops_.empty());
  // Connect NetWork.
  int ret;
  for (auto cur_op : all_ops_) {
    for (auto in_tensor : cur_op->inputs()) {
      // Data From CPU
      if (IsSubGraphInputTensor(this->inputs(), in_tensor)) {
        nvinfer1::ITensor *trt_tensor = SetTensorRTNetworkInput(in_tensor);
        if (trt_tensor == nullptr) {
          MS_LOG(ERROR) << "SetTensorRTNetworkInput failed for " << in_tensor.Name();
          return RET_ERROR;
        }
        cur_op->AddInnerInTensors(ITensorHelper{trt_tensor, in_tensor.format()});
        continue;
      }

      ITensorHelper trt_tensor = FindTensorRTInputs(cur_op, in_tensor);
      if (trt_tensor.trt_tensor_ == nullptr) {
        // weight tensor
        if (trt_specific_weight_nodes_.find(cur_op->type()) == trt_specific_weight_nodes_.end()) {
          if (in_tensor.Data() == nullptr) {
            MS_LOG(ERROR) << "Weight Tensor data is nullptr.";
            return RET_ERROR;
          }
          trt_tensor.trt_tensor_ = lite::ConvertConstantTensor(this->network_, in_tensor);
          trt_tensor.format_ = Format::NHWC;
          MS_LOG(INFO) << "auto convert constant tensor for: " << in_tensor.Name();
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

  ret = MarkOutputs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MarkOutputs failed in TensorRT network";
    return ret;
  }
  std::string network_name =
    "network_" + std::string(network_->getInput(0)->getName()) + "_" + std::string(network_->getOutput(0)->getName());
  network_->setName(network_name.c_str());
  this->name_ = network_name;

  ret = BuildEngine();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create engine failed in TensorRT network";
    return ret;
  }
  return RET_OK;
}

int TensorRTSubGraph::MarkOutputs() {
  // Mark NetWork Output Tensor.
  for (auto out_tensor : outputs_) {
    for (auto out_op : this->out_ops_) {
      for (size_t index = 0; index < out_op->outputs().size(); index++) {
        if (out_op->outputs()[index] == out_tensor) {
          nvinfer1::ITensor *out_trt_tensor = out_op->GetInnerOutTensor()[index].trt_tensor_;
          if (out_op->GetInnerOutTensor()[index].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
              out_op->GetInnerOutTensor()[index].format_ == Format::NCHW &&
              !SameDims(out_op->GetInnerOutTensor()[index].trt_tensor_->getDimensions(), out_tensor.Shape())) {
            // transpose subgraph output from nchw to nhwc
            nvinfer1::IShuffleLayer *transpose_layer_out =
              NCHW2NHWC(network_, *out_op->GetInnerOutTensor()[index].trt_tensor_);
            if (transpose_layer_out == nullptr) {
              MS_LOG(ERROR) << "op action convert failed";
              return RET_ERROR;
            }
            transpose_layer_out->setName((out_tensor.Name() + "_transpose2NHWC").c_str());
            out_trt_tensor = transpose_layer_out->getOutput(0);
          }

          out_trt_tensor->setName(out_tensor.Name().c_str());
          MS_LOG(INFO) << "markOutput for: " << out_tensor.Name();
          this->network_->markOutput(*out_trt_tensor);
          for (int n = 0; n < out_trt_tensor->getDimensions().nbDims; n++) {
            if (out_trt_tensor->getDimensions().d[n] == -1) {
              output_batchsize_index_ = n;
              break;
            }
          }
        }
      }
    }
  }
  return RET_OK;
}

int TensorRTSubGraph::Prepare() {
  lite::SetCudaDevice(device_info_);
  if (this->engine_ == nullptr) {
    MS_LOG(ERROR) << "engine_ is null in this builder_";
    return RET_ERROR;
  }
  this->trt_context_ = this->engine_->createExecutionContext();
  if (this->trt_context_ == nullptr) {
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
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "malloc for inputs tensor device memory failed.";
      return RET_ERROR;
    }
    int index = this->engine_->getBindingIndex(tensor.Name().c_str());
    tensor_bindings_[index] = device_ptr;
    trt_in_tensor_name_.push_back(tensor.Name());
    nvinfer1::Dims input_dims = ConvertCudaDims(tensor.Shape());
    for (int od = 0; od < input_dims.nbDims; od++) {
      MS_LOG(DEBUG) << "in tensor " << tensor.Name() << " dims at " << od << " is " << input_dims.d[od];
    }

    if (!this->trt_context_->setBindingDimensions(index, input_dims)) {
      MS_LOG(ERROR) << "invalid input dims of " << tensor.Name();
      return RET_ERROR;
    }
  }

  if (!this->trt_context_->allInputDimensionsSpecified()) {
    MS_LOG(ERROR) << "input dims need to be specified.";
    return RET_ERROR;
  }

  for (auto tensor : outputs_) {
    tensor.MutableData();
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor, tensor.DataSize());
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "malloc for outputs tensor device memory failed.";
      return RET_ERROR;
    }
    int index = this->engine_->getBindingIndex(tensor.Name().c_str());
    tensor_bindings_[index] = device_ptr;
    trt_out_tensor_name_.push_back(tensor.Name());
  }
  return RET_OK;
}

int TensorRTSubGraph::ReSize() {
  for (size_t i = 0; i < trt_in_tensor_name_.size(); i++) {
    // only support resize batch size
    for (int j = 0; j < this->network_->getNbInputs(); j++) {
      if (std::strcmp(this->network_->getInput(j)->getName(), trt_in_tensor_name_[i].c_str()) != 0) {
        continue;
      }
      nvinfer1::Dims contruct_dim = this->network_->getInput(j)->getDimensions();
      if (static_cast<size_t>(contruct_dim.nbDims) != inputs_[i].Shape().size()) {
        MS_LOG(ERROR) << "invalid resize input.";
        return RET_ERROR;
      }
      if (input_hw_index_ == -1) {
        // only NHWC format support HW resize, otherwise only support batchsize resize
        for (int d = 0; d < contruct_dim.nbDims; d++) {
          if (d != input_batchsize_index_ && contruct_dim.d[d] != inputs_[i].Shape()[d]) {
            MS_LOG(ERROR) << "only support dynamic batch size resize input.";
            return RET_ERROR;
          }
        }
      } else {
        if (contruct_dim.d[DIMENSION_4D - 1] != inputs_[i].Shape()[DIMENSION_4D - 1]) {
          MS_LOG(ERROR) << "don't support dynamic channel resize input.";
          return RET_ERROR;
        }
      }
    }
    MS_LOG(INFO) << "resize at input_batch_index " << input_batchsize_index_ << ", update batch size to "
                 << inputs_[i].Shape()[input_batchsize_index_];
    runtime_->SetBatchSize(inputs_[i].Shape()[input_batchsize_index_]);

    // inputs_ is dupulated by mindrt, name is untustable.
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(trt_in_tensor_name_[i], inputs_[i].DataSize(),
                                                                ConvertDataType(inputs_[i].DataType()));
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "realloc for input tensor device memory failed.";
      return RET_ERROR;
    }
    int index = this->engine_->getBindingIndex(trt_in_tensor_name_[i].c_str());
    tensor_bindings_[index] = device_ptr;
    // Set actual input size
    nvinfer1::Dims input_dims = ConvertCudaDims(inputs_[i].Shape());
    for (int od = 0; od < input_dims.nbDims; od++) {
      MS_LOG(DEBUG) << "in tensor " << trt_in_tensor_name_[i] << " dims at " << od << " is " << input_dims.d[od];
    }

    if (!this->trt_context_->setBindingDimensions(index, input_dims)) {
      MS_LOG(ERROR) << "invalid input dims of " << inputs_[i].Name();
      return RET_ERROR;
    }
  }
  if (!this->trt_context_->allInputDimensionsSpecified()) {
    MS_LOG(ERROR) << "input dims need to be specified.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < trt_out_tensor_name_.size(); i++) {
    int index = this->engine_->getBindingIndex(trt_out_tensor_name_[i].c_str());
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(trt_out_tensor_name_[i], outputs_[i].DataSize(),
                                                                ConvertDataType(outputs_[i].DataType()));
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "realloc for outputs tensor device memory failed.";
      return RET_ERROR;
    }
    tensor_bindings_[index] = device_ptr;
  }
  return RET_OK;
}

int TensorRTSubGraph::Execute() {
  lite::SetCudaDevice(device_info_);
  if (runtime_->GetBatchSize() <= 0) {
    MS_LOG(ERROR) << "TensorRTSubGraph has invalid batch size.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (runtime_->GetAllocator()->GetMemIsValid(trt_in_tensor_name_[i])) {
      MS_LOG(INFO) << "no need memcpy to cuda for input tensor: " << trt_in_tensor_name_[i];
      continue;
    }
    int ret = runtime_->GetAllocator()->SyncMemInHostAndDevice(inputs_[i], trt_in_tensor_name_[i], true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "sync mem from host to device failed for " << trt_in_tensor_name_[i];
      return ret;
    }
    runtime_->GetAllocator()->MarkMemValid(trt_in_tensor_name_[i], true);
  }

  auto ret = this->trt_context_->executeV2(tensor_bindings_);
  if (!ret) {
    MS_LOG(ERROR) << "TensorRT execute failed.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < trt_out_tensor_name_.size(); i++) {
    int index = this->engine_->getBindingIndex(trt_out_tensor_name_[i].c_str());
    // actual output tensor dims
    auto out_dims = this->trt_context_->getBindingDimensions(index);
    std::vector<int64_t> new_shape = lite::ConvertMSShape(out_dims);
    // batchsize resize need set new batch size
    if (runtime_->GetBatchSize() != new_shape[output_batchsize_index_]) {
      new_shape[output_batchsize_index_] = runtime_->GetBatchSize();
    }
    for (int od = 0; od < out_dims.nbDims; od++) {
      MS_LOG(DEBUG) << "out tensor " << trt_out_tensor_name_[i] << " dims at " << od << " is " << new_shape[od];
    }
    outputs_[i].SetShape(new_shape);

    if (outputs_[i].MutableData() == nullptr) {
      MS_LOG(ERROR) << "realloc for outputs tensor failed.";
      return RET_ERROR;
    }
    runtime_->GetAllocator()->MarkMemValid(trt_out_tensor_name_[i], true);
    int sync_ret = runtime_->GetAllocator()->SyncMemInHostAndDevice(outputs_[i], trt_out_tensor_name_[i], false);
    if (sync_ret != RET_OK) {
      MS_LOG(ERROR) << "sync mem from device to host failed for " << trt_out_tensor_name_[i];
      return sync_ret;
    }
    runtime_->GetAllocator()->MarkMemValid(trt_out_tensor_name_[i], false);
  }
  // make mem invalid, prepare for next execute
  for (size_t i = 0; i < inputs_.size(); i++) {
    runtime_->GetAllocator()->MarkMemValid(trt_in_tensor_name_[i], false);
  }
  return RET_OK;
}

ITensorHelper TensorRTSubGraph::FindTensorRTInputs(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor) {
  for (auto input_op : cur_op->in_ops()) {
    for (size_t i = 0; i < input_op->outputs().size(); i++) {
      auto out_tensor = input_op->outputs().at(i);
      if (in_tensor.Name().compare(out_tensor.Name()) == 0) {
        return input_op->GetInnerOutTensor().at(i);
      }
    }
  }
  return ITensorHelper{};
}
}  // namespace mindspore::lite
