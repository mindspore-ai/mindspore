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

#include "src/extendrt/delegate/tensorrt/tensorrt_subgraph.h"
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include "src/runtime/delegate/delegate_utils.h"

namespace mindspore::lite {
namespace {
size_t DataType2Size(DataType datatype) {
  std::map<DataType, size_t> TypeByte = {
    {DataType::kTypeUnknown, 0},       {DataType::kObjectTypeString, 0},  {DataType::kNumberTypeBool, 1},
    {DataType::kNumberTypeInt8, 1},    {DataType::kNumberTypeInt16, 2},   {DataType::kNumberTypeInt32, 4},
    {DataType::kNumberTypeInt64, 8},   {DataType::kNumberTypeUInt8, 1},   {DataType::kNumberTypeUInt16, 2},
    {DataType::kNumberTypeUInt32, 4},  {DataType::kNumberTypeUInt64, 8},  {DataType::kNumberTypeFloat16, 2},
    {DataType::kNumberTypeFloat32, 4}, {DataType::kNumberTypeFloat64, 8},
  };
  return TypeByte[datatype];
}
std::string GetNameByBindingIndex(const MSTensor &tensor, size_t index) {
  std::string tensor_name = tensor.Name();
  if (index != 0) {
    tensor_name += " [profile " + std::to_string(index) + "]";
  }
  return tensor_name;
}
std::string GetNameByBindingIndex(const std::string &name, size_t index) {
  std::string tensor_name = name;
  if (index != 0) {
    tensor_name += " [profile " + std::to_string(index) + "]";
  }
  return tensor_name;
}
}  // namespace
TensorRTSubGraph::~TensorRTSubGraph() {
  if (ctx_ != nullptr) {
    delete ctx_;
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
    delete[] tensor_bindings_;
    tensor_bindings_ = nullptr;
  }
  for (auto op : all_ops_) {
    delete op;
  }
}

std::experimental::optional<bool> TensorRTSubGraph::IsValidProfileDims() const {
  if (min_dims_.empty() && opt_dims_.empty() && max_dims_.empty()) {
    MS_LOG(WARNING) << "number of min opt max profile number is 0 !";
    return false;
  }
  if (min_dims_.size() != opt_dims_.size() || min_dims_.size() != max_dims_.size()) {
    MS_LOG(ERROR) << "number of min opt max profile tensor name not equal !";
    return {};
  }
  std::unordered_map<std::string, int> tensor_name2profile_num;
  for (auto it = min_dims_.begin(); it != min_dims_.end(); ++it) {
    if (max_dims_.find(it->first) == max_dims_.end() || opt_dims_.find(it->first) == opt_dims_.end()) {
      MS_LOG(ERROR) << "min opt max profile name set not equal !";
      return {};
    }
    tensor_name2profile_num[it->first] = it->second.size();
    if (tensor_name2profile_num[it->first] == 0) {
      MS_LOG(ERROR) << "min dims profile num for " << it->first << " is 0!";
      return {};
    }
    int nbdims = it->second.front().nbDims;
    if (opt_dims_.at(it->first).size() != tensor_name2profile_num[it->first]) {
      MS_LOG(ERROR) << "opt dims profile num for " << it->first << " is not equal min dims!";
      return {};
    }
    if (max_dims_.at(it->first).size() != tensor_name2profile_num[it->first]) {
      MS_LOG(ERROR) << "max dims profile num for " << it->first << " is not equal min dims!";
      return {};
    }
    if (std::any_of(min_dims_.at(it->first).begin(), min_dims_.at(it->first).end(),
                    [=](const nvinfer1::Dims &dims) { return dims.nbDims != nbdims; })) {
      MS_LOG(ERROR) << "min dims profile dims for " << it->first << " is not equal!";
      return {};
    }
    if (std::any_of(opt_dims_.at(it->first).begin(), opt_dims_.at(it->first).end(),
                    [=](const nvinfer1::Dims &dims) { return dims.nbDims != nbdims; })) {
      MS_LOG(ERROR) << "opt dims profile dims for " << it->first << " is not equal to min dims!";
      return {};
    }
    if (std::any_of(opt_dims_.at(it->first).begin(), opt_dims_.at(it->first).end(),
                    [=](const nvinfer1::Dims &dims) { return dims.nbDims != nbdims; })) {
      MS_LOG(ERROR) << "max dims profile dims for " << it->first << " is not equal to min dims!";
      return {};
    }
  }
  auto it = tensor_name2profile_num.begin();
  if (std::any_of(tensor_name2profile_num.begin(), tensor_name2profile_num.end(),
                  [=](const std::pair<std::string, int> &p) { return p.second != it->second; })) {
    MS_LOG(WARNING) << "different tensor profile num not equal!";
    return false;
  }
  return true;
}

int TensorRTSubGraph::Init(cudaStream_t stream) {
  auto ret = GetGraphInOutOps(inputs_, outputs_, &in_ops_, &out_ops_, all_ops_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get TensorRT subgraph input and output ops failed.";
    return RET_ERROR;
  }
  ctx_ = new TensorRTContext();
  if (ctx_ == nullptr) {
    MS_LOG(ERROR) << "New TensorRTContext failed.";
    return RET_ERROR;
  }
  ctx_->SetRuntime(runtime_);
  if (!ctx_->Init()) {
    MS_LOG(ERROR) << "New TensorRTContext failed.";
    return RET_ERROR;
  }
  if (SetDeviceConfig(stream) != RET_OK) {
    MS_LOG(WARNING) << "set tensorrt config failed.";
  }
  serializer_ = std::make_shared<TensorRTSerializer>(serialize_file_path_);
  if (serializer_ == nullptr) {
    MS_LOG(ERROR) << "create Serializer failed.";
    return RET_ERROR;
  }
  auto valid_opt = IsValidProfileDims();
  if (!valid_opt) {
    MS_LOG(ERROR) << "Config is not valid";
    return RET_ERROR;
  }
  using_input_ranges_ = valid_opt.value();
  if (using_input_ranges_) {
    for (size_t i = 0; i != min_dims_.begin()->second.size(); ++i) {
      profiles_.push_back(runtime_->GetBuilder()->createOptimizationProfile());
    }
  } else {
    profiles_.push_back(runtime_->GetBuilder()->createOptimizationProfile());
  }
  for (size_t i = 0; i != profiles_.size(); ++i) {
    if (profiles_[i] == nullptr) {
      MS_LOG(ERROR) << "create optimization profile failed.";
      return RET_ERROR;
    }
  }
  engine_ = serializer_->GetSerializedEngine();
  if (engine_ != nullptr) {
    MS_LOG(INFO) << "using serialized engine " << serialize_file_path_;
    return RET_OK;
  }
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (inputs_[i].Shape().size() != DIMENSION_4D) {
      input_hw_index_ = -1;
    }
  }
  return RET_OK;
}

int TensorRTSubGraph::BuildEngine() {
  // print all network ops
  for (auto &profile : profiles_) {
    if (this->config_->addOptimizationProfile(profile) == -1) {
      MS_LOG(ERROR) << "addOptimizationProfile failed.";
      return RET_ERROR;
    }
  }
  MS_LOG(INFO) << "build engine for tensorrt network: " << ctx_->network()->getName();
  for (int i = 0; i < ctx_->network()->getNbLayers(); i++) {
    MS_LOG(DEBUG) << "tensorrt op: " << ctx_->network()->getLayer(i)->getName();
  }
  MS_LOG(DEBUG) << "end of tensorrt network: " << ctx_->network()->getName();

  this->engine_ = runtime_->GetBuilder()->buildEngineWithConfig(*ctx_->network(), *this->config_);
  if (this->engine_ == nullptr) {
    MS_LOG(ERROR) << "Create engine failed in TensorRT network";
    return RET_ERROR;
  }
  if (serialize_file_path_.size() > 0) {
    serializer_->SaveSerializedEngine(engine_);
  }
  return RET_OK;
}

int TensorRTSubGraph::SetDeviceConfig(cudaStream_t stream) {
  if (config_ == nullptr) {
    this->config_ = runtime_->GetBuilder()->createBuilderConfig();
    if (this->config_ == nullptr) {
      MS_LOG(ERROR) << "create builder config failed.";
      return RET_ERROR;
    }
  }
  // set fp16
  if (device_info_->GetEnableFP16() && runtime_->GetBuilder()->platformHasFastFp16()) {
    MS_LOG(INFO) << "set fp16 flag successfully for tensorrt.";
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    runtime_->SetRuntimePrecisionMode(RuntimePrecisionMode_FP16);
  }

  // set int8
  if (IsInt8Mode() && runtime_->GetBuilder()->platformHasFastInt8()) {
    MS_LOG(INFO) << "set int8 flag successfully for tensorrt.";
    config_->setFlag(nvinfer1::BuilderFlag::kINT8);
    // Mark calibrator as null
    config_->setInt8Calibrator(nullptr);
    input_hw_index_ = -1;
  } else {
    MS_LOG(INFO) << "inputs no quant params or platform not support int8.";
  }
  runtime_->SetCudaStream(stream);
  config_->setProfileStream(stream);
  stream_ = stream;
  MS_LOG(INFO) << GetRankID() << " tensorrt subgraph stream: " << stream_;

  // config setMaxWorkspaceSize to 1152 MB for max limit
  config_->setMaxWorkspaceSize(2047 * (1 << 20));
  return RET_OK;
}

bool TensorRTSubGraph::IsInt8Mode() {
  for (auto cur_op : all_ops_) {
    if (cur_op->GetQuantType() == schema::QuantType_QUANT_ALL) {
      return true;
    }
  }
  return false;
}

nvinfer1::ITensor *TensorRTSubGraph::SetTensorRTNetworkInput(const mindspore::MSTensor &in_tensor, size_t index) {
  for (int i = 0; i < ctx_->network()->getNbInputs(); i++) {
    if (in_tensor.Name().compare(ctx_->network()->getInput(i)->getName()) == 0) {
      MS_LOG(INFO) << "input tensor is already added in network: " << in_tensor.Name();
      return ctx_->network()->getInput(i);
    }
  }

  auto cuda_dtype = ConvertDataType(in_tensor.DataType());
  if (static_cast<int>(cuda_dtype) == -1) {
    MS_LOG(ERROR) << "Unsupported input data type " << static_cast<int>(in_tensor.DataType());
    return nullptr;
  }
  nvinfer1::Dims input_dims;
  if (using_input_ranges_) {
    if (min_dims_.find(in_tensor.Name()) == min_dims_.end()) {
      MS_LOG(ERROR) << "profile config do not have input tensor name : " << in_tensor.Name();
      return nullptr;
    }
    input_dims = SetInputDimsProfile(in_tensor);
  } else {
    input_dims = ParseInputDimsProfile(in_tensor);
  }
  MS_LOG(INFO) << "add network input: " << in_tensor.Name();
  return ctx_->network()->addInput(in_tensor.Name().c_str(), cuda_dtype, input_dims);
}

nvinfer1::Dims TensorRTSubGraph::SetInputDimsProfile(const mindspore::MSTensor &in_tensor) {
  auto &min_profile_dims = min_dims_[in_tensor.Name()];
  auto &max_profile_dims = max_dims_[in_tensor.Name()];
  nvinfer1::Dims input_dims;
  input_dims.nbDims = min_profile_dims.front().nbDims;
  for (int i = 0; i != input_dims.nbDims; ++i) {
    int min_dim = std::numeric_limits<int>::max();
    int max_dim = std::numeric_limits<int>::min();
    for (int j = 0; j != min_profile_dims.size(); ++j) {
      min_dim = std::min(min_profile_dims[j].d[i], min_dim);
      max_dim = std::max(max_profile_dims[j].d[i], max_dim);
    }
    input_dims.d[i] = (min_dim == max_dim ? min_dim : -1);
  }
  DebugDims("input dims", input_dims);
  for (int i = 0; i != opt_dims_[in_tensor.Name()].size(); ++i) {
    if (!profiles_[i]->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMIN,
                                     min_dims_[in_tensor.Name()][i])) {
      MS_LOG(ERROR) << "setDimensions of kMIN failed for " << in_tensor.Name();
      return input_dims;
    }
    if (!profiles_[i]->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kOPT,
                                     opt_dims_[in_tensor.Name()][i])) {
      MS_LOG(ERROR) << "setDimensions of kOPT failed for " << in_tensor.Name();
      return input_dims;
    }
    if (!profiles_[i]->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMAX,
                                     max_dims_[in_tensor.Name()][i])) {
      MS_LOG(ERROR) << "setDimensions of kMAX failed for " << in_tensor.Name();
      return input_dims;
    }
    DebugDims("min dims", min_dims_[in_tensor.Name()][i]);
    DebugDims("opt dims", opt_dims_[in_tensor.Name()][i]);
    DebugDims("max dims", max_dims_[in_tensor.Name()][i]);
  }
  return input_dims;
}

nvinfer1::Dims TensorRTSubGraph::ParseInputDimsProfile(const mindspore::MSTensor &in_tensor) {
  nvinfer1::Dims input_dims = ConvertCudaDims(in_tensor.Shape());
  if (input_batchsize_index_ != -1) {
    input_dims.d[0] = -1;
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
  if (input_batchsize_index_ != -1) {
    input_dims_min.d[0] = 1;
    if (input_hw_index_ != -1) {
      input_dims_min.d[input_hw_index_] = 1;
      input_dims_min.d[input_hw_index_ + 1] = 1;
    }
  }
  if (!profiles_.front()->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMIN, input_dims_min)) {
    MS_LOG(ERROR) << "setDimensions of kMIN failed for " << in_tensor.Name();
    return input_dims;
  }
  nvinfer1::Dims input_dims_opt = ConvertCudaDims(in_tensor.Shape());
  if (!profiles_.front()->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kOPT, input_dims_opt)) {
    MS_LOG(ERROR) << "setDimensions of kOPT failed for " << in_tensor.Name();
    return input_dims;
  }
  nvinfer1::Dims input_dims_max = ConvertCudaDims(in_tensor.Shape());
  // input_dims_max should be the same with input network dims
  if (!profiles_.front()->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMAX, input_dims_max)) {
    MS_LOG(ERROR) << "setDimensions of kMAX failed for " << in_tensor.Name();
    return input_dims;
  }
  min_dims_[in_tensor.Name()].push_back(input_dims_min);
  opt_dims_[in_tensor.Name()].push_back(input_dims_opt);
  max_dims_[in_tensor.Name()].push_back(input_dims_max);
  DebugDims("input min dims", input_dims_min);
  DebugDims("input opt dims", input_dims_opt);
  DebugDims("input max dims", input_dims_max);
  return input_dims;
}

int TensorRTSubGraph::ParseInputsProfile() {
  MS_LOG(INFO) << "using serialied engine.";
  for (auto in_tensor : inputs_) {
    auto dim = ParseInputDimsProfile(in_tensor);
    if (dim.nbDims <= 0) {
      MS_LOG(ERROR) << "input dims is invalid.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int TensorRTSubGraph::GetInputIndexByName(const std::string &name) {
  for (int i = 0; i != inputs().size(); ++i) {
    if (inputs()[i].Name() == name) {
      return i;
    }
  }
  return -1;
}

int TensorRTSubGraph::BuildTensorRTGraph() {
  MS_ASSERT(!all_ops_.empty());
  int ret;
  if (engine_ != nullptr) {
    return ParseInputsProfile();
  }
  // build engine online
  for (auto cur_op : all_ops_) {
    cur_op->SetRuntime(runtime_);
    for (int i = 0; i != cur_op->inputs().size(); ++i) {
      // Data From CPU
      auto in_tensor = cur_op->inputs()[i];
      if (IsSubGraphInputTensor(this->inputs(), in_tensor)) {
        nvinfer1::ITensor *trt_tensor = SetTensorRTNetworkInput(in_tensor, GetInputIndexByName(in_tensor.Name()));
        if (trt_tensor == nullptr) {
          MS_LOG(ERROR) << "SetTensorRTNetworkInput failed for " << in_tensor.Name();
          return RET_ERROR;
        }

        // avoid bool input tensor
        cur_op->SetSupportInputBool(false);

        ctx_->RegisterTensorWithSameName(ITensorHelper{trt_tensor, in_tensor.format(), true}, in_tensor.Name());
        continue;
      }

      ITensorHelper trt_tensor = FindTensorRTInputs(cur_op, in_tensor);
      if (trt_tensor.trt_tensor_ == nullptr) {
        // weight tensor
        if (IsCached(cur_op, in_tensor) && in_tensor.Data() != nullptr) {
          ret = HandleCacheTensor(cur_op, in_tensor);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "HandleCacheTensor failed for " << in_tensor.Name();
            return RET_ERROR;
          }
        } else if (trt_specific_weight_nodes_.find(cur_op->type()) == trt_specific_weight_nodes_.end()) {
          if (in_tensor.Data() == nullptr) {
            MS_LOG(ERROR) << "Weight Tensor data is nullptr.";
            return RET_ERROR;
          }
          trt_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx_, in_tensor, cur_op->GetOpName());
          trt_tensor.format_ = Format::NHWC;
          MS_LOG(INFO) << "auto convert constant tensor for: " << in_tensor.Name();
          ctx_->RegisterTensor(trt_tensor, in_tensor.Name());
        }
      } else {
        ctx_->RegisterTensor(trt_tensor, in_tensor.Name());
      }
    }
    MS_LOG(DEBUG) << "Parsing TensorRT op for " << cur_op->GetOpName();

    ret = cur_op->AddInnerOp(ctx_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Add op failed in TensorRT network: " << cur_op->GetOpName();
      return RET_ERROR;
    }
    ret = cur_op->SetInt8DynamicRange(ctx_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set Int8 dynamic range failed in TensorRT network: " << cur_op->GetOpName();
      return RET_ERROR;
    }
  }
  ret = MarkOutputs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MarkOutputs failed in TensorRT network";
    return ret;
  }

  std::string network_name = "network_" + std::string(ctx_->network()->getInput(0)->getName()) + "_" +
                             std::string(ctx_->network()->getOutput(0)->getName());
  ctx_->network()->setName(network_name.c_str());
  this->name_ = network_name;
  ret = BuildEngine();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create engine failed in TensorRT network";
    return ret;
  }
  return RET_OK;
}

bool TensorRTSubGraph::OutputFormatCheck(ITensorHelper output_helper, const mindspore::MSTensor &out_tensor) {
  auto out_shape = out_tensor.Shape();
  auto out_dims = output_helper.trt_tensor_->getDimensions();
  if (out_dims.nbDims != DIMENSION_4D) {
    return false;
  }
  if (output_helper.format_ == Format::NHWC) {
    return false;
  }
  if (out_shape.empty()) {
    return false;
  }
  for (int i = 0; i < out_dims.nbDims; i++) {
    if (out_shape[i] == -1) {
      return false;
    }
  }
  if (SameDims(out_dims, out_shape)) {
    return false;
  }
  return true;
}

int TensorRTSubGraph::MarkOutputs() {
  // Mark NetWork Output Tensor.
  for (const auto &out_tensor : outputs_) {
    for (auto out_op : this->out_ops_) {
      for (size_t index = 0; index < out_op->outputs().size(); index++) {
        if (out_op->outputs()[index] == out_tensor) {
          MS_LOG(INFO) << "markOutput for: " << out_tensor.Name();
          auto output_helper = out_op->output(ctx_, index);
          nvinfer1::ITensor *out_trt_tensor = output_helper.trt_tensor_;
          if (OutputFormatCheck(output_helper, out_tensor)) {
            // transpose subgraph output from nchw to nhwc
            nvinfer1::IShuffleLayer *transpose_layer_out = NCHW2NHWC(ctx_, *output_helper.trt_tensor_);
            if (transpose_layer_out == nullptr) {
              MS_LOG(ERROR) << "op action convert failed";
              return RET_ERROR;
            }
            transpose_layer_out->setName((out_tensor.Name() + "_transpose2NHWC").c_str());
            out_trt_tensor = transpose_layer_out->getOutput(0);
          }

          out_trt_tensor->setName(("__" + out_tensor.Name()).c_str());
          out_trt_tensor = ctx_->network()->addIdentity(*out_trt_tensor)->getOutput(0);
          out_trt_tensor->setName(out_tensor.Name().c_str());
          ctx_->network()->markOutput(*out_trt_tensor);
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
  int ret = lite::SetCudaDevice(device_info_);
  if (ret != RET_OK) {
    return ret;
  }
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

  profile_index_ = MaxVolumnProfileIndex();
  if (this->trt_context_->setOptimizationProfile(profile_index_)) {
    MS_LOG(INFO) << "setOptimizationProfile: " << profile_index_;
  }
  for (size_t i = 0; i != inputs_.size(); ++i) {
    auto &tensor = inputs_[i];
    int volumn =
      std::accumulate(max_dims_[tensor.Name()][profile_index_].d,
                      max_dims_[tensor.Name()][profile_index_].d + max_dims_[tensor.Name()][profile_index_].nbDims, 1,
                      std::multiplies<int>());
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor, volumn * DataType2Size(tensor.DataType()));
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "malloc for inputs tensor device memory failed.";
      return RET_ERROR;
    }
    trt_in_tensor_name_.push_back(tensor.Name());
    std::string tensor_name = GetNameByBindingIndex(tensor, profile_index_);
    int index = this->engine_->getBindingIndex(tensor_name.c_str());
    MS_LOG(INFO) << "device index " << index << " for tensor : " << tensor_name << " attr: " << device_ptr;
    tensor_bindings_[index] = device_ptr;
    nvinfer1::Dims input_dims = max_dims_[tensor.Name()][profile_index_];
    for (int od = 0; od < input_dims.nbDims; od++) {
      MS_LOG(DEBUG) << "in tensor " << tensor.Name() << " dims at " << od << " is " << input_dims.d[od];
    }
    if (!this->trt_context_->setBindingDimensions(index, input_dims)) {
      MS_LOG(ERROR) << "invalid input dims of " << tensor.Name();
      return RET_ERROR;
    }
  }

  // malloc for cache weight tensor
  for (auto cache_tensor : cache_const_inputs_) {
    size_t data_size = cache_mgr_->GetCacheDataSize(cache_tensor);
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(cache_tensor, data_size);
    runtime_->GetAllocator()->MarkMemValid(cache_tensor.Name().c_str(), true);
    int index = this->engine_->getBindingIndex(cache_tensor.Name().c_str());
    tensor_bindings_[index] = device_ptr;
    auto cache_ret = cache_mgr_->SetDeviceCacheAddr(cache_tensor.Name(), device_ptr, data_size);
    if (cache_ret != kSuccess) {
      MS_LOG(ERROR) << "SetDeviceCacheAddr failed, cache tensor: " << cache_tensor.Name();
      return RET_ERROR;
    }
  }

  if (!this->trt_context_->allInputDimensionsSpecified()) {
    MS_LOG(ERROR) << "input dims need to be specified.";
    return RET_ERROR;
  }
  for (auto op : all_ops_) {
    ret = op->Prepare(tensor_bindings_, engine_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "prepare op failed of " << op->GetOpName();
      return RET_ERROR;
    }
  }
  for (auto tensor : outputs_) {
    std::string max_tensor_name = GetNameByBindingIndex(tensor, profile_index_);
    int max_index = this->engine_->getBindingIndex(max_tensor_name.c_str());
    auto out_dims = trt_context_->getBindingDimensions(max_index);
    int elem_num = std::accumulate(out_dims.d, out_dims.d + out_dims.nbDims, 1, std::multiplies<int>());
    DebugDims("out dims", out_dims);
    if (tensor.Data() == nullptr) {
      MS_LOG(INFO) << "Set output shape by tensorrt binding output";
      tensor.SetShape(lite::ConvertMSShape(out_dims));
      tensor.MutableData();
    }
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor, elem_num * DataType2Size(tensor.DataType()));
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "malloc for outputs tensor device memory failed.";
      return RET_ERROR;
    }
    for (size_t j = 0; j != profiles_.size(); ++j) {
      std::string tensor_name = GetNameByBindingIndex(tensor, j);
      int index = this->engine_->getBindingIndex(tensor_name.c_str());
      tensor_bindings_[index] = device_ptr;
    }
    trt_out_tensor_name_.push_back(tensor.Name());
  }
  return RET_OK;
}

std::experimental::optional<int> TensorRTSubGraph::SelectProfile() const {
  std::vector<int> profile_index;
  for (int i = 0; i != profiles_.size(); ++i) {
    bool condition = true;
    for (size_t j = 0; j != trt_in_tensor_name_.size(); ++j) {
      auto &tensor = inputs_[j];
      nvinfer1::Dims input_dims = ConvertCudaDims(inputs_[j].Shape());
      for (int od = 0; od < input_dims.nbDims; od++) {
        if (input_dims.d[od] < min_dims_.at(trt_in_tensor_name_[j])[i].d[od] ||
            input_dims.d[od] > max_dims_.at(trt_in_tensor_name_[j])[i].d[od]) {
          condition = false;
          break;
        }
      }
    }
    if (condition) {
      profile_index.push_back(i);
    }
  }
  if (profile_index.empty()) {
    return {};
  }
  return profile_index.front();
}

size_t TensorRTSubGraph::MaxVolumnProfileIndex() const {
  int max_volumn = std::numeric_limits<int>::min();
  size_t max_volumn_index = 0;
  for (size_t i = 0; i != max_dims_.begin()->second.size(); ++i) {
    // depend on the first input tensor
    int volumn =
      std::accumulate(max_dims_.begin()->second[i].d,
                      max_dims_.begin()->second[i].d + max_dims_.begin()->second[i].nbDims, 1, std::multiplies<int>());
    if (volumn > max_volumn) {
      max_volumn_index = i;
      max_volumn = volumn;
    }
  }
  return max_volumn_index;
}

int TensorRTSubGraph::ReSize() {
  auto profile_index_opt = SelectProfile();
  if (!profile_index_opt) {
    MS_LOG(ERROR) << "do not have profile in range!";
    return RET_ERROR;
  }
  profile_index_ = profile_index_opt.value();
  if (this->trt_context_->setOptimizationProfile(profile_index_)) {
    MS_LOG(INFO) << "setOptimizationProfile: " << profile_index_;
  }
  for (size_t i = 0; i < trt_in_tensor_name_.size(); i++) {
    if (ctx_->network() != nullptr) {
      for (int j = 0; j < ctx_->network()->getNbInputs(); j++) {
        if (trt_in_tensor_name_[i].compare(ctx_->network()->getInput(j)->getName()) != 0) {
          continue;
        }
        nvinfer1::Dims construct_dims = ctx_->network()->getInput(j)->getDimensions();
        bool ret = ValidInputResizeDims(construct_dims, inputs_[i].Shape());
        if (!ret) {
          MS_LOG(ERROR) << "input resize shape is invalid.";
          return RET_ERROR;
        }
      }
    }

    // inputs_ is dupulated by mindrt, name is untustable.
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(trt_in_tensor_name_[i], inputs_[i].DataSize(),
                                                                ConvertDataType(inputs_[i].DataType()));
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "realloc for input tensor device memory failed.";
      return RET_ERROR;
    }
    int index = this->engine_->getBindingIndex(GetNameByBindingIndex(trt_in_tensor_name_[i], profile_index_).c_str());
    MS_LOG(INFO) << "device index " << index << " for tensor : " << trt_in_tensor_name_[i] << " attr: " << device_ptr;
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
    int index = this->engine_->getBindingIndex(GetNameByBindingIndex(trt_out_tensor_name_[i], profile_index_).c_str());
    auto out_dims = trt_context_->getBindingDimensions(index);
    DebugDims("out dims", out_dims);
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

bool TensorRTSubGraph::ValidInputResizeDims(const nvinfer1::Dims &construct_dims,
                                            const std::vector<int64_t> &resize_input_shape) {
  if (static_cast<size_t>(construct_dims.nbDims) != resize_input_shape.size()) {
    MS_LOG(ERROR) << "invalid resize input.";
    return false;
  }
  return true;
}

int TensorRTSubGraph::Execute() {
  int ret = lite::SetCudaDevice(device_info_);
  if (ret != RET_OK) {
    return ret;
  }
  for (size_t i = 0; i < inputs_.size(); i++) {
    if (runtime_->GetAllocator()->GetMemIsValid(trt_in_tensor_name_[i])) {
      MS_LOG(INFO) << "no need memcpy to cuda for input tensor: " << trt_in_tensor_name_[i];
      continue;
    }

    auto iter = model_input_to_cache_tensors_.find(trt_in_tensor_name_[i]);
    if (iter != model_input_to_cache_tensors_.end()) {
      for (auto &cache_tensor : iter->second) {
        ret = cache_mgr_->CacheHandle(cache_tensor.Name(), inputs_[i],
                                      runtime_->GetAllocator()->GetDevicePtr(trt_in_tensor_name_[i]));
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "handle cache failed " << trt_in_tensor_name_[i];
          return RET_ERROR;
        }
        runtime_->GetAllocator()->MarkMemValid(trt_in_tensor_name_[i], true);
        MS_LOG(DEBUG) << cache_tensor.Name() << " CacheHandle succ " << trt_in_tensor_name_[i];
      }
      continue;
    }

    ret = runtime_->GetAllocator()->SyncMemInHostAndDevice(inputs_[i], trt_in_tensor_name_[i], true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "sync mem from host to device failed for " << trt_in_tensor_name_[i];
      return ret;
    }
    runtime_->GetAllocator()->MarkMemValid(trt_in_tensor_name_[i], true);
  }

  if (!this->trt_context_->executeV2(tensor_bindings_)) {
    MS_LOG(ERROR) << "TensorRT execute failed.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < trt_out_tensor_name_.size(); i++) {
    int index = this->engine_->getBindingIndex(GetNameByBindingIndex(trt_out_tensor_name_[i], profile_index_).c_str());
    // actual output tensor dims
    auto out_dims = this->trt_context_->getBindingDimensions(index);
    std::vector<int64_t> new_shape = lite::ConvertMSShape(out_dims);
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
        return input_op->output(ctx_, i);
      }
    }
  }
  return ITensorHelper{};
}
bool TensorRTSubGraph::IsCached(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor) {
  return cache_mgr_ != nullptr && cache_mgr_->IsCacheTensor(in_tensor);
}

void TensorRTSubGraph::FindCacheTensorInfo(TensorRTOp *cur_op, mindspore::MSTensor device_cache_tensor) {
  auto iter = network_cache_tensor_info_.find(cur_op->GetOpName());
  if (iter != network_cache_tensor_info_.end()) {
    return;
  }
  std::queue<TensorRTOp *> front_ops;
  front_ops.push(cur_op);
  network_cache_tensor_info_[cur_op->GetOpName()].front_op_can_cache_ = true;
  iter = network_cache_tensor_info_.find(cur_op->GetOpName());
  while (!front_ops.empty()) {
    auto front_op = front_ops.front();
    iter->second.front_op_can_cache_ = CanOpCache(front_op) ? iter->second.front_op_can_cache_ : false;
    for (auto in_tensor : front_op->inputs()) {
      if (IsSubGraphInputTensor(this->inputs(), in_tensor)) {
        iter->second.network_input_tensor_.push_back(in_tensor);
        model_input_to_cache_tensors_[in_tensor.Name()].push_back(device_cache_tensor);
        MS_LOG(DEBUG) << cur_op->GetOpName() << "'s network input tensor name is " << in_tensor.Name()
                      << ", can cache: " << iter->second.front_op_can_cache_;
      }
    }
    for (auto fronts_op : front_op->in_ops()) {
      front_ops.push(fronts_op);
    }
    front_ops.pop();
  }
}

bool TensorRTSubGraph::CanOpCache(TensorRTOp *cur_op) { return true; }

int TensorRTSubGraph::HandleCacheTensor(TensorRTOp *cur_op, const mindspore::MSTensor &in_tensor) {
  FindCacheTensorInfo(cur_op, in_tensor);
  // cache kernel weight tensor
  cache_const_inputs_.push_back(in_tensor);
  auto shape = cache_mgr_->GetCacheShape(in_tensor);
  MS_LOG(INFO) << "auto add cache constant tensor for: " << in_tensor.Name();
  auto cuda_dtype = ConvertDataType(in_tensor.DataType());
  nvinfer1::Dims input_dims = ConvertCudaDims(shape);
  nvinfer1::ITensor *cache_input = ctx_->network()->addInput(in_tensor.Name().c_str(), cuda_dtype, input_dims);
  if (cache_input == nullptr) {
    MS_LOG(ERROR) << "add cache Weight Tensor data is nullptr.";
    return RET_ERROR;
  }
  for (int i = 0; i != opt_dims_.size(); ++i) {
    if (!profiles_.front()->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMIN, input_dims)) {
      MS_LOG(ERROR) << "setDimensions of kMIN failed for " << in_tensor.Name();
      return RET_ERROR;
    }
    if (!profiles_.front()->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kOPT, input_dims)) {
      MS_LOG(ERROR) << "setDimensions of kOPT failed for " << in_tensor.Name();
      return RET_ERROR;
    }
    if (!profiles_.front()->setDimensions(in_tensor.Name().c_str(), nvinfer1::OptProfileSelector::kMAX, input_dims)) {
      MS_LOG(ERROR) << "setDimensions of kMAX failed for " << in_tensor.Name();
      return RET_ERROR;
    }
  }
  ITensorHelper trt_tensor{cache_input, Format::NHWC, true};
  ctx_->RegisterTensor(trt_tensor, in_tensor.Name());
  return RET_OK;
}
}  // namespace mindspore::lite
