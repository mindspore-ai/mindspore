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
#include <fstream>
#include <limits>
#include <unordered_map>
#include <iomanip>
#include "src/extendrt/delegate/delegate_utils.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/common/utils.h"

#include "ops/transpose.h"
#include "ops/reshape.h"
#include "ops/strided_slice.h"
#include "ops/expand_dims.h"
#include "ops/fusion/topk_fusion.h"
#include "ops/broadcast_to.h"

namespace mindspore::lite {
TensorRTSubGraph::TensorRTSubGraph(std::vector<TensorRTOp *> ops, const std::vector<TensorInfo> &inputs,
                                   const std::vector<TensorInfo> &outputs, const mindspore::Context *ctx,
                                   std::shared_ptr<GPUDeviceInfo> device_info, TensorRTRuntime *runtime,
                                   bool support_resize, bool support_hw_resize,
                                   const ProfileConfigs &trt_profile_config)
    : inputs_(inputs),
      outputs_(outputs),
      all_ops_(std::move(ops)),
      device_info_(device_info),
      runtime_(runtime),
      trt_profile_config_(trt_profile_config) {
  trt_specific_weight_handled_inner_ = {
    ops::kNameTranspose, ops::kNameReshape, ops::kNameExpandDims, ops::kNameTopKFusion, ops::kNameBroadcastTo,
  };
  if (!support_resize) {
    input_batchsize_index_ = -1;
    input_hw_index_ = -1;
  }
  if (!support_hw_resize) {
    input_hw_index_ = -1;
  }
}

TensorRTSubGraph::~TensorRTSubGraph() {
  if (ctx_ != nullptr) {
    delete ctx_;
  }
  if (config_ != nullptr) {
    config_->destroy();
    config_ = nullptr;
  }
#ifdef PROFILER_
  auto profile = dynamic_cast<SimpleProfiler *>(trt_context_->getProfiler());
  if (profile != nullptr) std::cout << *profile << std::endl;
  delete profile;
#endif
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

bool TensorRTSubGraph::IsValidProfileDims() const {
  if (trt_profile_config_.profiles.empty()) {
    MS_LOG(INFO) << "Number of profiles is 0.";
    return false;
  }
  for (auto &profile : trt_profile_config_.profiles) {
    if (profile.inputs.size() != trt_profile_config_.input_infos.size()) {
      MS_LOG(WARNING) << "Profile input size " << profile.inputs.size() << " != input shape size "
                      << trt_profile_config_.input_infos.size();
      return false;
    }
    for (size_t i = 0; i < profile.inputs.size(); i++) {
      const auto &profile_input = profile.inputs[i];
      const auto &input_info = trt_profile_config_.input_infos[i];
      if (profile_input.min_dims.size() != input_info.input_shape.size()) {
        MS_LOG(WARNING) << "Profile input " << input_info.name << " min dims number " << profile_input.min_dims.size()
                        << " != input shape dim number " << input_info.input_shape.size();
        return false;
      }
      if (profile_input.max_dims.size() != input_info.input_shape.size()) {
        MS_LOG(WARNING) << "Profile input " << input_info.name << " max dims number " << profile_input.max_dims.size()
                        << " != input shape dim number " << input_info.input_shape.size();
        return false;
      }
      if (profile_input.opt_dims.size() != input_info.input_shape.size()) {
        MS_LOG(WARNING) << "Profile input " << input_info.name << " opt dims number " << profile_input.opt_dims.size()
                        << " != input shape dim number " << input_info.input_shape.size();
        return false;
      }
    }
  }
  return true;
}

int TensorRTSubGraph::Init(cudaStream_t stream, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
  auto ret = GetGraphInOutOps(inputs_, outputs_, &in_ops_, &out_ops_, all_ops_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get TensorRT subgraph input and output ops failed.";
    return RET_ERROR;
  }
  ctx_ = new (std::nothrow) TensorRTContext();
  if (ctx_ == nullptr) {
    MS_LOG(ERROR) << "New TensorRTContext failed.";
    return RET_ERROR;
  }
  ctx_->SetRuntime(runtime_);
  if (!ctx_->Init()) {
    MS_LOG(ERROR) << "New TensorRTContext failed.";
    return RET_ERROR;
  }
  if (SetDeviceConfig(stream, cublas_handle, cublaslt_handle) != RET_OK) {
    MS_LOG(WARNING) << "set tensorrt config failed.";
  }
  serializer_ = std::make_shared<TensorRTSerializer>(serialize_file_path_);
  if (serializer_ == nullptr) {
    MS_LOG(ERROR) << "create Serializer failed.";
    return RET_ERROR;
  }
  using_input_ranges_ = IsValidProfileDims();
  if (using_input_ranges_) {
    for (size_t i = 0; i != trt_profile_config_.profiles.size(); ++i) {
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

int TensorRTSubGraph::SetDeviceConfig(cudaStream_t stream, cublasHandle_t cublas_handle,
                                      cublasLtHandle_t cublaslt_handle) {
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
  runtime_->SetCudaStream(stream, cublas_handle, cublaslt_handle);
  config_->setProfileStream(stream);
  stream_ = stream;

  MS_LOG(INFO) << GetRankID() << " tensorrt subgraph stream: " << stream_;

  // config setMaxWorkspaceSize to 2047 MB for max limit
  constexpr size_t kWorkspaceSize = 2047 * (1 << 20);
  config_->setMaxWorkspaceSize(kWorkspaceSize);
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

nvinfer1::ITensor *TensorRTSubGraph::SetTensorRTNetworkInput(const TensorInfo &in_tensor, int index) {
  if (index < 0) {
    return nullptr;
  }
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
    input_dims = SetInputDimsProfile(in_tensor, index);
  } else {
    input_dims = ParseInputDimsProfile(in_tensor, index);
  }
  MS_LOG(INFO) << "add network input: " << in_tensor.Name();
  return ctx_->network()->addInput(in_tensor.Name().c_str(), cuda_dtype, input_dims);
}

nvinfer1::Dims TensorRTSubGraph::SetInputDimsProfile(const TensorInfo &in_tensor, int index) {
  auto input_info = trt_profile_config_.input_infos[index];
  auto input_dims = ConvertCudaDims(input_info.input_shape);
  DebugDims("input dims", input_dims);
  for (size_t i = 0; i < trt_profile_config_.profiles.size(); i++) {
    auto &profile = trt_profile_config_.profiles[i];
    auto min_dims = ConvertCudaDims(profile.inputs[index].min_dims);
    if (!profiles_[i]->setDimensions(input_info.name.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims)) {
      MS_LOG(ERROR) << "setDimensions of kMIN failed for " << input_info.name;
      return input_dims;
    }
    auto opt_dims = ConvertCudaDims(profile.inputs[index].opt_dims);
    if (!profiles_[i]->setDimensions(input_info.name.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims)) {
      MS_LOG(ERROR) << "setDimensions of kOPT failed for " << input_info.name;
      return input_dims;
    }

    auto max_dims = ConvertCudaDims(profile.inputs[index].max_dims);
    if (!profiles_[i]->setDimensions(input_info.name.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims)) {
      MS_LOG(ERROR) << "setDimensions of kMAX failed for " << input_info.name;
      return input_dims;
    }
    DebugDims("min dims", min_dims);
    DebugDims("opt dims", opt_dims);
    DebugDims("max dims", max_dims);
  }
  return input_dims;
}

nvinfer1::Dims TensorRTSubGraph::ParseInputDimsProfile(const TensorInfo &in_tensor, int index) {
  nvinfer1::Dims input_dims = ConvertCudaDims(in_tensor.Shape());
  nvinfer1::Dims input_dims_min = ConvertCudaDims(in_tensor.Shape());
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
  if (trt_profile_config_.profiles.empty()) {
    ProfileItem profile_item;
    profile_item.inputs.resize(inputs_.size());
    trt_profile_config_.profiles.push_back(profile_item);
  }
  auto &profile_item = trt_profile_config_.profiles.back();
  profile_item.inputs[index].min_dims = ConvertMSShape(input_dims_min);
  profile_item.inputs[index].opt_dims = ConvertMSShape(input_dims_opt);
  profile_item.inputs[index].max_dims = ConvertMSShape(input_dims_max);

  DebugDims("input min dims", input_dims_min);
  DebugDims("input opt dims", input_dims_opt);
  DebugDims("input max dims", input_dims_max);
  return input_dims;
}

int TensorRTSubGraph::ParseInputsProfile() {
  MS_LOG(INFO) << "using serialied engine.";
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto dim = ParseInputDimsProfile(inputs_[i], i);
    if (dim.nbDims <= 0) {
      MS_LOG(ERROR) << "input dims is invalid.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int TensorRTSubGraph::GetInputIndexByName(const std::string &name) {
  for (size_t i = 0; i != inputs().size(); ++i) {
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
    for (size_t i = 0; i != cur_op->inputs().size(); ++i) {
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
        auto weight_handled_inner =
          cur_op->IsWeightInputHanledInner() ||
          trt_specific_weight_handled_inner_.find(cur_op->type()) != trt_specific_weight_handled_inner_.end();
        if (!weight_handled_inner) {
          if (!in_tensor.IsConst()) {
            MS_LOG(ERROR) << "Weight Tensor data is not const.";
            return RET_ERROR;
          }
          trt_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx_, in_tensor, cur_op->GetOpName());
          trt_tensor.format_ = Format::NCHW;
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

int TensorRTSubGraph::MarkOutputs() {
  // Mark NetWork Output Tensor.
  for (const auto &out_tensor : outputs_) {
    std::string output_name = out_tensor.Name();
    auto input_it = std::find_if(inputs_.begin(), inputs_.end(),
                                 [=](const TensorInfo &input) { return input.Name() == output_name; });
    if (input_it != inputs_.end()) {
      nvinfer1::ITensor *trt_tensor = SetTensorRTNetworkInput(*input_it, GetInputIndexByName(input_it->Name()));
      ctx_->network()->markOutput(*trt_tensor);
      continue;
    }
    if (out_tensor.IsConst()) {
      MS_LOG(INFO) << "markOutput for: " << out_tensor.Name();
      auto output_helper = ctx_->MsName2Tensor(out_tensor.Name());
      if (output_helper.trt_tensor_ == nullptr) {
        output_helper.trt_tensor_ = lite::ConvertConstantTensor(ctx_, out_tensor, out_tensor.Name());
        output_helper.format_ = Format::NCHW;
        MS_LOG(INFO) << "auto convert constant tensor for: " << out_tensor.Name();
        ctx_->RegisterTensor(output_helper, out_tensor.Name());
      }
      nvinfer1::ITensor *out_trt_tensor = output_helper.trt_tensor_;
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
    for (auto out_op : this->out_ops_) {
      for (size_t index = 0; index < out_op->outputs().size(); index++) {
        if (out_op->outputs()[index] == out_tensor) {
          MS_LOG(INFO) << "markOutput for: " << out_tensor.Name();
          auto output_helper = out_op->output(ctx_, index);
          nvinfer1::ITensor *out_trt_tensor = output_helper.trt_tensor_;
          out_trt_tensor->setName(("__" + out_tensor.Name()).c_str());
          auto out_layer = ctx_->network()->addIdentity(*out_trt_tensor);
          if (out_tensor.DataType() == DataType::kNumberTypeFloat16) {
            MS_LOG(WARNING) << "Cast output tensor " << out_tensor.Name() << " to fp16";
            out_layer->setOutputType(0, nvinfer1::DataType::kHALF);
          }
          out_trt_tensor = out_layer->getOutput(0);
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

#ifdef PROFILER_
  auto profiler = new SimpleProfiler("myprofiler");
  if (profiler == nullptr) {
    MS_LOG(WARNING) << "Cannot create profiler";
  }
  this->trt_context_->setProfiler(profiler);
#endif

  int binding_num = this->engine_->getNbBindings();
  if (binding_num <= 0) {
    MS_LOG(ERROR) << "TensorRTSubGraph binding num < 0.";
    return RET_ERROR;
  }
  tensor_bindings_ = new (std::nothrow) void *[binding_num];
  if (tensor_bindings_ == nullptr) {
    MS_LOG(ERROR) << "malloc tensor binding array failed.";
    return RET_ERROR;
  }
  profile_index_ = MaxVolumnProfileIndex();
  if (this->trt_context_->setOptimizationProfile(profile_index_)) {
    MS_LOG(INFO) << "setOptimizationProfile: " << profile_index_;
  }
  const auto &profile = trt_profile_config_.profiles[profile_index_];
  for (size_t i = 0; i != inputs_.size(); ++i) {
    auto &tensor = inputs_[i];
    auto max_profile_dims = profile.inputs[i].max_dims;
    tensor.SetShape(max_profile_dims);
    int volumn = std::accumulate(max_profile_dims.begin(), max_profile_dims.end(), 1, std::multiplies<int>());
    auto type_size = lite::DataTypeSize(static_cast<enum TypeId>(tensor.DataType()));
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor, volumn * type_size);
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "malloc for inputs tensor device memory failed.";
      return RET_ERROR;
    }
    auto tensor_name = tensor.Name();
    trt_in_tensor_name_.push_back(tensor_name);
    int index = GetProfileBindingIndex(tensor_name, profile_index_);
    MS_LOG(INFO) << "device index " << index << " for tensor : " << tensor_name << " attr: " << device_ptr;
    tensor_bindings_[index] = device_ptr;
    nvinfer1::Dims input_dims = ConvertCudaDims(profile.inputs[i].max_dims);
    if (!this->trt_context_->setBindingDimensions(index, input_dims)) {
      MS_LOG(ERROR) << "invalid input dims of " << tensor.Name();
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
  for (auto &tensor : outputs_) {
    int max_index = GetProfileBindingIndex(tensor.Name(), profile_index_);
    auto out_dims = trt_context_->getBindingDimensions(max_index);
    int elem_num = std::accumulate(out_dims.d, out_dims.d + out_dims.nbDims, 1, std::multiplies<int>());
    DebugDims("out dims", out_dims);
    MS_LOG(INFO) << "Set output shape by tensorrt binding output";
    tensor.SetShape(lite::ConvertMSShape(out_dims));
    auto type_size = lite::DataTypeSize(static_cast<enum TypeId>(tensor.DataType()));
    if (tensor.DataType() == DataType::kNumberTypeBool) {
      type_size = lite::DataTypeSize(static_cast<enum TypeId>(DataType::kNumberTypeInt32));
    }
    auto device_ptr = runtime_->GetAllocator()->MallocDeviceMem(tensor, elem_num * type_size);
    if (device_ptr == nullptr) {
      MS_LOG(ERROR) << "malloc for outputs tensor device memory failed.";
      return RET_ERROR;
    }
    for (size_t j = 0; j != profiles_.size(); ++j) {
      int index = GetProfileBindingIndex(tensor.Name(), j);
      tensor_bindings_[index] = device_ptr;
    }
    trt_out_tensor_name_.push_back(tensor.Name());
  }
  return RET_OK;
}

int TensorRTSubGraph::SelectProfile(const std::vector<ShapeVector> &new_shapes) const {
  std::vector<int> profile_index;
  for (size_t i = 0; i < profiles_.size(); ++i) {
    const auto &profile = trt_profile_config_.profiles[i];
    bool condition = true;
    for (size_t j = 0; j < trt_in_tensor_name_.size(); ++j) {
      auto new_shape = new_shapes[j];
      auto profile_input = profile.inputs[j];
      if (new_shape.size() != profile_input.max_dims.size()) {
        condition = false;
      } else {
        for (size_t od = 0; od < new_shape.size(); od++) {
          if (new_shape[od] < profile_input.min_dims[od] || new_shape[od] > profile_input.max_dims[od]) {
            condition = false;
            break;
          }
        }
      }
    }
    if (condition) {
      profile_index.push_back(i);
    }
  }
  return profile_index.empty() ? -1 : profile_index.front();
}

size_t TensorRTSubGraph::MaxVolumnProfileIndex() const {
  int max_volumn = std::numeric_limits<int>::min();
  size_t max_volumn_index = 0;
  for (size_t i = 0; i < trt_profile_config_.profiles.size(); ++i) {
    const auto &profile = trt_profile_config_.profiles[i];
    // depend on the first input tensor
    int64_t volumn = std::accumulate(profile.inputs[0].max_dims.begin(), profile.inputs[0].max_dims.end(), 1,
                                     std::multiplies<int64_t>());
    if (volumn > max_volumn) {
      max_volumn_index = i;
      max_volumn = volumn;
    }
  }
  return max_volumn_index;
}

int TensorRTSubGraph::GetProfileBindingIndex(const std::string &name, size_t profile_index) {
  std::string binding_name = name;
  if (profile_index != 0) {
    binding_name += " [profile " + std::to_string(profile_index) + "]";
  }
  return this->engine_->getBindingIndex(binding_name.c_str());
}

int TensorRTSubGraph::OnNewInputShapes(const std::vector<ShapeVector> &new_shapes) {
  if (inputs_.size() != new_shapes.size()) {
    MS_LOG(ERROR) << "Graph inputs size " << inputs_.size() << " != resize input size " << new_shapes.size();
    return RET_ERROR;
  }
  auto select_profile_index = SelectProfile(new_shapes);
  if (select_profile_index < 0) {
    MS_LOG(ERROR) << "Not support input shape " << new_shapes;
    return RET_ERROR;
  }
  profile_index_ = static_cast<size_t>(select_profile_index);
  if (this->trt_context_->setOptimizationProfile(profile_index_)) {
    MS_LOG(INFO) << "setOptimizationProfile: " << profile_index_;
  }
  int batch_size = -1;
  for (size_t i = 0; i < trt_in_tensor_name_.size(); i++) {
    if (inputs_[i].Shape() == new_shapes[i]) {
      continue;
    }
    if (input_batchsize_index_ == -1) {
      MS_LOG(ERROR) << "current network don't support resize.";
      return RET_ERROR;
    }
    inputs_[i].SetShape(new_shapes[i]);
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

    MS_LOG(INFO) << "resize at input_batch_index " << input_batchsize_index_ << ", update batch size to "
                 << inputs_[i].Shape()[input_batchsize_index_];
    int new_batch_size = inputs_[i].Shape()[input_batchsize_index_];
    if (batch_size != -1 && batch_size != new_batch_size) {
      MS_LOG(ERROR) << "Batch size " << batch_size << " of input 0 != batch size " << new_batch_size << " of input "
                    << i;
      return RET_ERROR;
    }
    batch_size = new_batch_size;

    int index = GetProfileBindingIndex(trt_in_tensor_name_[i], profile_index_);
    // Set actual input size
    nvinfer1::Dims input_dims = ConvertCudaDims(inputs_[i].Shape());
    for (int od = 0; od < input_dims.nbDims; od++) {
      MS_LOG(DEBUG) << "in tensor " << trt_in_tensor_name_[i] << " dims at " << od << " is " << input_dims.d[od];
    }

    if (!this->trt_context_->setBindingDimensions(index, input_dims)) {
      MS_LOG(ERROR) << "invalid input dims of " << inputs_[i].Name() << ", profile index: " << profile_index_
                    << ", dst dims: " << CudaDimsAsString(input_dims);
      return RET_ERROR;
    }
  }
  if (!this->trt_context_->allInputDimensionsSpecified()) {
    MS_LOG(ERROR) << "input dims need to be specified.";
    return RET_ERROR;
  }
  if (batch_size != -1) {
    for (size_t i = 0; i < trt_out_tensor_name_.size(); i++) {
      auto index = GetProfileBindingIndex(trt_out_tensor_name_[i], profile_index_);
      auto out_dims = trt_context_->getBindingDimensions(index);
      DebugDims("out dims", out_dims);
      auto new_shape = lite::ConvertMSShape(out_dims);
      MS_LOG(INFO) << "Set output shape of " << trt_out_tensor_name_[i] << " to " << new_shape
                   << "  by tensorrt binding output";
      outputs_[i].SetShape(new_shape);
    }
  }
  return RET_OK;
}

int TensorRTSubGraph::PreExecute(const std::vector<tensor::Tensor> &inputs, const std::vector<tensor::Tensor> &outputs,
                                 bool sync) {
  if (inputs_.size() != inputs.size()) {
    MS_LOG(ERROR) << "Graph inputs size " << inputs_.size() << " != execute inputs size " << inputs.size();
    return RET_ERROR;
  }
  if (!outputs.empty() && outputs.size() != outputs_.size()) {
    MS_LOG(ERROR) << "Graph outputs size " << outputs_.size() << " != execute outputs size " << outputs.size();
    return RET_ERROR;
  }
  std::vector<ShapeVector> new_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(new_shapes), [](auto &t) { return t.shape_c(); });
  auto ret = OnNewInputShapes(new_shapes);
  if (ret != RET_OK) {
    return ret;
  }
  for (size_t i = 0; i < trt_in_tensor_name_.size(); i++) {
    auto trt_tensor_name = trt_in_tensor_name_[i];
    void *device_ptr = nullptr;
    auto input_device_address = inputs[i].device_address();
    if (input_device_address != nullptr && input_device_address->GetMutablePtr() != nullptr) {
      device_ptr = input_device_address->GetMutablePtr();
    } else {
      device_ptr = runtime_->GetAllocator()->MallocDeviceMem(trt_tensor_name, inputs_[i].DataSize(),
                                                             ConvertDataType(inputs_[i].DataType()));
      if (device_ptr == nullptr) {
        MS_LOG(ERROR) << "realloc for input tensor device memory failed.";
        return RET_ERROR;
      }
      ret = runtime_->GetAllocator()->SyncMemHostToDevice(inputs[i], trt_tensor_name, sync);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "sync mem from host to device failed for " << trt_tensor_name;
        return RET_ERROR;
      }
      runtime_->GetAllocator()->MarkMemValid(trt_tensor_name, true);
    }
    int index = GetProfileBindingIndex(trt_tensor_name, profile_index_);
    MS_LOG(INFO) << "device index " << index << " for tensor : " << trt_tensor_name << " attr: " << device_ptr;
    tensor_bindings_[index] = device_ptr;
  }
  for (size_t i = 0; i < trt_out_tensor_name_.size(); i++) {
    const auto &trt_out_tensor_name = trt_out_tensor_name_[i];
    int index = GetProfileBindingIndex(trt_out_tensor_name, profile_index_);
    void *device_ptr = nullptr;
    if (outputs.size() > i) {
      auto &output = outputs[i];
      if (output.device_address() && output.device_address()->GetMutablePtr()) {
        if (output.Size() < outputs_[i].DataSize()) {
          MS_LOG(ERROR) << "Specified output device data size " << output.Size()
                        << " cannot less than execute output data size " << outputs_[i].DataSize()
                        << ", output shape: " << outputs_[i].Shape();
          return RET_ERROR;
        }
        device_ptr = output.device_address()->GetMutablePtr();
      }
    }
    if (!device_ptr) {
      device_ptr = runtime_->GetAllocator()->MallocDeviceMem(trt_out_tensor_name, outputs_[i].DataSize(),
                                                             ConvertDataType(outputs_[i].DataType()));
      if (device_ptr == nullptr) {
        MS_LOG(ERROR) << "realloc for outputs tensor device memory failed.";
        return RET_ERROR;
      }
    }
    tensor_bindings_[index] = device_ptr;
  }
  return RET_OK;
}

int TensorRTSubGraph::PostExecute(std::vector<tensor::Tensor> *outputs, bool sync) {
  if (!outputs->empty() && outputs->size() != outputs_.size()) {
    MS_LOG(ERROR) << "Graph outputs size " << outputs_.size() << " != execute outputs size " << outputs->size();
    return RET_ERROR;
  }
  auto has_outputs = !outputs->empty();
  for (size_t i = 0; i < trt_out_tensor_name_.size(); i++) {
    const auto &trt_out_tensor_name = trt_out_tensor_name_[i];
    auto index = GetProfileBindingIndex(trt_out_tensor_name, profile_index_);
    // actual output tensor dims
    auto out_dims = this->trt_context_->getBindingDimensions(index);
    std::vector<int64_t> new_shape = lite::ConvertMSShape(out_dims);
    for (int od = 0; od < out_dims.nbDims; od++) {
      MS_LOG(DEBUG) << "out tensor " << trt_out_tensor_name << " dims at " << od << " is " << new_shape[od];
    }
    runtime_->GetAllocator()->MarkMemValid(trt_out_tensor_name, true);
    if (has_outputs) {
      auto &tensor = outputs->at(i);
      auto dst_device = tensor.device_address();
      if (dst_device == nullptr || dst_device->GetMutablePtr() == nullptr) {
        if (tensor.Size() < outputs_[i].DataSize()) {
          MS_LOG(ERROR) << "Specified output host data size " << tensor.Size()
                        << " cannot less than execute output data size " << outputs_[i].DataSize()
                        << ", output shape: " << new_shape;
          return RET_ERROR;
        }
        auto host_address = tensor.data_c();
        if (host_address == nullptr) {
          MS_LOG(ERROR) << "Specified output device or host address cannot be nullptr";
          return RET_ERROR;
        }
        int sync_ret = runtime_->GetAllocator()->SyncMemDeviceToHost(host_address, outputs_[i].DataSize(),
                                                                     trt_out_tensor_name, sync);
        if (sync_ret != RET_OK) {
          MS_LOG(ERROR) << "sync mem from device to host failed for " << trt_out_tensor_name;
          return sync_ret;
        }
      }
    } else {
      tensor::Tensor output_tensor(static_cast<enum TypeId>(outputs_[i].DataType()), new_shape);
      int sync_ret = runtime_->GetAllocator()->SyncMemDeviceToHost(&output_tensor, trt_out_tensor_name, sync);
      if (sync_ret != RET_OK) {
        MS_LOG(ERROR) << "sync mem from device to host failed for " << trt_out_tensor_name;
        return sync_ret;
      }
      outputs->push_back(output_tensor);
    }
    runtime_->GetAllocator()->MarkMemValid(trt_out_tensor_name, false);
  }
  // make mem invalid, prepare for next execute
  for (size_t i = 0; i < inputs_.size(); i++) {
    runtime_->GetAllocator()->MarkMemValid(trt_in_tensor_name_[i], false);
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

int TensorRTSubGraph::Execute(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
#ifdef ASYNC_INFER
  bool sync = false;
#else
  bool sync = true;
#endif
  int ret = lite::SetCudaDevice(device_info_);
  if (ret != RET_OK) {
    return ret;
  }
  ret = PreExecute(inputs, *outputs, sync);
  if (ret != RET_OK) {
    return ret;
  }
  if (sync) {
    if (!this->trt_context_->executeV2(tensor_bindings_)) {
      MS_LOG(ERROR) << "TensorRT execute failed.";
      return RET_ERROR;
    }
  } else {
    if (!this->trt_context_->enqueueV2(tensor_bindings_, stream_, nullptr)) {
      MS_LOG(ERROR) << "TensorRT execute failed.";
      return RET_ERROR;
    }
  }
  ret = PostExecute(outputs, sync);
  if (ret != RET_OK) {
    return ret;
  }
  if (!sync) {
    cudaStreamSynchronize(stream_);
  }
  return RET_OK;
}

int TensorRTSubGraph::Resize(const std::vector<tensor::Tensor> &, const std::vector<ShapeVector> &new_shapes) {
  return OnNewInputShapes(new_shapes);
}

ITensorHelper TensorRTSubGraph::FindTensorRTInputs(TensorRTOp *cur_op, const TensorInfo &in_tensor) {
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
}  // namespace mindspore::lite
