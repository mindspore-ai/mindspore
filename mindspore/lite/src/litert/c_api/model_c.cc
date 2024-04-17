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
#include "include/c_api/model_c.h"
#include <vector>
#include <cstdint>
#include "include/api/context.h"
#include "include/api/serialization.h"
#include "include/api/types.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"

namespace mindspore {
class ModelC {
 public:
  ModelC() : model_(nullptr) {}
  ~ModelC() {
    for (auto in : inputs_) {
      if (in != nullptr) {
        delete in;
      }
    }
    for (auto out : outputs_) {
      if (out != nullptr) {
        delete out;
      }
    }
    for (auto out : outputs_train_) {
      if (out != nullptr) {
        delete out;
      }
    }
  }

  MSTensor **GetInputs(size_t *input_num);
  MSTensor **GetOutputs(size_t *output_num);
  mindspore::MSKernelCallBack TransCallBack(const MSKernelCallBackC &ms_callback);
  std::shared_ptr<Model> model_;
  std::shared_ptr<Context> context_;

 private:
  Status RunGraph(const MSKernelCallBackC &before, const MSKernelCallBackC &after);
  void ResetTensorData(std::vector<void *> old_data, std::vector<lite::Tensor *> tensors);
  LiteTensorImpl *TensorToTensorImpl(mindspore::lite::Tensor *tensor);

 private:
  MSTensor **GetOutputsTensor(size_t *output_num, std::vector<MSTensor *> *vec_tensors);
  std::vector<MSTensor *> inputs_;
  std::vector<MSTensor *> outputs_;
  std::vector<MSTensor *> outputs_train_;
};

MSTensor **ModelC::GetInputs(size_t *input_num) {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model_ is nullptr.";
    return nullptr;
  }
  if (!inputs_.empty()) {
    *input_num = inputs_.size();
    return inputs_.data();
  }
  auto inputs = model_->GetInputs();
  *input_num = inputs.size();
  inputs_.resize(inputs.size(), nullptr);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs_[i] = new (std::nothrow) MSTensor(inputs[i].impl());
    if (inputs_[i] == nullptr) {
      inputs_.clear();
      return nullptr;
    }
  }
  return inputs_.data();
}

MSTensor **ModelC::GetOutputs(size_t *output_num) {
  if (model_->GetTrainMode() == true) {
    return GetOutputsTensor(output_num, &outputs_train_);
  } else {
    return GetOutputsTensor(output_num, &outputs_);
  }
}

MSTensor **ModelC::GetOutputsTensor(size_t *output_num, std::vector<MSTensor *> *vec_tensors) {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "model_ is nullptr.";
    return nullptr;
  }
  if (!vec_tensors->empty()) {
    *output_num = vec_tensors->size();
    return vec_tensors->data();
  }

  auto outputs = model_->GetOutputs();
  *output_num = outputs.size();
  vec_tensors->resize(outputs.size(), nullptr);
  for (size_t i = 0; i < outputs.size(); i++) {
    (*vec_tensors)[i] = new (std::nothrow) MSTensor(outputs[i].impl());
    if ((*vec_tensors)[i] == nullptr) {
      vec_tensors->clear();
      return nullptr;
    }
  }
  return vec_tensors->data();
}

mindspore::MSKernelCallBack ModelC::TransCallBack(const MSKernelCallBackC &ms_callback) {
  mindspore::MSKernelCallBack call_back = nullptr;
  if (ms_callback != nullptr) {
    call_back = [&](const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
                    const mindspore::MSCallBackParam &opInfo) {
      std::vector<MSTensorHandle> vec_inputs;
      std::vector<MSTensorHandle> vec_outputs;
      MSCallBackParamC call_back = {const_cast<char *>(opInfo.node_name.c_str()),
                                    const_cast<char *>(opInfo.node_type.c_str())};
      size_t inputs_handle_num = inputs.size();
      for (size_t i = 0; i < inputs_handle_num; i++) {
        vec_inputs.push_back(static_cast<MSTensorHandle>(&(static_cast<std::vector<mindspore::MSTensor>>(inputs)[i])));
      }
      size_t outputs_handle_num = outputs.size();
      for (size_t i = 0; i < outputs_handle_num; i++) {
        vec_outputs.push_back(
          static_cast<MSTensorHandle>(&(static_cast<std::vector<mindspore::MSTensor>>(outputs)[i])));
      }
      MSTensorHandleArray handle_inputs = {inputs_handle_num, vec_inputs.data()};
      MSTensorHandleArray handle_outputs = {outputs_handle_num, vec_outputs.data()};
      return ms_callback(handle_inputs, handle_outputs, call_back);
    };
  }
  return call_back;
}
}  // namespace mindspore

MSModelHandle MSModelCreate() {
  auto impl = new (std::nothrow) mindspore::ModelC();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Model implement is nullptr.";
    return nullptr;
  }
  impl->model_ = std::make_shared<mindspore::Model>();
  if (impl->model_ == nullptr) {
    MS_LOG(ERROR) << "model_ is nullptr.";
    delete impl;
    return nullptr;
  }
  return static_cast<MSModelHandle>(impl);
}

void MSModelDestroy(MSModelHandle *model) {
  if (model != nullptr && *model != nullptr) {
    auto impl = static_cast<mindspore::ModelC *>(*model);
    delete impl;
    *model = nullptr;
  }
}

void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return;
}

size_t MSModelCalcWorkspaceSize(MSModelHandle model) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0;
}

MSStatus MSModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,
                      const MSContextHandle model_context) {
  if (model == nullptr || model_data == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model/model_data/model_context is nullptr.";
    return kMSStatusLiteNullptr;
  }
  if (model_type == kMSModelTypeInvalid) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return kMSStatusLiteParamInvalid;
  }
  mindspore::Context *context = static_cast<mindspore::Context *>(model_context);
  auto impl = static_cast<mindspore::ModelC *>(model);
  if (impl->context_.get() != context) {
    impl->context_.reset(context);
  }
  auto ret = impl->model_->Build(model_data, data_size, static_cast<mindspore::ModelType>(model_type), impl->context_);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,
                              const MSContextHandle model_context) {
  if (model == nullptr || model_path == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model/model_path/model_context is nullptr.";
    return kMSStatusLiteNullptr;
  }
  if (model_type == kMSModelTypeInvalid) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return kMSStatusLiteParamInvalid;
  }
  mindspore::Context *context = static_cast<mindspore::Context *>(model_context);
  auto impl = static_cast<mindspore::ModelC *>(model);
  if (impl->context_.get() != context) {
    impl->context_.reset(context);
  }
  auto ret = impl->model_->Build(model_path, static_cast<mindspore::ModelType>(model_type), impl->context_);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos,
                       size_t shape_info_num) {
  if (model == nullptr || shape_infos == nullptr) {
    MS_LOG(ERROR) << "model/shape_infos is nullptr.";
    return kMSStatusLiteNullptr;
  }
  std::vector<mindspore::MSTensor> vec_inputs;
  for (size_t i = 0; i < inputs.handle_num; ++i) {
    vec_inputs.push_back(*static_cast<mindspore::MSTensor *>(inputs.handle_list[i]));
  }

  std::vector<std::vector<int64_t>> vec_dims;
  for (size_t i = 0; i < shape_info_num; i++) {
    std::vector<int64_t> shape(shape_infos[i].shape, shape_infos[i].shape + shape_infos[i].shape_num);
    if (std::any_of(shape.begin(), shape.end(), [](int64_t val) { return val < 0 || val > INT32_MAX; })) {
      MS_LOG(ERROR) << "Invalid shape: " << shape << ", each dimension must be in [0, INT32_MAX]";
      return kMSStatusLiteInputParamInvalid;
    }
    vec_dims.push_back(shape);
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->Resize(vec_inputs, vec_dims);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                        const MSKernelCallBackC before, const MSKernelCallBackC after) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteNullptr;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t input_num;
  (void)impl->GetInputs(&input_num);
  if (input_num != inputs.handle_num) {
    MS_LOG(ERROR) << "Wrong input size.";
    return kMSStatusLiteError;
  }

  std::vector<mindspore::MSTensor> ms_tensor_inputs;
  for (size_t i = 0; i < inputs.handle_num; i++) {
    if (inputs.handle_list[i] != nullptr) {
      auto user_input = static_cast<mindspore::MSTensor *>(inputs.handle_list[i]);
      ms_tensor_inputs.push_back(*user_input);
    } else {
      MS_LOG(ERROR) << "input handle is nullptr.";
      return kMSStatusLiteNullptr;
    }
  }

  mindspore::MSKernelCallBack before_call_back = impl->TransCallBack(before);
  mindspore::MSKernelCallBack after_call_back = impl->TransCallBack(after);
  std::vector<mindspore::MSTensor> ms_tensor_outputs;
  auto ret = impl->model_->Predict(ms_tensor_inputs, &ms_tensor_outputs, before_call_back, after_call_back);
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "Predict fail, ret :" << ret;
  }
  outputs->handle_list = reinterpret_cast<MSTensorHandle *>(impl->GetOutputs(&(outputs->handle_num)));
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMSStatusLiteNotSupport;
}

MSStatus MSModelExportWeight(const MSModelHandle model, const char *export_path) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMSStatusLiteNotSupport;
}

MSTensorHandleArray MSModelGetInputs(const MSModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t input_num = 0;
  auto handles = reinterpret_cast<MSTensorHandle *>(impl->GetInputs(&input_num));
  return {input_num, handles};
}

MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t output_num;
  auto handles = reinterpret_cast<MSTensorHandle *>(impl->GetOutputs(&output_num));
  return {output_num, handles};
}

MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name) {
  if (model == nullptr || tensor_name == nullptr) {
    MS_LOG(ERROR) << "model/tensor_name is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t input_num;
  auto inputs = impl->GetInputs(&input_num);
  for (size_t i = 0; i < input_num; i++) {
    if (inputs[i]->Name() == tensor_name) {
      return static_cast<MSTensorHandle>(inputs[i]);
    }
  }
  MS_LOG(ERROR) << "tensor is not exist.";
  return nullptr;
}

MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char *tensor_name) {
  if (model == nullptr || tensor_name == nullptr) {
    MS_LOG(ERROR) << "model/tensor_name is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t output_num;
  auto outputs = impl->GetOutputs(&output_num);
  for (size_t i = 0; i < output_num; i++) {
    if (outputs[i]->Name() == tensor_name) {
      return static_cast<MSTensorHandle>(outputs[i]);
    }
  }
  MS_LOG(ERROR) << "tensor is not exist.";
  return nullptr;
}

MSTrainCfgHandle MSTrainCfgCreate() {
  auto impl = new (std::nothrow) mindspore::TrainCfg();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "TrainCfg implement is nullptr.";
    return nullptr;
  }
  return static_cast<MSTrainCfgHandle>(impl);
}

void MSTrainCfgDestroy(MSTrainCfgHandle *train_cfg) {
  if (train_cfg != nullptr && *train_cfg != nullptr) {
    auto impl = static_cast<mindspore::TrainCfg *>(*train_cfg);
    delete impl;
    *train_cfg = nullptr;
  }
}

char **MSTrainCfgGetLossName(MSTrainCfgHandle train_cfg, size_t *num) {
  if (train_cfg == nullptr || num == nullptr) {
    MS_LOG(ERROR) << "train_cfg/num is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  auto loss_name = impl->GetLossName();
  *num = loss_name.size();
  char **name = static_cast<char **>(malloc(loss_name.size() * sizeof(char *)));
  if (name == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc loss_name.";
    return nullptr;
  }
  for (size_t i = 0; i < loss_name.size(); i++) {
    name[i] = static_cast<char *>(malloc(loss_name[i].size() + 1));
    memcpy(name[i], loss_name[i].c_str(), loss_name[i].size() + 1);
  }
  return name;
}

void MSTrainCfgSetLossName(MSTrainCfgHandle train_cfg, const char **loss_name, size_t num) {
  if (train_cfg == nullptr || loss_name == nullptr || *loss_name == nullptr) {
    MS_LOG(ERROR) << "train_cfg/loss_name is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  std::vector<std::string> vec_name;
  for (size_t i = 0; i < num; i++) {
    vec_name.push_back(loss_name[i]);
  }
  impl->SetLossName(vec_name);
}

MSOptimizationLevel MSTrainCfgGetOptimizationLevel(MSTrainCfgHandle train_cfg) {
  if (train_cfg == nullptr) {
    MS_LOG(ERROR) << "train_cfg is nullptr, return kMSKO0";
    return kMSKO0;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  return static_cast<MSOptimizationLevel>(impl->optimization_level_);
}

void MSTrainCfgSetOptimizationLevel(MSTrainCfgHandle train_cfg, MSOptimizationLevel level) {
  if (train_cfg == nullptr) {
    MS_LOG(ERROR) << "train_cfg is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::TrainCfg *>(train_cfg);
  impl->optimization_level_ = static_cast<mindspore::OptimizationLevel>(level);
}

MSStatus MSTrainModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,
                           const MSContextHandle model_context, const MSTrainCfgHandle train_cfg) {
  if (model == nullptr || model_data == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model/model_data/model_context is nullptr.";
    return kMSStatusLiteNullptr;
  }
  if (model_type == kMSModelTypeInvalid) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);

  mindspore::Graph graph;
  auto status =
    mindspore::Serialization::Load(model_data, data_size, static_cast<mindspore::ModelType>(model_type), &graph);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "load ms file failed.";
    return kMSStatusLiteError;
  }
  auto context = static_cast<mindspore::Context *>(model_context);
  auto build_train_cfg = static_cast<mindspore::TrainCfg *>(train_cfg);
  if (impl->context_.get() != context) {
    impl->context_.reset(context);
  }
  auto ret = impl->model_->Build(static_cast<mindspore::GraphCell>(graph), impl->context_,
                                 std::shared_ptr<mindspore::TrainCfg>(build_train_cfg));
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Load and compile failed";
  }
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSTrainModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,
                                   const MSContextHandle model_context, const MSTrainCfgHandle train_cfg) {
  if (model == nullptr || model_path == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "model/model_path/model_context is nullptr.";
    return kMSStatusLiteNullptr;
  }
  if (model_type == kMSModelTypeInvalid) {
    MS_LOG(ERROR) << "model_type is invalid.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);

  mindspore::Graph graph;
  auto status = mindspore::Serialization::Load(model_path, static_cast<mindspore::ModelType>(model_type), &graph);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "load ms file failed. " << model_path;
    return kMSStatusLiteError;
  }
  auto context = static_cast<mindspore::Context *>(model_context);
  auto build_train_cfg = static_cast<mindspore::TrainCfg *>(train_cfg);
  if (impl->context_.get() != context) {
    impl->context_.reset(context);
  }
  auto ret = impl->model_->Build(static_cast<mindspore::GraphCell>(graph), impl->context_,
                                 std::shared_ptr<mindspore::TrainCfg>(build_train_cfg));
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Load and compile failed";
  }
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelSetLearningRate(MSModelHandle model, float learning_rate) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->SetLearningRate(learning_rate);
  return static_cast<MSStatus>(ret.StatusCode());
}

float MSModelGetLearningRate(MSModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  return impl->model_->GetLearningRate();
}

MSStatus MSRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->RunStep(impl->TransCallBack(before), impl->TransCallBack(after));
  return static_cast<MSStatus>(ret.StatusCode());
}

MSTensorHandleArray MSModelGetWeights(MSModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto features = impl->model_->GetFeatureMaps();
  size_t handle_num = features.size();

  mindspore::MSTensor **handle_list =
    static_cast<mindspore::MSTensor **>(malloc(handle_num * sizeof(mindspore::MSTensor *)));
  if (handle_list == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc handle_list.";
    return {0, nullptr};
  }
  for (size_t i = 0; i < handle_num; i++) {
    handle_list[i] = new (std::nothrow) mindspore::MSTensor(features[i].impl());
  }
  return {handle_num, reinterpret_cast<MSTensorHandle *>(handle_list)};
}

MSStatus MSModelUpdateWeights(MSModelHandle model, const MSTensorHandleArray new_weights) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<mindspore::MSTensor> weights;
  for (size_t i = 0; i < new_weights.handle_num; i++) {
    weights.push_back(*static_cast<mindspore::MSTensor *>(new_weights.handle_list[i]));
  }
  auto ret = impl->model_->UpdateWeights(weights);
  return static_cast<MSStatus>(ret.StatusCode());
}

bool MSModelGetTrainMode(MSModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return false;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  return impl->model_->GetTrainMode();
}

MSStatus MSModelSetTrainMode(MSModelHandle model, bool train) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->SetTrainMode(train);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelSetupVirtualBatch(MSModelHandle model, int virtual_batch_multiplier, float lr, float momentum) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->model_->SetupVirtualBatch(virtual_batch_multiplier, lr, momentum);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSExportModel(MSModelHandle model, MSModelType model_type, const char *model_file,
                       MSQuantizationType quantization_type, bool export_inference_only, char **output_tensor_name,
                       size_t num) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<std::string> tensor_name;
  for (size_t i = 0; i < num; i++) {
    tensor_name.push_back(output_tensor_name[i]);
  }
  auto ret = mindspore::Serialization::ExportModel(
    *(impl->model_.get()), static_cast<mindspore::ModelType>(model_type), model_file,
    static_cast<mindspore::QuantizationType>(quantization_type), export_inference_only, tensor_name);
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "export model fail, ret :" << ret;
  }
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSExportModelBuffer(MSModelHandle model, MSModelType model_type, char **model_data, size_t *data_size,
                             MSQuantizationType quantization_type, bool export_inference_only,
                             char **output_tensor_name, size_t num) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<std::string> tensor_name;
  for (size_t i = 0; i < num; i++) {
    tensor_name.push_back(output_tensor_name[i]);
  }
  mindspore::Buffer buffer;
  auto ret = mindspore::Serialization::ExportModel(*(impl->model_.get()), static_cast<mindspore::ModelType>(model_type),
                                                   &buffer, static_cast<mindspore::QuantizationType>(quantization_type),
                                                   export_inference_only, tensor_name);
  auto data = reinterpret_cast<char *>(buffer.MutableData());
  *model_data = reinterpret_cast<char *>(malloc(buffer.DataSize()));
  if (*model_data == nullptr) {
    MS_LOG(ERROR) << "malloc model_data failed.";
    return kMSStatusLiteNullptr;
  }
  *data_size = buffer.DataSize();
  memcpy(*model_data, data, buffer.DataSize());
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "export model fail, ret :" << ret;
  }
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSExportWeightsCollaborateWithMicro(MSModelHandle model, MSModelType model_type, const char *weight_file,
                                             bool is_inference, bool enable_fp16, char **changeable_weights_name,
                                             size_t num) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "model is nullptr.";
    return kMSStatusLiteParamInvalid;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  std::vector<std::string> weights_name;
  for (size_t i = 0; i < num; i++) {
    weights_name.push_back(changeable_weights_name[i]);
  }
  auto ret = mindspore::Serialization::ExportWeightsCollaborateWithMicro(
    *(impl->model_.get()), static_cast<mindspore::ModelType>(model_type), weight_file, is_inference, enable_fp16,
    weights_name);
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "export model fail, ret :" << ret;
  }
  return static_cast<MSStatus>(ret.StatusCode());
}
