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
#include "include/api/types.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/litert/cxx_api/converters.h"
#include "src/litert/lite_session.h"
#include "src/litert/cpu_info.h"

namespace mindspore {
class ModelC {
 public:
  ModelC() : session_(nullptr), context_(nullptr) {}
  ~ModelC() {
    for (auto &impl : tensor_map_) {
      delete impl.second;
    }
  }

  Status Build(const void *model_data, size_t data_size, ModelType model_type, const ContextC *model_context);
  Status Build(const std::string &model_path, ModelType model_type, const ContextC *model_context);
  Status Resize(const std::vector<LiteTensorImpl *> &inputs, const std::vector<std::vector<int64_t>> &shapes);

  Status Predict(const MSTensorHandle *inputs, size_t input_num, MSTensorHandle **outputs, size_t *output_num,
                 const MSKernelCallBackC &before, const MSKernelCallBackC &after);

  LiteTensorImpl **GetInputs(size_t *input_num);
  LiteTensorImpl **GetOutputs(size_t *output_num);

 private:
  std::shared_ptr<lite::LiteSession> session_ = nullptr;
  std::shared_ptr<const ContextC> context_ = nullptr;
  std::map<mindspore::lite::Tensor *, LiteTensorImpl *> tensor_map_;
  std::vector<LiteTensorImpl *> inputs_;
  std::vector<LiteTensorImpl *> outputs_;
  Status RunGraph(const MSKernelCallBackC &before, const MSKernelCallBackC &after);
  void ResetTensorData(std::vector<void *> old_data, std::vector<lite::Tensor *> tensors);
  LiteTensorImpl *TensorToTensorImpl(mindspore::lite::Tensor *tensor);
};

Status ModelC::Build(const void *model_data, size_t data_size, ModelType model_type, const ContextC *model_context) {
  if (!PlatformInstructionSetSupportCheck()) {
    MS_LOG(ERROR) << "The platform exist don't support's instruction.";
    return kLiteNotSupport;
  }

  context_.reset(model_context);
  session_ = std::make_shared<lite::LiteSession>();
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return kLiteNullptr;
  }
  auto ret = session_->Init(ContextUtils::Convert(model_context));
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    return static_cast<StatusCode>(ret);
  }
  ret = session_->LoadModelAndCompileByBuf(static_cast<const char *>(model_data), model_type, data_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Load and compile failed";
  }
  return static_cast<StatusCode>(ret);
}

Status ModelC::Build(const std::string &model_path, ModelType model_type, const ContextC *model_context) {
  if (!PlatformInstructionSetSupportCheck()) {
    MS_LOG(ERROR) << "The platform exist don't support's instruction.";
    return kLiteNotSupport;
  }
  context_.reset(model_context);
  session_ = std::make_shared<lite::LiteSession>();
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return kLiteNullptr;
  }
  auto ret = session_->Init(ContextUtils::Convert(model_context));
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    return static_cast<StatusCode>(ret);
  }
  ret = session_->LoadModelAndCompileByPath(model_path, model_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Load and compile failed";
  }
  return static_cast<StatusCode>(ret);
}

Status ModelC::Resize(const std::vector<LiteTensorImpl *> &inputs, const std::vector<std::vector<int64_t>> &shapes) {
  std::vector<lite::Tensor *> inner_input;
  size_t input_num = inputs.size();
  for (size_t i = 0; i < input_num; i++) {
    auto input = inputs[i];
    if (input == nullptr || input->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Input tensor is null.";
      return kLiteInputTensorError;
    }
    inner_input.push_back(input->lite_tensor());
  }
  size_t shape_num = shapes.size();
  std::vector<std::vector<int32_t>> inner_shapes(shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    std::transform(shapes[i].begin(), shapes[i].end(), std::back_inserter(inner_shapes[i]),
                   [](int64_t value) { return static_cast<int32_t>(value); });
  }
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session implement is null.";
    return kLiteNullptr;
  }
  auto ret = session_->Resize(inner_input, inner_shapes);
  return static_cast<StatusCode>(ret);
}

void ModelC::ResetTensorData(std::vector<void *> old_data, std::vector<lite::Tensor *> tensors) {
  for (size_t j = 0; j < old_data.size(); j++) {
    tensors.at(j)->set_data(old_data.at(j));
  }
}

Status ModelC::Predict(const MSTensorHandle *inputs, size_t input_num, MSTensorHandle **outputs, size_t *output_num,
                       const MSKernelCallBackC &before, const MSKernelCallBackC &after) {
  if (outputs == nullptr || session_ == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kLiteError;
  }
  auto model_inputs = session_->GetInputs();
  if (model_inputs.size() != input_num) {
    MS_LOG(ERROR) << "Wrong input size.";
    return kLiteError;
  }
  std::vector<void *> old_data;
  for (size_t i = 0; i < input_num; i++) {
    auto real_input = model_inputs[i];
    auto user_input = static_cast<LiteTensorImpl *>(inputs[i]);
    if (user_input->DataType() != static_cast<DataType>(real_input->data_type())) {
      ResetTensorData(old_data, model_inputs);
      MS_LOG(ERROR) << "DataType does not match, input:" << user_input->Name()
                    << ", real:" << real_input->tensor_name();
      return kLiteInputTensorError;
    }
    if (user_input->Data() == nullptr) {
      ResetTensorData(old_data, model_inputs);
      MS_LOG(ERROR) << "Tensor " << user_input->Name() << " has no data.";
      return kLiteInputTensorError;
    }
    old_data.push_back(real_input->data());
    if (real_input->data_type() == kObjectTypeString) {
      std::vector<int32_t> shape;
      std::transform(user_input->Shape().begin(), user_input->Shape().end(), std::back_inserter(shape),
                     [](int64_t value) { return static_cast<int32_t>(value); });
      real_input->set_shape(shape);
      real_input->set_data(user_input->MutableData());
    } else {
      if (user_input->MutableData() != real_input->data()) {
        if (real_input->Size() != user_input->DataSize()) {
          ResetTensorData(old_data, model_inputs);
          MS_LOG(ERROR) << "Tensor " << user_input->Name() << " has wrong data size.";
          return kLiteInputTensorError;
        }
        real_input->set_data(user_input->MutableData());
      }
    }
  }
  auto ret = RunGraph(before, after);
  ResetTensorData(old_data, model_inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed.";
    return ret;
  }

  *outputs = reinterpret_cast<MSTensorHandle *>(GetOutputs(output_num));
  return kSuccess;
}

Status ModelC::RunGraph(const MSKernelCallBackC &before, const MSKernelCallBackC &after) {
  KernelCallBack before_call_back = nullptr;
  KernelCallBack after_call_back = nullptr;
  if (before != nullptr) {
    before_call_back = [&](const std::vector<mindspore::lite::Tensor *> &before_inputs,
                           const std::vector<mindspore::lite::Tensor *> &before_outputs,
                           const MSCallBackParam &call_param) {
      std::vector<LiteTensorImpl> inputs_impl;
      std::vector<LiteTensorImpl> outputs_impl;
      std::vector<MSTensorHandle> op_inputs;
      std::vector<MSTensorHandle> op_outputs;
      size_t op_input_num = before_inputs.size();
      for (size_t i = 0; i < op_input_num; i++) {
        inputs_impl.emplace_back(before_inputs[i]);
        op_inputs.push_back(&(inputs_impl.back()));
      }
      size_t op_output_num = before_outputs.size();
      for (size_t i = 0; i < op_output_num; i++) {
        outputs_impl.emplace_back(before_outputs[i]);
        op_outputs.push_back(&(outputs_impl.back()));
      }
      const MSCallBackParamC op_info = {const_cast<char *>(call_param.node_name.c_str()),
                                        const_cast<char *>(call_param.node_type.c_str())};
      MSTensorHandleArray inputs = {op_input_num, op_inputs.data()};
      MSTensorHandleArray outputs = {op_output_num, op_outputs.data()};
      return before(inputs, outputs, op_info);
    };
  }
  if (after != nullptr) {
    after_call_back = [&](const std::vector<mindspore::lite::Tensor *> &after_inputs,
                          const std::vector<mindspore::lite::Tensor *> &after_outputs,
                          const MSCallBackParam &call_param) {
      std::vector<LiteTensorImpl> inputs_impl;
      std::vector<LiteTensorImpl> outputs_impl;
      std::vector<MSTensorHandle> op_inputs;
      std::vector<MSTensorHandle> op_outputs;
      size_t op_input_num = after_inputs.size();
      for (size_t i = 0; i < op_input_num; i++) {
        inputs_impl.emplace_back(after_inputs[i]);
        op_inputs.push_back(&(inputs_impl.back()));
      }
      size_t op_output_num = after_outputs.size();
      for (size_t i = 0; i < op_output_num; i++) {
        outputs_impl.emplace_back(after_outputs[i]);
        op_outputs.push_back(&(outputs_impl.back()));
      }
      const MSCallBackParamC op_info = {const_cast<char *>(call_param.node_name.c_str()),
                                        const_cast<char *>(call_param.node_type.c_str())};
      MSTensorHandleArray inputs = {op_input_num, op_inputs.data()};
      MSTensorHandleArray outputs = {op_output_num, op_outputs.data()};
      return after(inputs, outputs, op_info);
    };
  }
  auto ret = session_->RunGraph(before_call_back, after_call_back);
  return static_cast<StatusCode>(ret);
}

LiteTensorImpl *ModelC::TensorToTensorImpl(mindspore::lite::Tensor *tensor) {
  LiteTensorImpl *impl = nullptr;
  auto iter = tensor_map_.find(tensor);
  if (iter != tensor_map_.end()) {
    impl = iter->second;
  } else {
    impl = new (std::nothrow) LiteTensorImpl(tensor);
    if (impl == nullptr || impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return nullptr;
    }
    tensor_map_[tensor] = impl;
  }
  return impl;
}

LiteTensorImpl **ModelC::GetInputs(size_t *input_num) {
  if (session_ == nullptr || input_num == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return nullptr;
  }
  auto inputs = session_->GetInputs();
  *input_num = inputs.size();
  if (inputs_.capacity() < *input_num) {
    inputs_.reserve(*input_num);
  }
  inputs_.clear();
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_),
                 [&](lite::Tensor *input) { return TensorToTensorImpl(input); });
  return inputs_.data();
}

LiteTensorImpl **ModelC::GetOutputs(size_t *output_num) {
  if (session_ == nullptr || output_num == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return nullptr;
  }
  auto outputs = session_->GetOutputs();
  *output_num = outputs.size();
  if (outputs_.capacity() < *output_num) {
    outputs_.reserve(*output_num);
  }
  outputs_.clear();
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(outputs_),
                 [&](std::unordered_map<std::string, mindspore::lite::Tensor *>::value_type iter) {
                   return TensorToTensorImpl(iter.second);
                 });
  return outputs_.data();
}
}  // namespace mindspore

MSModelHandle MSModelCreate() {
  auto impl = new (std::nothrow) mindspore::ModelC();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
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
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSStatusLiteNullptr;
  }
  if (model_type == kMSModelTypeInvalid) {
    MS_LOG(ERROR) << "param is invalid.";
    return kMSStatusLiteParamInvalid;
  }
  mindspore::ContextC *context = static_cast<mindspore::ContextC *>(model_context);
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->Build(model_data, data_size, static_cast<mindspore::ModelType>(model_type), context);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,
                              const MSContextHandle model_context) {
  if (model == nullptr || model_path == nullptr || model_context == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSStatusLiteNullptr;
  }
  if (model_type == kMSModelTypeInvalid) {
    MS_LOG(ERROR) << "param is invalid.";
    return kMSStatusLiteParamInvalid;
  }
  mindspore::ContextC *context = static_cast<mindspore::ContextC *>(model_context);
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->Build(model_path, static_cast<mindspore::ModelType>(model_type), context);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos,
                       size_t shape_info_num) {
  if (model == nullptr || shape_infos == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSStatusLiteNullptr;
  }
  std::vector<mindspore::LiteTensorImpl *> vec_inputs;
  std::transform(inputs.handle_list, inputs.handle_list + inputs.handle_num, std::back_inserter(vec_inputs),
                 [](MSTensorHandle value) { return static_cast<mindspore::LiteTensorImpl *>(value); });
  std::vector<std::vector<int64_t>> vec_dims;
  for (size_t i = 0; i < shape_info_num; i++) {
    std::vector<int64_t> shape(shape_infos[i].shape, shape_infos[i].shape + shape_infos[i].shape_num);
    vec_dims.push_back(shape);
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->Resize(vec_inputs, vec_dims);
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                        const MSKernelCallBackC before, const MSKernelCallBackC after) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSStatusLiteNullptr;
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  auto ret = impl->Predict(inputs.handle_list, inputs.handle_num, &(outputs->handle_list), &(outputs->handle_num),
                           before, after);
  if (!ret.IsOk()) {
    MS_LOG(ERROR) << "Predict fail, ret :" << ret;
  }
  return static_cast<MSStatus>(ret.StatusCode());
}

MSStatus MSModelRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMSStatusLiteNotSupport;
}

MSStatus MSModelSetTrainMode(const MSModelHandle model, bool train) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMSStatusLiteNotSupport;
}

MSStatus MSModelExportWeight(const MSModelHandle model, const char *export_path) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMSStatusLiteNotSupport;
}

MSTensorHandleArray MSModelGetInputs(const MSModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t input_num;
  auto handles = reinterpret_cast<MSTensorHandle *>(impl->GetInputs(&input_num));
  return {input_num, handles};
}

MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model) {
  if (model == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return {0, nullptr};
  }
  auto impl = static_cast<mindspore::ModelC *>(model);
  size_t output_num;
  auto handles = reinterpret_cast<MSTensorHandle *>(impl->GetOutputs(&output_num));
  return {output_num, handles};
}

MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name) {
  if (model == nullptr || tensor_name == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
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
    MS_LOG(ERROR) << "param is nullptr.";
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
