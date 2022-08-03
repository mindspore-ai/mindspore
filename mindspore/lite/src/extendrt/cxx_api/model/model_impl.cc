/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <algorithm>

#include "extendrt/cxx_api/model/model_impl.h"
#include "extendrt/cxx_api/dlutils.h"

namespace mindspore {
Status ModelImpl::Build(const void *model_data, size_t data_size, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  graph_ = std::make_shared<Graph>();
  auto ret = Serialization::Load(model_data, data_size, model_type, graph_.get());
  if (ret != kSuccess) {
    return ret;
  }
  session_ = InferSession::CreateSession(model_context);
  session_->Init(model_context);
  return session_->CompileGraph(graph_->graph_data_->GetFuncGraph());
}
Status ModelImpl::Build(const std::string &model_path, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  graph_ = std::make_shared<Graph>();
  auto ret = Serialization::Load(model_path, model_type, graph_.get());
  if (ret != kSuccess) {
    return ret;
  }
  session_ = InferSession::CreateSession(model_context);
  session_->Init(model_context);
  return session_->CompileGraph(graph_->graph_data_->GetFuncGraph());
}
Status ModelImpl::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  return kSuccess;
}

std::vector<MSTensor> ModelImpl::GetInputs() {
  MS_EXCEPTION_IF_NULL(session_);
  std::vector<MSTensor> inputs;

  auto graph_inputs = session_->GetInputs();
  auto graph_input_names = session_->GetInputNames();

  for (size_t i = 0; i < graph_inputs.size(); i++) {
    auto graph_input = graph_inputs[i];
    std::string graph_input_name = graph_input_names[i];
    auto type_id = graph_input->data_type_c();
    auto data_type = static_cast<mindspore::DataType>(type_id);
    MSTensor ms_tensor(graph_input_name, data_type, graph_input->shape_c(), graph_input->data_c(), graph_input->Size());
    inputs.push_back(ms_tensor);
  }

  return inputs;
}

std::vector<MSTensor> ModelImpl::GetOutputs() {
  MS_EXCEPTION_IF_NULL(session_);
  std::vector<MSTensor> outputs;

  auto graph_outputs = session_->GetOutputs();
  auto graph_output_names = session_->GetOutputNames();

  for (size_t i = 0; i < graph_outputs.size(); i++) {
    auto graph_output = graph_outputs[i];
    std::string graph_output_name = graph_output_names[i];
    auto type_id = graph_output->data_type_c();
    auto data_type = static_cast<mindspore::DataType>(type_id);
    MSTensor ms_tensor(graph_output_name, data_type, graph_output->shape_c(), graph_output->data_c(),
                       graph_output->Size());
    outputs.push_back(ms_tensor);
  }

  return outputs;
}

MSTensor ModelImpl::GetInputByTensorName(const std::string &name) { return MSTensor(); }
std::vector<std::string> ModelImpl::GetOutputTensorNames() { return std::vector<std::string>(); }
MSTensor ModelImpl::GetOutputByTensorName(const std::string &name) { return MSTensor(); }

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(session_);
  std::vector<mindspore::tensor::TensorPtr> graph_inputs = MSTensorToTensorPtr(inputs);
  std::vector<mindspore::tensor::TensorPtr> graph_outputs;
  auto ret = session_->RunGraph(graph_inputs, &graph_outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ModelImpl::Predict RunGraph failed with " << ret;
    return ret;
  }
  auto ms_outputs = TensorPtrToMSTensor(graph_outputs, session_->GetOutputNames());
  (void)std::copy(ms_outputs.begin(), ms_outputs.end(), std::back_inserter(*outputs));
  return kSuccess;
}

std::vector<mindspore::tensor::TensorPtr> ModelImpl::MSTensorToTensorPtr(const std::vector<MSTensor> &ms_tensors) {
  std::vector<mindspore::tensor::TensorPtr> tensor_ptrs;

  for (auto ms_tensor : ms_tensors) {
    auto data_type = ms_tensor.DataType();
    auto type_id = static_cast<mindspore::TypeId>(data_type);
    auto shape = ms_tensor.Shape();
    auto data = ms_tensor.MutableData();
    auto data_size = ms_tensor.DataSize();
    auto tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(type_id, shape, data, data_size);
    tensor_ptrs.push_back(tensor_ptr);
  }
  return tensor_ptrs;
}

std::vector<MSTensor> ModelImpl::TensorPtrToMSTensor(std::vector<mindspore::tensor::TensorPtr> tensor_ptrs,
                                                     const std::vector<std::string> &tensor_names) {
  std::vector<MSTensor> ms_tensors;

  for (size_t i = 0; i < tensor_ptrs.size(); i++) {
    auto graph_tensor = tensor_ptrs[i];
    std::string graph_tensor_name = tensor_names[i];
    auto type_id = graph_tensor->data_type_c();
    auto data_type = static_cast<mindspore::DataType>(type_id);
    MSTensor ms_tensor(graph_tensor_name, data_type, graph_tensor->shape_c(), graph_tensor->data_c(),
                       graph_tensor->Size());
    ms_tensors.push_back(ms_tensor);
  }

  return ms_tensors;
}

bool ModelImpl::HasPreprocess() { return graph_->graph_data_->GetPreprocess().empty() ? false : true; }

Status ModelImpl::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  // Config preprocessor, temporary way to let mindspore.so depends on _c_dataengine
  std::string dataengine_so_path;
  Status dlret = DLSoPath(&dataengine_so_path);
  CHECK_FAIL_AND_RELEASE(dlret, nullptr, "Parse dataengine_so failed: " + dlret.GetErrDescription());

  // Run preprocess
  if (!HasPreprocess()) {
    MS_LOG(ERROR) << "Attempt to predict with data preprocessor, but no preprocessor is defined in MindIR.";
    return Status(kMEFailed, "Attempt to predict with data preprocessor, but no preprocessor is defined in MindIR.");
  }

  void *handle = nullptr;
  void *function = nullptr;
  dlret = DLSoOpen(dataengine_so_path, "ExecuteRun_C", &handle, &function);
  CHECK_FAIL_AND_RELEASE(dlret, handle, "Parse ExecuteRun_C failed: " + dlret.GetErrDescription());
  auto ExecuteRun =
    (void (*)(const std::vector<std::shared_ptr<dataset::Execute>> &, const std::vector<mindspore::MSTensor> &,
              std::vector<mindspore::MSTensor> *, Status *))(function);

  // perform preprocess on each tensor separately
  std::vector<std::shared_ptr<dataset::Execute>> preprocessor = graph_->graph_data_->GetPreprocess();
  std::vector<std::vector<MSTensor>> output_unbatch;
  std::vector<MSTensor> output_batched;
  for (auto tensor : inputs) {
    std::vector<MSTensor> temp;
    ExecuteRun(preprocessor, tensor, &temp, &dlret);
    CHECK_FAIL_AND_RELEASE(dlret, handle, "Run preprocess failed: " + dlret.GetErrDescription());
    output_unbatch.push_back(temp);
  }

  // Construct a tensor with batch dim
  output_batched.resize(output_unbatch[0].size());
  for (size_t i = 0; i < output_batched.size(); i++) {
    std::vector<int64_t> ori_shape = output_unbatch[0][i].Shape();
    ori_shape.insert(ori_shape.begin(), output_unbatch.size());
    output_batched[i] = mindspore::MSTensor("outputs", output_unbatch[0][i].DataType(), ori_shape, nullptr,
                                            output_unbatch[0][i].DataSize() * output_unbatch.size());
  }

  // Copy unbatch data into tensor
  for (size_t i = 0; i < output_unbatch[0].size(); i++) {
    size_t offset = 0;
    for (size_t j = 0; j < output_unbatch.size(); j++) {
      auto ret =
        memcpy_s(reinterpret_cast<unsigned uint8_t *>(output_batched[i].MutableData()) + offset,
                 output_unbatch[j][i].DataSize(), output_unbatch[j][i].MutableData(), output_unbatch[j][i].DataSize());
      if (ret) {
        MS_LOG(ERROR) << "Memory copy failed to construct High-Dim Tensor.";
        return Status(kMEFailed, "Memory copy failed to construct High-Dim Tensor.");
      }
      offset += output_unbatch[j][i].DataSize();
    }
  }
  *outputs = output_batched;
  DLSoClose(handle);
  return kSuccess;
#else
  MS_LOG(ERROR) << "Data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Data preprocess is not supported on Windows yet.");
#endif
}

Status ModelImpl::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs,
                                        std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  // Run preprocess
  std::vector<MSTensor> preprocess_outputs;
  Status ret = Preprocess(inputs, &preprocess_outputs);
  if (ret != kSuccess) {
    return ret;
  }

  // Run prediction
  ret = Predict(preprocess_outputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run predict failed: " << ret.GetErrDescription();
    return ret;
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "Predict with data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Predict with data preprocess is not supported on Windows yet.");
#endif
}
}  // namespace mindspore
