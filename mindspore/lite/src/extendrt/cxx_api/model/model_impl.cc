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
#include "extendrt/cxx_api/file_utils.h"
#include "extendrt/utils/tensor_utils.h"
#include "mindspore/core/utils/ms_context.h"
#include "extendrt/mindir_loader/mindir_model/mindir_model_util.h"
#include "src/extendrt/convert/runtime_convert.h"
#include "src/common/config_file.h"
#include "src/extendrt/utils/serialization.h"
#include "mindapi/ir/func_graph.h"
#include "mindapi/base/base.h"

namespace mindspore {
namespace {
const char *const kExecutionPlan = "execution_plan";
const char *const kConfigModelFileSection = "model_file";
const char *const kConfigMindIRPathKey = "mindir_path";
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
}  // namespace
void ModelImpl::SetMsContext() {
  if (MsContext::GetInstance() == nullptr) {
    MS_LOG(INFO) << "MsContext::GetInstance() is nullptr.";
    MsContext::device_type_seter([](std::shared_ptr<MsContext> &device_type_seter) {
      auto back_policy_env = std::getenv("MSLITE_ENABLE_HELPER");
      if (back_policy_env != nullptr) {
        device_type_seter.reset(new (std::nothrow) MsContext("ge", kAscendDevice));
      } else {
        device_type_seter.reset(new (std::nothrow) MsContext("vm", kCPUDevice));
      }
    });
  }
}

ConverterPlugin::~ConverterPlugin() {
#ifndef _WIN32
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
#endif
}

ConverterPlugin &ConverterPlugin::Instance() {
  static ConverterPlugin instance;
  return instance;
}

ConverterFunc ConverterPlugin::GetConverterFunc() {
#ifndef _WIN32
  if (converter_func_ == nullptr) {
    std::string plugin_path;
    auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite"}, "libruntime_convert_plugin.so", &plugin_path);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Get path of libruntime_convert_plugin.so failed. error: " << ret;
      return nullptr;
    }
    void *function = nullptr;
    ret = DLSoOpen(plugin_path, "RuntimeConvert", &handle_, &function, true);
    if (ret != kSuccess) {
      MS_LOG(WARNING) << "DLSoOpen RuntimeConvert failed, so path: " << plugin_path;
      return nullptr;
    }
    converter_func_ = reinterpret_cast<ConverterFunc>(function);
  }
#endif
  return converter_func_;
}

Status ModelImpl::BuildByBufferImpl(const void *model_data, size_t data_size, ModelType model_type,
                                    const std::shared_ptr<Context> &model_context, const std::string &model_path) {
  const void *model_buff = model_data;
  size_t model_size = data_size;
  auto mindir_path = GetConfig(kConfigModelFileSection, kConfigMindIRPathKey);
  std::string weight_path = "./";
  std::string base_path = "";
  if (!mindir_path.empty()) {
    base_path = mindir_path;
  } else {
    // user does not set mindir_path, convert from model_path
    base_path = model_path;
  }
  if (base_path.find("/") != std::string::npos) {
    weight_path = base_path.substr(0, base_path.rfind("/"));
  }
  SetMsContext();
  session_ = InferSession::CreateSession(model_context, config_info_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Create session failed.";
    return kLiteNullptr;
  }
  auto ret = session_->Init(model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Init session failed.";
    return ret;
  }
  if (infer::mindir::MindirModelUtil::NeedRuntimeConvert(model_data, data_size, model_context)) {
    return CompileGraphOnline(model_data, data_size, model_context);
  }
  graph_ = std::make_shared<Graph>();
  ret = mindspore::infer::Serialization::Load(model_buff, model_size, model_type, graph_.get(), Key{}, kDecModeAesGcm,
                                              weight_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Serialization::Load model failed.";
    return ret;
  }
  return session_->CompileGraph(graph_->graph_data_->GetFuncGraph());
}

Status ModelImpl::Build(const void *model_data, size_t data_size, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  return BuildByBufferImpl(model_data, data_size, model_type, model_context);
}

Status ModelImpl::Build(const std::string &model_path, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  if (model_path.empty()) {
    MS_LOG(ERROR) << "Model path cannot be empty";
    return kLiteNullptr;
  }
  auto buffer = ReadFile(model_path);
  if (buffer.DataSize() == 0) {
    MS_LOG(ERROR) << "Failed to read buffer from model file: " << model_path;
    return kLiteNullptr;
  }
  return BuildByBufferImpl(buffer.Data(), buffer.DataSize(), model_type, model_context, model_path);
}

Status ModelImpl::CompileGraphOnline(const void *model_data, size_t data_size,
                                     const std::shared_ptr<Context> &model_context) {
  MS_LOG(INFO) << "Need runtime convert";
  auto convert = ConverterPlugin::Instance().GetConverterFunc();
  if (convert == nullptr) {
    MS_LOG(ERROR) << "convert is nullptr";
    return kLiteNullptr;
  }
  auto api_graph = convert(static_cast<const char *>(model_data), data_size, model_context, config_info_);
  if (api_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to converter graph";
    return kLiteNullptr;
  }
  auto impl = api_graph->impl();
  auto inner_graph = std::dynamic_pointer_cast<FuncGraph>(impl);
  return session_->CompileGraph(inner_graph);
}

Status ModelImpl::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  MS_EXCEPTION_IF_NULL(session_);

  if (inputs.empty()) {
    MS_LOG(ERROR) << "Inputs is null.";
    return kLiteInputParamInvalid;
  }
  if (dims.empty()) {
    MS_LOG(ERROR) << "Dims is null.";
    return kLiteInputParamInvalid;
  }
  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "The size of inputs does not match the size of dims.";
    return kLiteInputParamInvalid;
  }
  auto model_inputs = session_->GetInputs();
  if (model_inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is null.";
    return kLiteParamInvalid;
  }
  if (inputs.size() != model_inputs.size()) {
    MS_LOG(ERROR) << "The size of inputs is incorrect.";
    return kLiteInputParamInvalid;
  }
  std::vector<mindspore::tensor::Tensor> resize_inputs = TensorUtils::MSTensorToTensor(inputs);
  return session_->Resize(resize_inputs, dims);
}

std::vector<MSTensor> ModelImpl::GetInputs() {
  MS_EXCEPTION_IF_NULL(session_);
  std::vector<MSTensor> inputs;

  auto graph_inputs = session_->GetInputs();

  for (size_t i = 0; i < graph_inputs.size(); i++) {
    auto tensor_impl = graph_inputs[i];
    inputs.push_back(MSTensor(tensor_impl));
  }
  return inputs;
}

std::vector<MSTensor> ModelImpl::GetOutputs() {
  MS_EXCEPTION_IF_NULL(session_);
  std::vector<MSTensor> outputs;
  auto graph_outputs = session_->GetOutputs();
  for (size_t i = 0; i < graph_outputs.size(); i++) {
    auto tensor_impl = graph_outputs[i];
    outputs.push_back(MSTensor(tensor_impl));
  }
  return outputs;
}

MSTensor ModelImpl::GetInputByTensorName(const std::string &name) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetInputByTensorName(name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

std::vector<std::string> ModelImpl::GetOutputTensorNames() {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    std::vector<std::string> empty;
    return empty;
  }
  return session_->GetOutputNames();
}

MSTensor ModelImpl::GetOutputByTensorName(const std::string &name) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetOutputByTensorName(name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(outputs);
  std::vector<mindspore::tensor::Tensor> graph_inputs = TensorUtils::MSTensorToTensor(inputs);
  std::vector<mindspore::tensor::Tensor> graph_outputs;
  std::vector<mindspore::tensor::Tensor> org_graph_outputs;
  if (!outputs->empty()) {
    graph_outputs = TensorUtils::MSTensorToTensor(*outputs);
    org_graph_outputs = graph_outputs;
  }
  auto ret = session_->RunGraph(graph_inputs, &graph_outputs, before, after);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ModelImpl::Predict RunGraph failed with " << ret;
    return ret;
  }
  bool output_remain = false;
  if (!org_graph_outputs.empty() && org_graph_outputs.size() == graph_outputs.size()) {
    output_remain = true;
    for (size_t i = 0; i < org_graph_outputs.size(); i++) {
      if (org_graph_outputs[i].data_ptr() != graph_outputs[i].data_ptr() ||
          org_graph_outputs[i].device_address() != graph_outputs[i].device_address()) {
        output_remain = false;
        break;
      }
    }
  }
  if (!output_remain) {
    *outputs = TensorUtils::TensorToMSTensor(graph_outputs, session_->GetOutputNames());
  }
  auto session_outputs = GetOutputs();
  if (graph_outputs.size() != session_outputs.size()) {
    MS_LOG(ERROR) << "Outputs count get from session " << session_outputs.size() << " != outputs count of RunGraph "
                  << graph_outputs.size();
    return kCoreFailed;
  }
  for (size_t i = 0; i < session_outputs.size(); i++) {
    auto &session_output = session_outputs[i];
    auto &execute_output = outputs->at(i);
    session_output.SetShape(execute_output.Shape());
    if (session_output.Data().get() != execute_output.Data().get()) {
      session_output.SetData(execute_output.MutableData(), false);
    }
    if (session_output.GetDeviceData() != execute_output.GetDeviceData()) {
      session_output.SetDeviceData(execute_output.GetDeviceData());
    }
  }
  return kSuccess;
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  return Predict(inputs, outputs, nullptr, nullptr);
}

Status ModelImpl::Predict() {
  auto inputs = GetInputs();
  auto outputs = GetOutputs();
  return Predict(inputs, &outputs);
}
bool ModelImpl::HasPreprocess() { return graph_->graph_data_->GetPreprocess().empty() ? false : true; }

Status ModelImpl::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  // Config preprocessor, temporary way to let mindspore.so depends on _c_dataengine
  std::string dataengine_so_path;
  Status dlret = DLSoPath({"libmindspore.so"}, "_c_dataengine", &dataengine_so_path);
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

Status ModelImpl::LoadConfig(const std::string &config_path) {
  ConfigInfos all_config_info;
  int ret = lite::GetAllSectionInfoFromConfigFile(config_path, &all_config_info);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "GetAllSectionInfoFromConfigFile fail!ret: " << ret;
    return kLiteFileError;
  }
  config_info_ = all_config_info;
  return kSuccess;
}

Status ModelImpl::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    if (config_info_.size() >= kMaxSectionNum) {
      MS_LOG(ERROR) << "config too many sections!";
      return kLiteError;
    }
    config_info_[section][config.first] = config.second;
    return kSuccess;
  }
  if (iter->second.size() >= kMaxConfigNumPerSection) {
    MS_LOG(ERROR) << "config too many items!";
    return kLiteError;
  }
  iter->second[config.first] = config.second;
  return kSuccess;
}

std::string ModelImpl::GetConfig(const std::string &section, const std::string &key) {
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    return "";
  }
  auto elem_iter = iter->second.find(key);
  if (elem_iter == iter->second.end()) {
    return "";
  }
  return elem_iter->second;
}
}  // namespace mindspore
