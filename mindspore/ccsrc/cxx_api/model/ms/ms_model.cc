/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "cxx_api/model/ms/ms_model.h"
#include <memory>
#include <algorithm>
#include <fstream>

#include "load_mindir/load_model.h"
#include "backend/session/session_basic.h"
#include "backend/session/session_factory.h"
#include "backend/session/executor_manager.h"
#include "base/base_ref_utils.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "utils/context/context_extends.h"
#include "runtime/device/kernel_runtime_manager.h"

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"

#ifdef ENABLE_D
#include "utils/ms_context.h"
#endif

using std::string;
using std::vector;

namespace py = pybind11;
namespace mindspore {
namespace api {
MsModel::MsModel(uint32_t device_id) : device_id_(device_id) {}
MsModel::~MsModel() = default;

TypeId TransInferDataType2TypeId(DataType data_type) {
  const std::map<api::DataType, TypeId> type2id_map{
    {api::kMsUnknown, TypeId::kNumberTypeBegin},   {api::kMsBool, TypeId::kNumberTypeBool},
    {api::kMsInt8, TypeId::kNumberTypeInt8},       {api::kMsUint8, TypeId::kNumberTypeUInt8},
    {api::kMsInt16, TypeId::kNumberTypeInt16},     {api::kMsUint16, TypeId::kNumberTypeUInt16},
    {api::kMsInt32, TypeId::kNumberTypeInt32},     {api::kMsUint32, TypeId::kNumberTypeUInt32},
    {api::kMsInt64, TypeId::kNumberTypeInt64},     {api::kMsUint64, TypeId::kNumberTypeUInt64},
    {api::kMsFloat16, TypeId::kNumberTypeFloat16}, {api::kMsFloat32, TypeId::kNumberTypeFloat32},
    {api::kMsFloat64, TypeId::kNumberTypeFloat64},
  };
  auto it = type2id_map.find(data_type);
  if (it == type2id_map.end()) {
    MS_LOG_WARNING << "Unsupported MSI data type " << data_type;
    return TypeId::kNumberTypeBegin;
  } else {
    return it->second;
  }
}

DataType TransTypeId2InferDataType(TypeId type_id) {
  const std::map<TypeId, api::DataType> id2type_map{
    {TypeId::kNumberTypeBegin, api::kMsUnknown},   {TypeId::kNumberTypeBool, api::kMsBool},
    {TypeId::kNumberTypeFloat64, api::kMsFloat64}, {TypeId::kNumberTypeInt8, api::kMsInt8},
    {TypeId::kNumberTypeUInt8, api::kMsUint8},     {TypeId::kNumberTypeInt16, api::kMsInt16},
    {TypeId::kNumberTypeUInt16, api::kMsUint16},   {TypeId::kNumberTypeInt32, api::kMsInt32},
    {TypeId::kNumberTypeUInt32, api::kMsUint32},   {TypeId::kNumberTypeInt64, api::kMsInt64},
    {TypeId::kNumberTypeUInt64, api::kMsUint64},   {TypeId::kNumberTypeFloat16, api::kMsFloat16},
    {TypeId::kNumberTypeFloat32, api::kMsFloat32},
  };
  auto it = id2type_map.find(type_id);
  if (it == id2type_map.end()) {
    MS_LOG_WARNING << "Unsupported data id " << type_id;
    return api::kMsUnknown;
  } else {
    return it->second;
  }
}

Buffer MsModel::ReadFile(const std::string &file) {
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return Buffer();
  }
  std::ifstream ifs(file);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << file << " is not exist";
    return Buffer();
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << file << "open failed";
    return Buffer();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  Buffer buffer;
  buffer.ResizeData(size);
  ifs.seekg(0, std::ios::beg);
  ifs.read(static_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}

Status MsModel::LoadModel(const Buffer &model_data, ModelType type, const std::map<std::string, std::string> &options) {
  auto status = InitEnv({});
  if (status != SUCCESS) {
    MS_LOG(ERROR) << "Init env failed";
    return FAILED;
  }
  std::shared_ptr<FuncGraph> anf_graph;
  Py_Initialize();
  try {
    anf_graph = ConvertStreamToFuncGraph(static_cast<const char *>(model_data.Data()), model_data.DataSize());
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference LoadModel failed";
    return FAILED;
  }
  Status ret = CompileGraph(anf_graph);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Compile graph model failed";
    return FAILED;
  }
  session_impl_->GetModelInputsInfo(graph_id_, &inputs_, &input_names_);
  session_impl_->GetModelOutputsInfo(graph_id_, &outputs_, &output_names_);
  if (inputs_.empty() || inputs_.size() != input_names_.size()) {
    MS_LOG_ERROR << "Get model inputs info failed";
    return FAILED;
  }
  if (outputs_.empty() || outputs_.size() != output_names_.size()) {
    MS_LOG_ERROR << "Get model outputs info failed";
    return FAILED;
  }
  MS_LOG(INFO) << "Load model success";

#ifdef ENABLE_D
  // set d context
  rtError_t rt_ret = rtCtxGetCurrent(&context_);
  if (rt_ret != RT_ERROR_NONE || context_ == nullptr) {
    MS_LOG(ERROR) << "the ascend device context is null";
    return FAILED;
  }
#endif
  load_flag_ = true;
  return SUCCESS;
}

Status MsModel::LoadModel(const std::string &file_name, ModelType type,
                          const std::map<std::string, std::string> &options) {
  auto graphBuf = ReadFile(file_name);
  if (graphBuf.DataSize() == 0) {
    MS_LOG(ERROR) << "Read model file failed, file name is " << file_name.c_str();
    return FAILED;
  }
  auto status = LoadModel(graphBuf, type, options);
  if (status != SUCCESS) {
    MS_LOG(ERROR) << "Load graph model failed, file name is " << file_name.c_str();
    return FAILED;
  }
  return SUCCESS;
}

Status MsModel::UnloadModel() {
  if (!load_flag_) {
    MS_LOG_ERROR << "Model has not been loaded";
    return FAILED;
  }
  FinalizeEnv();
  load_flag_ = false;
  return SUCCESS;
}

Status MsModel::Train(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status MsModel::Eval(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status MsModel::Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (!load_flag_) {
    MS_LOG(ERROR) << "No model is loaded, predict failed.";
    return FAILED;
  }
  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "inputs count not match, required count " << inputs_.size() << ", given count " << inputs.size();
    return INVALID_INPUTS;
  }
  std::vector<Buffer> request;
  std::vector<Buffer> reply;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    const auto &input_name = input_names_[i];
    auto iter = inputs.find(input_name);
    if (iter == inputs.end()) {
      MS_LOG(ERROR) << "Model missing input " << input_name;
      return INVALID_INPUTS;
    }

    if (iter->second.DataSize() != inputs_[i]->Size()) {
      MS_LOG(ERROR) << "input " << i << " data size not match, required size " << inputs_[i]->Size() << ", given count "
                    << iter->second.DataSize();
      return INVALID_INPUTS;
    }
    request.push_back(iter->second);
  }
  if (ExecuteModel(request, &reply) != SUCCESS) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return FAILED;
  }
  if (outputs_.size() != reply.size()) {
    MS_LOG(ERROR) << "Predict output size " << reply.size() << " not match output size got from model info "
                  << outputs_.size();
    return FAILED;
  }
  outputs->clear();
  for (size_t i = 0; i < reply.size(); i++) {
    outputs->emplace(output_names_[i], reply[i]);
  }
  return SUCCESS;
}

Status MsModel::ExecuteModel(const std::vector<Buffer> &request, std::vector<Buffer> *reply) {
  MS_EXCEPTION_IF_NULL(reply);
#ifdef ENABLE_D
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "rtCtx is nullptr";
    return FAILED;
  }
  rtError_t rt_ret = rtCtxSetCurrent(context_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "set Ascend rtCtx failed";
    return FAILED;
  }
#endif
  vector<tensor::TensorPtr> inputs;
  for (size_t i = 0; i < request.size(); i++) {
    auto &item = request[i];
    auto input = inputs_[i];
    if (input->Size() != item.DataSize()) {
      MS_LOG(ERROR) << "Predict input " << i << " data size " << item.DataSize() << " not match model input data size "
                    << input->Size();
      return FAILED;
    }
    auto ret = memcpy_s(input->data_c(), input->Size(), item.Data(), item.DataSize());
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Tensor copy failed";
      return FAILED;
    }
    inputs.push_back(input);
  }
  vector<tensor::TensorPtr> outputs = RunGraph(inputs);
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return FAILED;
  }
  reply->clear();
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(*reply),
                 [](const tensor::TensorPtr &tensor) { return Buffer(tensor->data_c(), tensor->Size()); });
  return SUCCESS;
}

Status MsModel::FinalizeEnv() {
  MS_LOG_INFO << "Start finalize env";
  py::gil_scoped_acquire acquire;
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return FAILED;
  }
  if (!context::CloseTsd(ms_context)) {
    MS_LOG(ERROR) << "Inference CloseTsd failed!";
    return FAILED;
  }
  MS_LOG_INFO << "End finalize env";
  return SUCCESS;
}

std::shared_ptr<FuncGraph> MsModel::LoadModel(const char *model_buf, size_t size, const std::string &device) {
  Py_Initialize();
  MS_EXCEPTION_IF_NULL(model_buf);
  try {
    auto anf_graph = ConvertStreamToFuncGraph(model_buf, size);
    return anf_graph;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference LoadModel failed: " << e.what();
    return nullptr;
  }
}

void MsModel::RegAllOp() {
  static std::mutex init_mutex;
  static bool Initialized = false;

  std::lock_guard<std::mutex> lock(init_mutex);
  if (Initialized) {
    return;
  }
  Initialized = true;
  auto ms_context_instance = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context_instance);
  ms_context_instance->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  try {
    std::shared_ptr<py::scoped_interpreter> guard;
    if (Py_IsInitialized() == 0) {
      guard = std::make_shared<py::scoped_interpreter>();
    }
    py::module c_expression = py::module::import("mindspore._c_expression");
    size_t ops_info_long = c_expression.attr("OpInfoLoaderPy")().attr("get_all_ops_info")().cast<size_t>();
    auto all_ops_info = reinterpret_cast<std::vector<kernel::OpInfo *> *>(static_cast<uintptr_t>(ops_info_long));
    for (auto op_info : *all_ops_info) {
      kernel::OpLib::RegOpInfo(std::shared_ptr<kernel::OpInfo>(op_info));
    }
    all_ops_info->clear();
    delete all_ops_info;
  } catch (const std::runtime_error &ex) {
    MS_LOG_EXCEPTION << ex.what();
  }
}

Status MsModel::CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr) {
  MS_ASSERT(session_impl_ != nullptr);
  try {
    graph_id_ = session_impl_->CompileGraph(NOT_NULL(funcGraphPtr));
    py::gil_scoped_release gil_release;
    return SUCCESS;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference CompileGraph failed: " << e.what();
    return FAILED;
  }
}

std::vector<tensor::TensorPtr> MsModel::RunGraph(const std::vector<tensor::TensorPtr> &inputs) {
  try {
    VectorRef outputs;
    session_impl_->RunGraph(graph_id_, inputs, &outputs);
    return TransformVectorRefToMultiTensor(outputs);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference Rungraph failed: " << e.what();
    return std::vector<tensor::TensorPtr>();
  }
}

Status MsModel::InitEnv(const std::unordered_map<std::string, std::string> &other_options) {
  RegAllOp();
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return FAILED;
  }
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id_);
  ms_context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  if (!context::OpenTsd(ms_context)) {
    MS_LOG(ERROR) << "Session init OpenTsd failed!";
    return FAILED;
  }
  session_impl_ = session::SessionFactory::Get().Create(kDavinciInferenceDevice);
  if (session_impl_ == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << kDavinciInferenceDevice
                  << " is available.";
    return FAILED;
  }
  session_impl_->Init(device_id_);
  return SUCCESS;
}

Status MsModel::CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs) const {
  MS_ASSERT(session_impl_ != nullptr);
  std::string error_msg;
  if (!session_impl_->CheckModelInputs(graph_id, inputs, &error_msg)) {
    return Status(INVALID_INPUTS, error_msg);
  }
  return SUCCESS;
}

Status MsModel::GetInputsInfo(std::vector<Tensor> *tensor_list) const {
  MS_EXCEPTION_IF_NULL(tensor_list);
  tensor_list->clear();
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto &tensor = inputs_[i];
    Tensor infer_tensor;
    infer_tensor.SetName(input_names_[i]);
    infer_tensor.SetDataType(TransTypeId2InferDataType(tensor->data_type()));
    infer_tensor.SetShape(tensor->shape());
    tensor_list->push_back(infer_tensor);
  }
  return SUCCESS;
}

Status MsModel::GetOutputsInfo(std::vector<Tensor> *tensor_list) const {
  MS_EXCEPTION_IF_NULL(tensor_list);
  tensor_list->clear();
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto &tensor = outputs_[i];
    Tensor infer_tensor;
    infer_tensor.SetName(output_names_[i]);
    infer_tensor.SetDataType(TransTypeId2InferDataType(tensor->data_type()));
    infer_tensor.SetShape(tensor->shape());
    tensor_list->push_back(infer_tensor);
  }
  return SUCCESS;
}
}  // namespace api
}  // namespace mindspore
