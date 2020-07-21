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

#include "backend/session/infer_session.h"
#include <memory>
#include <algorithm>
#include "include/inference.h"
#include "utils/load_onnx/anf_converter.h"
#include "backend/session/session_basic.h"
#include "backend/session/session_factory.h"
#include "utils/base_ref_utils.h"
#include "backend/kernel_compiler/oplib/oplib.h"

#ifdef ENABLE_D
#include "utils/context/ms_context.h"
#endif

using std::string;
using std::vector;

namespace py = pybind11;
namespace mindspore::inference {

std::shared_ptr<InferSession> InferSession::CreateSession(const std::string &device, uint32_t device_id) {
  try {
    auto session = std::make_shared<MSInferSession>();
    bool ret = session->InitEnv(device, device_id);
    if (!ret) {
      return nullptr;
    }
    return session;
  } catch (std::bad_alloc &e) {
    MS_LOG(ERROR) << "Inference CreatSession failed, failed to alloc memory";
    return nullptr;
  }
}

MSInferSession::MSInferSession() = default;
MSInferSession::~MSInferSession() = default;

std::shared_ptr<std::vector<char>> MSInferSession::ReadFile(const std::string &file) {
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  std::string realPath = file;
  std::ifstream ifs(realPath);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << realPath << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << realPath << "open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  std::shared_ptr<std::vector<char>> buf(new (std::nothrow) std::vector<char>(size));
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << realPath;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf->data(), size);
  ifs.close();

  return buf;
}

bool MSInferSession::LoadModelFromFile(const std::string &file_name, uint32_t &model_id) {
  auto graphBuf = ReadFile(file_name);
  if (graphBuf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed, file name is " << file_name.c_str();
    return false;
  }
  auto graph = LoadModel(graphBuf->data(), graphBuf->size(), device_type_);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Load graph model failed, file name is " << file_name.c_str();
    return false;
  }
  bool ret = CompileGraph(graph, model_id);
  if (!ret) {
    MS_LOG(ERROR) << "Compile graph model failed, file name is " << file_name.c_str();
    return false;
  }
  MS_LOG(INFO) << "Load model from file " << file_name << " success";

#ifdef ENABLE_D
  // set d context
  rtError_t rt_ret = rtCtxGetCurrent(&context_);
  if (rt_ret != RT_ERROR_NONE || context_ == nullptr) {
    MS_LOG(ERROR) << "the ascend device context is null";
    return false;
  }
#endif

  return true;
}

bool MSInferSession::UnloadModel(uint32_t model_id) { return true; }

tensor::TensorPtr ServingTensor2MSTensor(const InferTensorBase &out_tensor) {
  std::vector<int> shape;
  for (auto dim : out_tensor.shape()) {
    shape.push_back(static_cast<int>(dim));
  }
  TypeId data_type;
  const std::map<inference::DataType, TypeId> type2id_map{
    {inference::kMSI_Unknown, TypeId::kNumberTypeBegin},   {inference::kMSI_Bool, TypeId::kNumberTypeBool},
    {inference::kMSI_Int8, TypeId::kNumberTypeInt8},       {inference::kMSI_Uint8, TypeId::kNumberTypeUInt8},
    {inference::kMSI_Int16, TypeId::kNumberTypeInt16},     {inference::kMSI_Uint16, TypeId::kNumberTypeUInt16},
    {inference::kMSI_Int32, TypeId::kNumberTypeInt32},     {inference::kMSI_Uint32, TypeId::kNumberTypeUInt32},
    {inference::kMSI_Int64, TypeId::kNumberTypeInt64},     {inference::kMSI_Uint64, TypeId::kNumberTypeUInt64},
    {inference::kMSI_Float16, TypeId::kNumberTypeFloat16}, {inference::kMSI_Float32, TypeId::kNumberTypeFloat32},
    {inference::kMSI_Float64, TypeId::kNumberTypeFloat64},
  };
  auto it = type2id_map.find(out_tensor.data_type());
  if (it == type2id_map.end()) {
    MSI_LOG_WARNING << "undefined MSI data type " << out_tensor.data_type();
    return nullptr;
  } else {
    data_type = it->second;
  }

  auto ms_tensor = std::make_shared<tensor::Tensor>(data_type, shape);
  memcpy_s(ms_tensor->data_c(), ms_tensor->Size(), out_tensor.data(), out_tensor.data_size());
  return ms_tensor;
}

void MSTensor2ServingTensor(tensor::TensorPtr ms_tensor, InferTensorBase &out_tensor) {
  vector<int64_t> shape;
  for (auto dim : ms_tensor->shape()) {
    shape.push_back(dim);
  }
  out_tensor.set_shape(shape);

  const std::map<TypeId, inference::DataType> id2type_map{
    {TypeId::kNumberTypeBegin, inference::kMSI_Unknown},   {TypeId::kNumberTypeBool, inference::kMSI_Bool},
    {TypeId::kNumberTypeFloat64, inference::kMSI_Float64}, {TypeId::kNumberTypeInt8, inference::kMSI_Int8},
    {TypeId::kNumberTypeUInt8, inference::kMSI_Uint8},     {TypeId::kNumberTypeInt16, inference::kMSI_Int16},
    {TypeId::kNumberTypeUInt16, inference::kMSI_Uint16},   {TypeId::kNumberTypeInt32, inference::kMSI_Int32},
    {TypeId::kNumberTypeUInt32, inference::kMSI_Uint32},   {TypeId::kNumberTypeInt64, inference::kMSI_Int64},
    {TypeId::kNumberTypeUInt64, inference::kMSI_Uint64},   {TypeId::kNumberTypeFloat16, inference::kMSI_Float16},
    {TypeId::kNumberTypeFloat32, inference::kMSI_Float32},
  };
  auto it = id2type_map.find(ms_tensor->data_type());
  if (it == id2type_map.end()) {
    MSI_LOG_WARNING << "undefined MS data type " << ms_tensor->data_type();
    out_tensor.set_data_type(inference::kMSI_Unknown);
  } else {
    out_tensor.set_data_type(it->second);
  }
  out_tensor.set_data(ms_tensor->data_c(), ms_tensor->Size());
}

bool MSInferSession::ExecuteModel(uint32_t model_id, const RequestBase &request, ReplyBase &reply) {
#ifdef ENABLE_D
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "rtCtx is nullptr";
    return false;
  }
  rtError_t rt_ret = rtCtxSetCurrent(context_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "set Ascend rtCtx failed";
    return false;
  }
#endif

  vector<tensor::TensorPtr> inputs;
  for (size_t i = 0; i < request.size(); i++) {
    if (request[i] == nullptr) {
      MS_LOG(ERROR) << "Execute Model " << model_id << " Failed， input tensor is null, index " << i;
      return false;
    }
    auto input = ServingTensor2MSTensor(*request[i]);
    if (input == nullptr) {
      MS_LOG(ERROR) << "Tensor convert failed";
      return false;
    }
    inputs.push_back(input);
  }
  if (!CheckModelInputs(model_id, inputs)) {
    MS_LOG(ERROR) << "Check Model " << model_id << " Inputs Failed";
    return false;
  }
  vector<tensor::TensorPtr> outputs = RunGraph(model_id, inputs);
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Execute Model " << model_id << " Failed";
    return false;
  }
  reply.clear();
  for (const auto &tensor : outputs) {
    auto out_tensor = reply.add();
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "Execute Model " << model_id << " Failed， add output tensor failed";
      return false;
    }
    MSTensor2ServingTensor(tensor, *out_tensor);
  }
  return true;
}

bool MSInferSession::FinalizeEnv() {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return false;
  }
  if (!ms_context->CloseTsd()) {
    MS_LOG(ERROR) << "Inference CloseTsd failed!";
    return false;
  }
  return true;
}

std::shared_ptr<FuncGraph> MSInferSession::LoadModel(const char *model_buf, size_t size, const std::string &device) {
  try {
    auto anf_graph = lite::AnfConverter::RunAnfConverter(model_buf, size);
    return anf_graph;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference LoadModel failed";
    return nullptr;
  }
}

void MSInferSession::RegAllOp() {
  static std::mutex init_mutex;
  static bool Initialized = false;

  std::lock_guard<std::mutex> lock(init_mutex);
  if (Initialized) {
    return;
  }
  Initialized = true;
  MsContext::GetInstance()->set_execution_mode(kGraphMode);
  Py_Initialize();
  auto c_expression = PyImport_ImportModule("mindspore._c_expression");
  if (c_expression == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to import mindspore._c_expression  module.";
    return;
  }
  PyObject *c_expression_dict = PyModule_GetDict(c_expression);

  PyObject *op_info_loader_class = PyDict_GetItemString(c_expression_dict, "OpInfoLoaderPy");
  if (op_info_loader_class == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get op_info_loader_class from mindspore._c_expression.";
    return;
  }
  PyObject *op_info_loader = PyInstanceMethod_New(op_info_loader_class);
  if (op_info_loader == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to create op_info_loader instance.";
    return;
  }
  PyObject *op_info_loader_ins = PyObject_CallObject(op_info_loader, nullptr);
  if (op_info_loader_ins == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to call op_info_loader instance.";
    return;
  }
  auto all_ops_info_vector_addr_ul = PyObject_CallMethod(op_info_loader_ins, "get_all_ops_info", nullptr);
  if (all_ops_info_vector_addr_ul == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to call get_all_ops_addr.";
    return;
  }
  auto all_ops_info_vector_addr = PyLong_AsVoidPtr(all_ops_info_vector_addr_ul);
  auto all_ops_info = static_cast<std::vector<kernel::OpInfo *> *>(all_ops_info_vector_addr);
  for (auto op_info : *all_ops_info) {
    kernel::OpLib::RegOpInfo(std::shared_ptr<kernel::OpInfo>(op_info));
  }
  all_ops_info->clear();
  delete all_ops_info;
  Py_DECREF(op_info_loader);
  Py_DECREF(op_info_loader_class);
  Py_DECREF(c_expression_dict);
  Py_DECREF(c_expression);
  return;
}

bool MSInferSession::CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr, uint32_t &model_id) {
  MS_ASSERT(session_impl_ != nullptr);
  try {
    auto graph_id = session_impl_->CompileGraph(NOT_NULL(funcGraphPtr));
    py::gil_scoped_release gil_release;
    model_id = graph_id;
    return true;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference CompileGraph failed";
    return false;
  }
}

std::vector<tensor::TensorPtr> MSInferSession::RunGraph(uint32_t graph_id,
                                                        const std::vector<tensor::TensorPtr> &inputs) {
  try {
    VectorRef outputs;
    session_impl_->RunGraph(graph_id, inputs, &outputs);

    return TransformVectorRefToMultiTensor(outputs);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Inference Rungraph failed";
    return std::vector<tensor::TensorPtr>();
  }
}

string MSInferSession::AjustTargetName(const std::string &device) {
  if (device == kAscendDevice) {
    return std::string(kAscendDevice) + "Inference";
  } else {
    MS_LOG(ERROR) << "Only support device Ascend right now";
    return "";
  }
}

bool MSInferSession::InitEnv(const std::string &device, uint32_t device_id) {
  RegAllOp();
  auto ms_context = MsContext::GetInstance();
  ms_context->set_execution_mode(kGraphMode);
  ms_context->set_device_id(device_id);
  auto ajust_device = AjustTargetName(device);
  if (ajust_device == "") {
    return false;
  }
  ms_context->set_device_target(device);
  session_impl_ = session::SessionFactory::Get().Create(ajust_device);
  if (session_impl_ == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << device << " is available.";
    return false;
  }
  session_impl_->Init(device_id);
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "Get Context failed!";
    return false;
  }
  if (!ms_context->OpenTsd()) {
    MS_LOG(ERROR) << "Session init OpenTsd failed!";
    return false;
  }
  return true;
}

bool MSInferSession::CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs) const {
  MS_ASSERT(session_impl_ != nullptr);
  return session_impl_->CheckModelInputs(graph_id, inputs);
}

}  // namespace mindspore::inference
