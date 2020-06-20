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

#include <memory>
#include <algorithm>
#include "include/inference.h"
#include "session/session.h"
#include "utils/load_onnx/anf_converter.h"
#include "session/session_basic.h"
#include "session/session_factory.h"
#include "utils/base_ref_utils.h"
#include "kernel/oplib/oplib.h"
#ifdef ENABLE_D
#include "utils/context/ms_context.h"
#include "session/ascend_session.h"
#else
#include "session/cpu_session.h"
#endif

namespace py = pybind11;
namespace mindspore::inference {
std::shared_ptr<FuncGraph> LoadModel(const char *model_buf, size_t size, const std::string &device) {
  inference::Session::RegAllOp();
  auto anf_graph = lite::AnfConverter::RunAnfConverter(model_buf, size);
  return anf_graph;
}

std::shared_ptr<MSSession> MSSession::CreateSession(const std::string &device, uint32_t device_id) {
  auto session = std::make_shared<inference::Session>();
  auto ret = session->Init(device, device_id);
  if (ret != 0) {
    return nullptr;
  }
  return session;
}

void Session::RegAllOp() {
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

uint32_t Session::CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr) {
  MS_ASSERT(session_impl_ != nullptr);
  return session_impl_->CompileGraph(NOT_NULL(funcGraphPtr));
}

MultiTensor Session::RunGraph(uint32_t graph_id, const std::vector<std::shared_ptr<inference::MSTensor>> &inputs) {
  std::vector<tensor::TensorPtr> inTensors;
  bool has_error = false;
  std::transform(inputs.begin(), inputs.end(), inTensors.begin(),
                 [&has_error](const std::shared_ptr<inference::MSTensor> &tensor_ptr) -> tensor::TensorPtr {
                   if (tensor_ptr == nullptr) {
                     MS_LOG(WARNING) << "input MSTensor is nullptr, return nullptr";
                     has_error = true;
                     return nullptr;
                   }
                   auto tensor = static_cast<inference::Tensor *>(tensor_ptr.get());
                   if (tensor == nullptr) {
                     MS_LOG(ERROR) << "Can not cast input MSTensor to tensor";
                     has_error = true;
                     return nullptr;
                   }
                   return tensor->tensor();
                 });
  if (has_error) {
    MS_LOG(ERROR) << "Init Tensor failed, returning empty result";
    std::vector<std::vector<std::shared_ptr<inference::MSTensor>>> multiTensor;
    return multiTensor;
  }
  VectorRef outputs;
  session_impl_->RunGraph(graph_id, inTensors, &outputs);

  return TransformVectorRefToMultiTensor(outputs);
}

int Session::Init(const std::string &device, uint32_t device_id) {
  RegAllOp();
  session_impl_ = session::SessionFactory::Get().Create(device);
  if (session_impl_ == nullptr) {
    MS_LOG(ERROR) << "Session create failed!, please make sure target device:" << device << " is available.";
    return -1;
  }
  session_impl_->Init(device_id);
  return 0;
}

Session::Session() = default;
}  // namespace mindspore::inference
