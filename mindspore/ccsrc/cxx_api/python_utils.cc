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
#include "cxx_api/python_utils.h"
#include <mutex>
#include <vector>
#include <memory>
#include "mindspore/core/utils/ms_context.h"
#include "pybind11/pybind11.h"
#include "backend/kernel_compiler/oplib/oplib.h"

namespace py = pybind11;

namespace mindspore::api {
void RegAllOpFromPython() {
  static std::mutex init_mutex;
  static bool Initialized = false;

  std::lock_guard<std::mutex> lock(init_mutex);
  if (Initialized) {
    return;
  }
  Initialized = true;
  MsContext::GetInstance()->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  Py_Initialize();
  auto c_expression = PyImport_ImportModule("mindspore._c_expression");
  MS_EXCEPTION_IF_NULL(c_expression);
  PyObject *c_expression_dict = PyModule_GetDict(c_expression);
  MS_EXCEPTION_IF_NULL(c_expression_dict);

  PyObject *op_info_loader_class = PyDict_GetItemString(c_expression_dict, "OpInfoLoaderPy");
  MS_EXCEPTION_IF_NULL(op_info_loader_class);
  PyObject *op_info_loader = PyInstanceMethod_New(op_info_loader_class);
  MS_EXCEPTION_IF_NULL(op_info_loader);
  PyObject *op_info_loader_ins = PyObject_CallObject(op_info_loader, nullptr);
  MS_EXCEPTION_IF_NULL(op_info_loader_ins);
  auto all_ops_info_vector_addr_ul = PyObject_CallMethod(op_info_loader_ins, "get_all_ops_info", nullptr);
  MS_EXCEPTION_IF_NULL(all_ops_info_vector_addr_ul);
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
}

bool PythonIsInited() { return Py_IsInitialized() != 0; }
}  // namespace mindspore::api
