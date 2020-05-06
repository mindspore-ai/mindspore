/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "kernel/tbe/tbe_python_funcs.h"
#include "kernel/tbe/tbe_utils.h"
#include "common/utils.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace kernel {
using mindspore::kernel::tbe::TbeUtils;
constexpr auto kTbeProcessModule = "mindspore._extends.parallel_compile.tbe_compiler.tbe_process";
constexpr auto kCreateTbeParallelCompilerFunc = "create_tbe_parallel_compiler";
constexpr auto kOpSelectFormatFunc = "op_select_format";
constexpr auto kCheckSupportedFunc = "check_supported";
constexpr auto kTBEException = "TBEException";

PyObject *TbePythonFuncs::pCreateTbeParallelCompilerFunc_ = nullptr;
PyObject *TbePythonFuncs::pTbeCompiler_ = nullptr;
PyObject *TbePythonFuncs::pOpSelectFormatFunc_ = nullptr;
PyObject *TbePythonFuncs::pCheckSupportedFunc_ = nullptr;
bool TbePythonFuncs::Init() {
  static bool initialized = false;
  if (initialized) {
    return true;
  }
  // Initialize cache
  TbeUtils::LoadCache();

  // tbe_process
  PyObject *pTbeProcessModule = nullptr;
  pTbeProcessModule = PyImport_ImportModule(kTbeProcessModule);
  if (pTbeProcessModule == nullptr) {
    MS_LOG(ERROR) << "Failed to import [" << kTbeProcessModule << "] module.";
    return false;
  }

  pCreateTbeParallelCompilerFunc_ = PyObject_GetAttrString(pTbeProcessModule, kCreateTbeParallelCompilerFunc);
  if (pCreateTbeParallelCompilerFunc_ == nullptr) {
    MS_LOG(ERROR) << "Failed to transform opModule and FuncName to PyObject, opModule:[" << kTbeProcessModule
                  << "], FuncName:[" << kCreateTbeParallelCompilerFunc << "].";
    return false;
  }

  pTbeCompiler_ = PyEval_CallObject(pCreateTbeParallelCompilerFunc_, nullptr);
  if (pTbeCompiler_ == nullptr) {
    PyErr_Print();
    MS_EXCEPTION(ArgumentError) << "Failed to call function : create_parallel_compiler.";
    return false;
  }

  pOpSelectFormatFunc_ = PyObject_GetAttrString(pTbeProcessModule, kOpSelectFormatFunc);
  if (pOpSelectFormatFunc_ == nullptr) {
    MS_LOG(ERROR) << "Failed to transform opModule and FuncName to PyObject, opModule:[" << kTbeProcessModule
                  << "], FuncName:[" << kOpSelectFormatFunc << "].";
    return false;
  }

  pCheckSupportedFunc_ = PyObject_GetAttrString(pTbeProcessModule, kCheckSupportedFunc);
  if (pCheckSupportedFunc_ == nullptr) {
    MS_LOG(ERROR) << "Failed to transform opModule and FuncName to PyObject, opModule:[" << kTbeProcessModule
                  << "], FuncName:[" << kCheckSupportedFunc << "].";
    return false;
  }
  initialized = true;
  MS_LOG(INFO) << "TbePythonFuncs initialized Success.";
  return true;
}

std::string TbePythonFuncs::PyObjectToStr(PyObject *PyObj) {
  char *pChar = nullptr;
  std::string str_res;
  if (PyObj == nullptr) {
    MS_LOG(ERROR) << "Input parameter is nullptr.";
    return str_res;
  }
  PyObject *strArgs = PyObject_Str(PyObj);
  if (strArgs != nullptr) {
    (void)PyArg_Parse(strArgs, "s", &pChar);
  }
  if (pChar == nullptr) {
    MS_LOG(ERROR) << "pChar is nullptr.";
    return str_res;
  }
  str_res = pChar;
  return str_res;
}

std::string TbePythonFuncs::OpSelectFormat(const nlohmann::json &kernel_json) {
  PyObject *pArg = nullptr;
  PyObject *pRet = nullptr;
  std::string res_json_str;

  if (!Init()) {
    MS_LOG(ERROR) << "TbePythonFuncs Initialize Failed !";
    return res_json_str;
  }

  // assembly Args
  pArg = PyTuple_New(1);
  std::string json_str = kernel_json.dump();
  (void)PyTuple_SetItem(pArg, 0, Py_BuildValue("s", json_str.c_str()));
  if (pArg == nullptr) {
    MS_LOG(ERROR) << "Failed to generate parameter from kernel_json to PyObject.";
    return res_json_str;
  }

  // call functions
  if (pOpSelectFormatFunc_ == nullptr) {
    MS_LOG(ERROR) << "function is nullptr.";
    return res_json_str;
  }

  pRet = PyEval_CallObject(pOpSelectFormatFunc_, pArg);
  if (pRet == nullptr) {
    PyErr_Print();
    MS_EXCEPTION(ArgumentError) << "Failed to call function [" << kOpSelectFormatFunc
                                << "], function args:" << PyObjectToStr(pArg);
  }

  char *pstr = nullptr;
  (void)PyArg_Parse(pRet, "s", &pstr);
  res_json_str = pstr;
  if (res_json_str.compare(0, strlen(kTBEException), kTBEException) == 0) {
    MS_EXCEPTION(ArgumentError) << "Failed to call function [" << kOpSelectFormatFunc << "], " << res_json_str
                                << " ,function args:" << PyObjectToStr(pArg);
  }
  return res_json_str;
}

bool TbePythonFuncs::CheckSupported(const nlohmann::json &kernel_json) {
  PyObject *pArg = nullptr;
  PyObject *pRes = nullptr;
  bool ret = false;

  if (!Init()) {
    MS_LOG(ERROR) << "TbePythonFuncs Initialize Failed !";
    return ret;
  }
  // assembly Args
  pArg = PyTuple_New(1);
  std::string json_str = kernel_json.dump();
  PyObject *arg1 = Py_BuildValue("s", json_str.c_str());
  (void)PyTuple_SetItem(pArg, 0, arg1);
  if (pArg == nullptr) {
    MS_LOG(ERROR) << "Failed to generate parameter from kernel_json to PyObject.";
    return ret;
  }

  // call functions
  if (pCheckSupportedFunc_ == nullptr) {
    MS_LOG(ERROR) << "function is nullptr.";
    return ret;
  }

  pRes = PyEval_CallObject(pCheckSupportedFunc_, pArg);
  if (pRes == nullptr) {
    PyErr_Print();
    MS_EXCEPTION(ArgumentError) << "Failed to call function [" << kCheckSupportedFunc
                                << "], function args: " << PyObjectToStr(pArg);
  }
  if (PyBool_Check(pRes)) {
    ret = PyObject_IsTrue(pRes) != 0;
  } else {
    char *pstr = nullptr;
    (void)PyArg_Parse(pRes, "s", &pstr);
    std::string res_str = pstr;
    if (res_str.compare(0, strlen(kTBEException), kTBEException) == 0) {
      MS_EXCEPTION(ArgumentError) << "Failed to call function [" << kCheckSupportedFunc << "], " << res_str
                                  << ", function args: " << PyObjectToStr(pArg);
    }
  }

  return ret;
}

PyObject *TbePythonFuncs::TbeParallelCompiler() {
  if (!Init()) {
    MS_LOG(ERROR) << "TbePythonFuncs Initialize Failed !";
    return nullptr;
  }
  return pTbeCompiler_;
}
}  // namespace kernel
}  // namespace mindspore
