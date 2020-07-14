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

#ifndef MINDSPORE_CCSRC_KERNEL_TBE_TBE_PYTHON_FUNCS_H_
#define MINDSPORE_CCSRC_KERNEL_TBE_TBE_PYTHON_FUNCS_H_

#include <string>
#include <nlohmann/json.hpp>
#include "pybind11/stl.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
class TbePythonFuncs {
 public:
  TbePythonFuncs() = default;
  ~TbePythonFuncs() = default;
  static std::string OpSelectFormat(const nlohmann::json &kernel_json);
  static bool CheckSupported(const nlohmann::json &kernel_json);
  static PyObject *TbeParallelCompiler();

 private:
  static bool Init();
  static std::string PyObjectToStr(_object *PyObj);
  static PyObject *pCreateTbeParallelCompilerFunc_;
  static PyObject *pTbeCompiler_;
  static PyObject *pOpSelectFormatFunc_;
  static PyObject *pCheckSupportedFunc_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_TBE_TBE_PYTHON_FUNCS_H_
