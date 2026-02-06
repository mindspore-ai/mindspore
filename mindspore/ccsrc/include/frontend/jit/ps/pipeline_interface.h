/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_PIPELINE_INTERFACE_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_PIPELINE_INTERFACE_H_

#include <utility>
#include <string>
#include <memory>

#include "pybind11/pybind11.h"

#include "base/base.h"
#include "base/base_ref.h"
#include "include/utils/visible.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace pipeline {
namespace py = pybind11;

FRONTEND_EXPORT py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs);

FRONTEND_EXPORT FuncGraphPtr LoadMindIR(const std::string &file_name, const char *dec_key, const size_t key_len,
                                        const std::string &dec_mode, const py::object decrypt = py::none());
FRONTEND_EXPORT FuncGraphPtr SplitMindIR(const std::string &file_name);
FRONTEND_EXPORT FuncGraphPtr SplitDynamicMindIR(const std::string &file_name, size_t device_num, size_t rank_id,
                                                bool sapp);

FRONTEND_EXPORT void InitPipeline();
FRONTEND_EXPORT bool RunJitPipeline();
FRONTEND_EXPORT std::string DumpFuncGraph(const py::object &obj);
FRONTEND_EXPORT void PreJit(const py::object &args, const py::object &kwargs);

FRONTEND_EXPORT py::object BaseRefToPyDataWithUserData(const BaseRef &value, const abstract::AbstractBasePtr &abs);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_JIT_PS_PIPELINE_INTERFACE_H_
