/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_NP_DTYPE_NP_DTYPES_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_NP_DTYPE_NP_DTYPES_H_
#include <string>
#include "pybind11/pybind11.h"

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef BUILDING_NP_DTYPE_DLL
#define NP_DTYPE_API __declspec(dllexport)
#else
#define NP_DTYPE_API __declspec(dllimport)
#endif
#else
#define NP_DTYPE_API __attribute__((visibility("default")))
#endif

namespace py = pybind11;
namespace mindspore {
namespace np_dtypes {
std::string NP_DTYPE_API GetNumpyVersion();
std::string NP_DTYPE_API GetMinimumSupportedNumpyVersion();
bool NP_DTYPE_API NumpyVersionValid(std::string version);
}  // namespace np_dtypes
int NP_DTYPE_API GetBFloat16NpDType();
void NP_DTYPE_API RegNumpyTypes(py::module *m);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_NP_DTYPE_NP_DTYPES_H_
