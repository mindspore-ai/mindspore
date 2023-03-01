/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_DATA_CONVERTER_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_DATA_CONVERTER_H_

#include <deque>
#include <memory>
#include <vector>
#include <string>
#include "utils/ordered_map.h"
#include "utils/hash_map.h"
#include "pipeline/jit/parse/parse_base.h"
#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parse {
// data convert for parse
namespace data_converter {
void CacheObjectValue(const std::string &obj_key, const ValuePtr &data);
bool GetObjectValue(const std::string &obj_key, ValuePtr *const data);

void SetObjGraphValue(const std::string &obj_key, const FuncGraphPtr &data);

const mindspore::OrderedMap<std::string, std::vector<FuncGraphPtr>> &GetObjGraphs();

std::vector<std::string> GetObjKey(const py::object &obj);
ResolveTypeDef GetObjType(const py::object &obj);
ClassInstanceTypeDef GetClassInstanceType(const py::object &obj);

bool IsCellInstance(const py::object &obj);
bool IsNumpyArrayInstance(const py::object &obj);
bool IsMsClassInstance(const py::object &obj);
bool IsJITForbiddenAPI(const py::object &obj);
bool IsClassType(const py::object &obj);
py::object CreatePythonObject(const py::object &type, const py::tuple &args_kwargs);
py::object CallPythonScript(const py::object &script, const py::tuple &args_kwargs);
py::set GetPythonScriptIds(const py::object &script);
void MakeProperNameToFuncGraph(const FuncGraphPtr &func_graph, std::string name);
ValuePtr PyDataToValue(const py::object &obj);
ValuePtr PyDataToStubNode(const py::object &obj);
void ClearObjectCache();
}  // namespace data_converter

FuncGraphPtr ConvertToBpropCut(const py::object &obj);
}  // namespace parse
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_DATA_CONVERTER_H_
