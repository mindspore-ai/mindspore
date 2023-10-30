/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/op_def.h"
namespace mindspore::ops {
extern std::unordered_map<std::string, OpDefPtr> gOpDefTable;  // defined in gen_ops_def.cc
OpDefPtr GetOpDef(const std::string &op_name) {
  auto it = gOpDefTable.find(op_name);
  if (it != gOpDefTable.end()) {
    return it->second;
  }
  return nullptr;
}

void AddOpDef(const std::string &op_name, const OpDefPtr op_def) { (void)gOpDefTable.emplace(op_name, op_def); }

bool IsPrimitiveFunction(const std::string &op_name) { return GetOpDef(op_name) != nullptr; }

std::string EnumToString(OP_DTYPE dtype) {
  static const std::unordered_map<OP_DTYPE, std::string> kEnumToStringMap = {
    {DT_BOOL, "bool"},
    {DT_INT, "int"},
    {DT_FLOAT, "float"},
    {DT_NUMBER, "Number"},
    {DT_TENSOR, "Tensor"},
    {DT_STR, "string"},
    {DT_ANY, "Any"},
    {DT_TUPLE_BOOL, "tuple of bool"},
    {DT_TUPLE_INT, "tuple of int"},
    {DT_TUPLE_FLOAT, "tuple of float"},
    {DT_TUPLE_NUMBER, "tuple of Number"},
    {DT_TUPLE_TENSOR, "tuple of Tensor"},
    {DT_TUPLE_STR, "tuple of string"},
    {DT_TUPLE_ANY, "tuple of Any"},
    {DT_LIST_BOOL, "list of bool"},
    {DT_LIST_INT, "list of int"},
    {DT_LIST_FLOAT, "list of float"},
    {DT_LIST_NUMBER, "list of number"},
    {DT_LIST_TENSOR, "list of tensor"},
    {DT_LIST_STR, "list of string"},
    {DT_LIST_ANY, "list of Any"},
  };

  auto it = kEnumToStringMap.find(dtype);
  if (it == kEnumToStringMap.end()) {
    MS_LOG(ERROR) << "Failed to map Enum[" << dtype << "] to String.";
  }
  return it->second;
}
}  // namespace mindspore::ops
