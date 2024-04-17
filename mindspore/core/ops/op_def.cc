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
}  // namespace mindspore::ops
