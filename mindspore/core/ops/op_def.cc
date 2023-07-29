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
#ifdef ENABLE_GEN_CODE
extern std::unordered_map<std::string, OpDefPtr> gOpDefTable;  // defined in gen_ops_def.cc
#endif
OpDefPtr GetOpDef(const std::string &op_name) {
#ifdef ENABLE_GEN_CODE
  auto it = gOpDefTable.find(op_name);
  if (it != gOpDefTable.end()) {
    return it->second;
  }
#endif
  return nullptr;
}
}  // namespace mindspore::ops
