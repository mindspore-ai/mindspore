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

#ifndef MINDSPORE_CORE_OPS_OP_DEF_H_
#define MINDSPORE_CORE_OPS_OP_DEF_H_
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "ir/dtype/type_id.h"
namespace mindspore::ops {

enum OP_DTYPE {
  DT_BOOL,
  DT_INT,
  DT_FLOAT,
  DT_TENSOR,
  DT_STR,
  DT_ARRAY_BOOL,
  DT_ARRAY_INT,
  DT_ARRAY_FLOAT,
  DT_ARRAY_TENSOR,
  DT_ARRAY_STR
};

struct OpArg {
  std::string arg_name_;
  OP_DTYPE arg_dtype_;
  bool as_init_arg_;  // true if this is a primitive init arg.
};

struct OpDef {
  std::string name_;
  std::vector<OpArg> args_;
  std::vector<OpArg> returns_;
};

using OpDefPtr = OpDef *;

OpDefPtr GetOpDef(const std::string &op_name);
}  // namespace mindspore::ops
#endif
