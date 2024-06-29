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
#include "ops_func_impl/op_func_impl.h"
namespace mindspore::ops {

enum OP_DTYPE : int64_t {
  DT_BEGIN = 0,
  DT_BOOL,
  DT_INT,
  DT_FLOAT,
  DT_NUMBER,
  DT_TENSOR,
  DT_STR,
  DT_ANY,
  DT_TUPLE_BOOL,
  DT_TUPLE_INT,
  DT_TUPLE_FLOAT,
  DT_TUPLE_NUMBER,
  DT_TUPLE_TENSOR,
  DT_TUPLE_STR,
  DT_TUPLE_ANY,
  DT_LIST_BOOL,
  DT_LIST_INT,
  DT_LIST_FLOAT,
  DT_LIST_NUMBER,
  DT_LIST_TENSOR,
  DT_LIST_STR,
  DT_LIST_ANY,
  DT_TYPE,
  DT_END,
};

struct OpInputArg {
  std::string arg_name_;
  OP_DTYPE arg_dtype_;
  bool as_init_arg_;  // true if this is a primitive init arg.
  std::string arg_handler_;
  std::vector<OP_DTYPE> cast_dtype_;
  bool is_optional_;
};

struct OpOutputArg {
  std::string arg_name_;
  OP_DTYPE arg_dtype_;
  int64_t inplace_input_index_;
};

struct OpDef {
  std::string name_;
  std::vector<OpInputArg> args_;
  std::vector<OpOutputArg> returns_;
  std::vector<Signature> signatures_;
  std::unordered_map<std::string, size_t> indexes_;
  OpFuncImpl &func_impl_;
  bool enable_dispatch_;
  bool is_view_;
};

using OpDefPtr = OpDef *;

MS_CORE_API OpDefPtr GetOpDef(const std::string &op_name);
MS_CORE_API void AddOpDef(const std::string &op_name, const OpDefPtr op_def);
MS_CORE_API bool IsPrimitiveFunction(const std::string &op_name);

class OpDefRegHelper {
 public:
  OpDefRegHelper(const std::string &op_name, const OpDefPtr op_def) { AddOpDef(op_name, op_def); }
  ~OpDefRegHelper() = default;
};

#define REGISTER_PRIMITIVE_OP_DEF(op_name, op_def) \
  static auto op_def_helper_##op_name = OpDefRegHelper(op_name, op_def);
}  // namespace mindspore::ops
#endif
