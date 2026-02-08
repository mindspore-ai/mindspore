/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "pybind_api/ops/custom/op_def_builder.h"

namespace mindspore::ops {
namespace {
// Convert a string representation of a type to its corresponding enum value.
OP_DTYPE TypeStrToEnum(const std::string &s) {
  static const std::unordered_map<std::string, OP_DTYPE> kMap = {{"bool", DT_BOOL},
                                                                 {"int", DT_INT},
                                                                 {"float", DT_FLOAT},
                                                                 {"number", DT_NUMBER},
                                                                 {"tensor", DT_TENSOR},
                                                                 {"str", DT_STR},
                                                                 {"any", DT_ANY},
                                                                 {"type", DT_TYPE},
                                                                 {"none", DT_NONE},
                                                                 {"tuple_bool", DT_TUPLE_BOOL},
                                                                 {"tuple_int", DT_TUPLE_INT},
                                                                 {"tuple_float", DT_TUPLE_FLOAT},
                                                                 {"tuple_number", DT_TUPLE_NUMBER},
                                                                 {"tuple_tensor", DT_TUPLE_TENSOR},
                                                                 {"tuple_str", DT_TUPLE_STR},
                                                                 {"tuple_any", DT_TUPLE_ANY},
                                                                 {"list_bool", DT_LIST_BOOL},
                                                                 {"list_int", DT_LIST_INT},
                                                                 {"list_float", DT_LIST_FLOAT},
                                                                 {"list_number", DT_LIST_NUMBER},
                                                                 {"list_tensor", DT_LIST_TENSOR},
                                                                 {"list_str", DT_LIST_STR},
                                                                 {"list_any", DT_LIST_ANY}};

  auto it = kMap.find(s);
  if (it != kMap.end()) {
    return it->second;
  }
  MS_LOG(EXCEPTION) << "TypeStrToEnum received unknown dtype string " << s;
}

PyFuncInferImpl gCustomPyFuncInferImpl;
static const OpDef gOpDefTemplate = {
  /* name_            */ "",
  /* args_            */ {},
  /* returns_         */ {},
  /* signatures_      */ {},
  /* indexes_         */ {},
  /* func_impl_       */ gCustomPyFuncInferImpl,
  /* enable_dispatch_ */ false,
  /* is_view_         */ false,
  /* is_graph_view_   */ false};
}  // namespace

// Add an argument (input or output) to the operator definition.
PyFuncOpDefBuilder &PyFuncOpDefBuilder::Arg(const std::string &arg_name, const std::string &role,
                                            const std::string &obj_type) {
  if (role == "input") {
    (void)AddInputImpl(arg_name, obj_type);
  } else if (role == "output") {
    AddOutputImpl(arg_name, obj_type, 0);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported role " << role << ", expected input or output";
  }
  return *this;
}

// Register the operator definition to the global registry.
void PyFuncOpDefBuilder::Register() {
  if (mindspore::ops::GetOpDef(name_) != nullptr) {
    MS_LOG(EXCEPTION) << "OpDef named " << name_ << " already exists";
  }
  mindspore::ops::AddOpDef(name_, op_def_);
}

// Add an input argument to the operator definition.
size_t PyFuncOpDefBuilder::AddInputImpl(const std::string &arg_name, const std::string &obj_type) {
  if (op_def_->indexes_.find(arg_name) != op_def_->indexes_.end()) {
    MS_LOG(EXCEPTION) << "Input arg " << arg_name << " already exists";
  }

  OpInputArg arg;
  arg.arg_name_ = arg_name;
  arg.arg_dtype_ = TypeStrToEnum(obj_type);
  arg.as_init_arg_ = false;
  arg.arg_handler_.clear();
  arg.cast_dtype_.clear();
  arg.is_optional_ = false;

  size_t idx = op_def_->args_.size();
  op_def_->args_.push_back(std::move(arg));
  op_def_->indexes_[arg_name] = idx;
  op_def_->signatures_.emplace_back(Signature(arg_name, SignatureEnumRW::kRWDefault,
                                              SignatureEnumKind::kKindPositionalKeyword, nullptr,
                                              SignatureEnumDType::kDTypeEmptyDefaultValue));
  return idx;
}

// Add an output argument to the operator definition.
void PyFuncOpDefBuilder::AddOutputImpl(const std::string &arg_name, const std::string &obj_type, int64_t input_index) {
  OpOutputArg arg;
  arg.arg_name_ = arg_name;
  arg.arg_dtype_ = TypeStrToEnum(obj_type);
  arg.inplace_input_index_ = -1;

  op_def_->returns_.push_back(std::move(arg));
}

// Create a new operator definition from the template.
OpDef *PyFuncOpDefBuilder::NewOp() {
  static std::vector<std::unique_ptr<OpDef>> pool;
  pool.emplace_back(std::make_unique<OpDef>(gOpDefTemplate));
  return pool.back().get();
}
}  // namespace mindspore::ops
