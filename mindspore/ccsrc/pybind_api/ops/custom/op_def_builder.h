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

#ifndef MINDSPORE_OPS_OP_DEF_PY_FUNC_OP_DEF_BUILDER_H_
#define MINDSPORE_OPS_OP_DEF_PY_FUNC_OP_DEF_BUILDER_H_
#include "ops/op_def.h"
#include "infer/ops_func_impl/py_func.h"
#include "mindapi/base/macros.h"

namespace mindspore::ops {
class PyFuncOpDefBuilder {
 public:
  explicit PyFuncOpDefBuilder(const std::string &name) : name_(name), op_def_(NewOp()) { op_def_->name_ = name_; }
  PyFuncOpDefBuilder &Arg(const std::string &arg_name, const std::string &role, const std::string &obj_type);
  void Register();

 private:
  size_t AddInputImpl(const std::string &arg_name, const std::string &obj_type);
  void AddOutputImpl(const std::string &arg_name, const std::string &obj_type, int64_t input_index);
  static OpDef *NewOp();
  std::string name_;
  OpDef *op_def_;
};
}  // namespace mindspore::ops
#endif  // MINDSPORE_OPS_OP_DEF_PY_FUNC_OP_DEF_BUILDER_H_
