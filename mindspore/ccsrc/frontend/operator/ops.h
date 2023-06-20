/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"

namespace mindspore {
// namespace to support primitive operators
namespace prim {
ValuePtr GetPythonOps(const std::string &op_name,
                      const std::string &module_name = "mindspore._extends.parse.standard_method",
                      bool use_signature = false);
class UnpackGraphPrimitive : public Primitive {
 public:
  explicit UnpackGraphPrimitive(const bool &with_sens, const bool &need_unpack_args)
      : Primitive("UnpackGraph"), with_sens_in_args_(with_sens), need_unpack_args_(need_unpack_args) {}
  ~UnpackGraphPrimitive() override = default;
  MS_DECLARE_PARENT(UnpackGraphPrimitive, Primitive)
  bool with_sens_in_args() const { return with_sens_in_args_; }
  bool need_unpack_args() const { return need_unpack_args_; }

 private:
  bool with_sens_in_args_;
  bool need_unpack_args_;
};
using UnpackGraphPrimitivePtr = std::shared_ptr<UnpackGraphPrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_OPS_H_
