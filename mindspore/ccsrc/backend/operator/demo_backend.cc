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

#include "backend/operator/ops_backend_def.h"
#include "ops/op_def.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops {
class DemoFuncImpl : public OpFuncImpl {
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) const override {
    // Demo: return a demo shape
    return nullptr;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) const override {
    return nullptr;
  }
};

auto gDemoFuncImpl = DemoFuncImpl();
OpDef gDemo = {
  .name_ = kNameDemo,
  .args_ =
    {
      {.arg_name_ = "x", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
    },
  .returns_ =
    {
      {.arg_name_ = "output", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
    },
  .indexes_ =
    {
      {"x", 0},
    },
  .func_impl_ = &gDemoFuncImpl,
};

REGISTER_PRIMITIVE_OP_DEF(kNameDemo, &gDemo);
}  // namespace mindspore::ops
