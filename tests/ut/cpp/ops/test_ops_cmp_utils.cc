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
#include "ops/test_ops_cmp_utils.h"
#include <memory>
#include <vector>
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
static void ShapeCompare(const abstract::BaseShapePtr &output, const abstract::BaseShapePtr &expect) {
  if (!(*output == *expect)) {
    MS_LOG(ERROR) << "Shape Compare Failed, start to print error info.";
    MS_LOG(ERROR) << "output [" << output->type_name() << "]: " << output->ToString();
    MS_LOG(ERROR) << "expect [" << expect->type_name() << "]: " << expect->ToString();
    ASSERT_TRUE(false);
  }
}

static void TypeCompare(const TypePtr &output, const TypePtr &expect) {
  if (!(*output == *expect)) {
    MS_LOG(ERROR) << "Type Compare Failed, start to print error info.";
    MS_LOG(ERROR) << "output [" << output->type_name() << "]: " << output->ToString();
    MS_LOG(ERROR) << "expect [" << expect->type_name() << "]: " << expect->ToString();
    ASSERT_TRUE(false);
  }
}

void TestOpFuncImplWithEltwiseOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                       const EltwiseOpParams &param) {
  auto primitive = std::make_shared<Primitive>(prim_name);
  ASSERT_NE(primitive, nullptr);
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  std::vector<abstract::AbstractBasePtr>input_args{std::move(x)};

  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_type, nullptr);

  ShapeCompare(infer_shape, expect_shape);
  TypeCompare(infer_type, expect_type);
}
}  // namespace ops
}  // namespace mindspore
