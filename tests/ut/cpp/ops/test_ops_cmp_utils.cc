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
void ShapeCompare(const abstract::BaseShapePtr &output, const abstract::BaseShapePtr &expect) {
  ASSERT_NE(output, nullptr);
  ASSERT_NE(expect, nullptr);
  if (!(*output == *expect)) {
    MS_LOG(ERROR) << "Shape Compare Failed, start to print error info.";
    MS_LOG(ERROR) << "output [" << output->type_name() << "]: " << output->ToString();
    MS_LOG(ERROR) << "expect [" << expect->type_name() << "]: " << expect->ToString();
    ASSERT_TRUE(false);
  }
}

void TypeCompare(const TypePtr &output, const TypePtr &expect) {
  ASSERT_NE(output, nullptr);
  ASSERT_NE(expect, nullptr);
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
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x)};
  for (auto attr : param.attr_list) {
    auto attr_abs = attr->ToAbstract();
    input_args.push_back(std::move(attr_abs));
  }

  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.out_type);
  ASSERT_NE(expect_type, nullptr);

  ShapeCompare(infer_shape, expect_shape);
  TypeCompare(infer_type, expect_type);
}

void TestOpFuncImplWithMultiInputOpParams(const OpFuncImplPtr &infer_impl, const std::string &prim_name,
                                          const MultiInputOpParams &param) {
  auto primitive = std::make_shared<Primitive>(prim_name);
  ASSERT_NE(primitive, nullptr);
  ASSERT_TRUE(param.in_shape_array.size() == param.in_type_list.size());
  ASSERT_TRUE(!(param.in_shape_array.empty() && param.in_type_list.empty() && param.attr_list.empty()));
  std::vector<abstract::AbstractBasePtr> input_args;
  for (size_t idx = 0; idx < param.in_shape_array.size(); ++idx) {
    auto input = std::make_shared<abstract::AbstractTensor>(param.in_type_list[idx], param.in_shape_array[idx]);
    input_args.push_back(std::move(input));
  }
  for (auto attr : param.attr_list) {
    auto attr_abs = attr->ToAbstract();
    input_args.push_back(std::move(attr_abs));
  }

  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  ASSERT_TRUE(!param.out_shape_array.empty());
  ASSERT_TRUE(!param.out_type_list.empty());
  ASSERT_TRUE(param.out_shape_array.size() == param.out_type_list.size());

  abstract::BaseShapePtr expect_shape;
  TypePtr expect_type;
  if (param.out_shape_array.size() > 1) {
    std::vector<abstract::BaseShapePtr> shape_list;
    std::vector<TypePtr> type_list;
    for (size_t idx = 0; idx < param.out_shape_array.size(); ++idx) {
      auto shape = std::make_shared<abstract::TensorShape>(param.out_shape_array[idx]);
      auto type = std::make_shared<TensorType>(param.out_type_list[idx]);
      shape_list.push_back(std::move(shape));
      type_list.push_back(std::move(type));
    }
    expect_shape = std::make_shared<abstract::TupleShape>(shape_list);
    expect_type = std::make_shared<Tuple>(type_list);
  } else {
    expect_shape = std::make_shared<abstract::TensorShape>(param.out_shape_array[0]);
    expect_type = std::make_shared<TensorType>(param.out_type_list[0]);
  }

  ShapeCompare(infer_shape, expect_shape);
  TypeCompare(infer_type, expect_type);
}
}  // namespace ops
}  // namespace mindspore
