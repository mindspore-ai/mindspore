/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/optimizer/helper.h"
#include "ops/test_ops.h"
#include "ops/ops_func_impl/xlogy.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"

namespace mindspore {
namespace ops {

struct XlogyTensorShape {
  std::vector<int64_t> x_shape;
  std::vector<int64_t> y_shape;
  std::vector<int64_t> out_shape;
};

struct XlogyTensorType {
  TypePtr x_type;
  TypePtr y_type;
  TypePtr out_type;
};

class TestXlogyTensor : public TestOps,
                        public testing::WithParamInterface<std::tuple<XlogyTensorShape, XlogyTensorType>> {};

TEST_P(TestXlogyTensor, XlogyTensor_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  XlogyFuncImpl xlogy_tensor_func_impl;
  auto prim = std::make_shared<Primitive>("XLogy");
  auto x = std::make_shared<abstract::AbstractTensor>(dtype_param.x_type, shape_param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(dtype_param.y_type, shape_param.y_shape);
  auto expect_shape = std::make_shared<abstract::TensorShape>(shape_param.out_shape);
  auto expect_dtype = std::make_shared<TensorType>(dtype_param.out_type);

  auto out_shape = xlogy_tensor_func_impl.InferShape(prim, {x, y});
  ASSERT_TRUE(*out_shape == *expect_shape);
  auto out_dtype = xlogy_tensor_func_impl.InferType(prim, {x, y});
  ASSERT_TRUE(*out_dtype == *expect_dtype);
}

class TestXlogyTensorSimpleInfer : public TestOps,
                                   public testing::WithParamInterface<std::tuple<XlogyTensorShape, XlogyTensorType>> {};

TEST_P(TestXlogyTensorSimpleInfer, simple_infer) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());
  XlogyFuncImpl xlogy_tensor_func_impl;

  auto prim = std::make_shared<Primitive>("XLogy");
  ASSERT_NE(prim, nullptr);
  auto x = std::make_shared<tensor::BaseTensor>(dtype_param.x_type->type_id(), shape_param.x_shape);
  ASSERT_NE(x, nullptr);
  auto y = std::make_shared<tensor::BaseTensor>(dtype_param.y_type->type_id(), shape_param.y_shape);
  ASSERT_NE(y, nullptr);
  ValuePtrList input_values;
  input_values.push_back(std::move(x));
  input_values.push_back(std::move(y));

  auto expect_shape = ShapeArray{shape_param.out_shape};
  auto expect_type = TypePtrList{dtype_param.out_type};

  auto output_shape = xlogy_tensor_func_impl.InferShape(prim, input_values);
  auto output_type = xlogy_tensor_func_impl.InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

auto XlogyTensorOpShapeTestCases = testing::ValuesIn({XlogyTensorShape{{10}, {}, {10}},
                                                      XlogyTensorShape{{10, 1, 2}, {}, {10, 1, 2}},
                                                      XlogyTensorShape{{10, 4, 2}, {}, {10, 4, 2}},
                                                      XlogyTensorShape{{10, 1, -1}, {}, {10, 1, -1}},
                                                      XlogyTensorShape{{-2}, {}, {-2}},
                                                      XlogyTensorShape{{}, {10}, {10}},
                                                      XlogyTensorShape{{}, {10, 1, 2}, {10, 1, 2}},
                                                      XlogyTensorShape{{}, {10, 4, 2}, {10, 4, 2}},
                                                      XlogyTensorShape{{}, {10, 1, -1}, {10, 1, -1}},
                                                      XlogyTensorShape{{}, {-2}, {-2}},
                                                      XlogyTensorShape{{4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
                                                      XlogyTensorShape{{2, 1, 4, 5, 6, 9}, {9}, {2, 1, 4, 5, 6, 9}},
                                                      XlogyTensorShape{{2, 3, 4, -1}, {2, 3, 4, 5}, {2, 3, 4, -1}},
                                                      XlogyTensorShape{{2, 3, 4, -1}, {-1, -1, 4, 5}, {-1, -1, 4, -1}},
                                                      XlogyTensorShape{{2, 1, 4, -1}, {-1, -1, 4, 5}, {-1, -1, 4, -1}},
                                                      XlogyTensorShape{{2, 1, 4, 5, 6, 9}, {-2}, {-2}},
                                                      XlogyTensorShape{{2, 1, 4, 5, -1, 9}, {-2}, {-2}},
                                                      XlogyTensorShape{{-2}, {2, 1, 4, 5, 6, 9}, {-2}},
                                                      XlogyTensorShape{{-2}, {2, 1, 4, 5, -1, 9}, {-2}},
                                                      XlogyTensorShape{{-2}, {-2}, {-2}}});

auto XlogyTensorOpSimpleInferShapeTestCases =
  testing::ValuesIn({XlogyTensorShape{{10}, {}, {10}}, XlogyTensorShape{{10, 1, 2}, {}, {10, 1, 2}},
                     XlogyTensorShape{{10, 4, 2}, {}, {10, 4, 2}}, XlogyTensorShape{{}, {10}, {10}},
                     XlogyTensorShape{{}, {10, 1, 2}, {10, 1, 2}}, XlogyTensorShape{{}, {10, 4, 2}, {10, 4, 2}},
                     XlogyTensorShape{{4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
                     XlogyTensorShape{{2, 1, 4, 5, 6, 9}, {9}, {2, 1, 4, 5, 6, 9}}});

auto XlogyTensorOpTypeTestCases =
  testing::ValuesIn({XlogyTensorType{kFloat16, kFloat16, kFloat16}, XlogyTensorType{kFloat32, kFloat16, kFloat32},
                     XlogyTensorType{kFloat32, kFloat64, kFloat64}, XlogyTensorType{kFloat64, kInt64, kFloat64},
                     XlogyTensorType{kFloat32, kBool, kFloat32}, XlogyTensorType{kBool, kUInt8, kFloat32},
                     XlogyTensorType{kInt64, kUInt8, kFloat32}, XlogyTensorType{kFloat32, kInt16, kFloat32},
                     XlogyTensorType{kFloat32, kFloat32, kFloat32}, XlogyTensorType{kFloat64, kFloat64, kFloat64}});

INSTANTIATE_TEST_CASE_P(TestXlogyTensor, TestXlogyTensor,
                        testing::Combine(XlogyTensorOpShapeTestCases, XlogyTensorOpTypeTestCases));

INSTANTIATE_TEST_CASE_P(TestXlogyTensorSimpleInfer, TestXlogyTensorSimpleInfer,
                        testing::Combine(XlogyTensorOpSimpleInferShapeTestCases, XlogyTensorOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
