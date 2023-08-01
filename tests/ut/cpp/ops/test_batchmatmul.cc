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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/test_ops.h"
#include "ops/batch_matmul.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
struct BmmParams {
  bool transpose_a;
  bool transpose_b;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector y_shape;
  TypePtr y_type;
  bool is_success;
  ShapeVector out_shape;
  TypePtr out_type;
};

struct BmmShapes {
  bool transpose_a;
  bool transpose_b;
  ShapeVector x_shape;
  ShapeVector y_shape;
  bool is_success;
  ShapeVector out_shape;
};

struct BmmTypes {
  TypePtr x_type;
  TypePtr y_type;
  bool is_success;
  TypePtr out_type;
  TypePtr cast_type;
};

class TestBmm : public TestOps, public testing::WithParamInterface<std::tuple<BmmShapes, BmmTypes>> {};

TEST_P(TestBmm, bmm) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &type_param = std::get<1>(GetParam());
  auto prim = std::make_shared<Primitive>(kNameBatchMatMul);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("transpose_a", MakeValue<bool>(shape_param.transpose_a));
  prim->set_attr("transpose_b", MakeValue<bool>(shape_param.transpose_b));
  if (type_param.cast_type != nullptr) {
    prim->set_attr("cast_type", type_param.cast_type);
  }
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(shape_param.x_shape), type_param.x_type);
  auto y = abstract::MakeAbstract(std::make_shared<abstract::Shape>(shape_param.y_shape), type_param.y_type);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  if (shape_param.is_success && type_param.is_success) {
    auto expect = abstract::MakeAbstract(std::make_shared<abstract::Shape>(shape_param.out_shape), type_param.out_type);
    auto out_abstract = BatchMatmulInfer(nullptr, prim, {x, y});
    ASSERT_NE(out_abstract, nullptr);
    ASSERT_TRUE(*out_abstract == *expect);
  } else {
    ASSERT_ANY_THROW(BatchMatmulInfer(nullptr, prim, {x, y}));
  }
}

INSTANTIATE_TEST_CASE_P(
  TestBmm, TestBmm,
  testing::Combine(testing::ValuesIn({
                     BmmShapes{false, false, {1, 3}, {3, 1}, true, {1, 1}},
                     BmmShapes{false, false, {3, 1, 3}, {3, 3, 1}, true, {3, 1, 1}},
                     BmmShapes{true, false, {3, 3, 1}, {3, 3, 1}, true, {3, 1, 1}},
                     BmmShapes{false, true, {3, 1, 3}, {3, 1, 3}, true, {3, 1, 1}},
                     BmmShapes{true, true, {3, 1, 3}, {3, 3, 1}, true, {3, 3, 3}},
                     BmmShapes{false, false, {3, 1, 2, 4}, {3, 4, 2}, true, {3, 3, 2, 2}},
                     BmmShapes{false, false, {1, 3, 2, 4}, {3, 1, 4, 2}, true, {3, 3, 2, 2}},
                     BmmShapes{false, false, {3, 2, 4}, {4, 2}, true, {3, 2, 2}},
                     BmmShapes{false, false, {2, 4}, {2, 2, 4, 2}, true, {2, 2, 2, 2}},
                     BmmShapes{
                       false, false, {6, 1, 8, 6, 1, 4}, {6, 4, 1, 1, 8, 1, 4, 6}, true, {6, 4, 6, 1, 8, 6, 1, 6}},
                     BmmShapes{false, false, {1, 3}, {2, 1}, false},        // 3 != 2
                     BmmShapes{false, false, {3, 1, 3}, {3, 2, 1}, false},  // 3 != 2
                     BmmShapes{true, false, {3, 2, 1}, {3, 3, 1}, false},   // 3 != 2
                     BmmShapes{false, true, {3, 1, 3}, {3, 1, 2}, false},
                     BmmShapes{true, true, {3, 1, 3}, {3, 3, 2}, false},  // 1 != 2
                     BmmShapes{false, false, {1}, {1}, false},            // rank must be >= 2
                   }),
                   testing::ValuesIn({
                     BmmTypes{kFloat32, kFloat32, true, kFloat32},
                     BmmTypes{kFloat16, kFloat16, true, kFloat16},
                     BmmTypes{kInt32, kInt32, true, kInt32},
                     BmmTypes{kInt8, kInt8, true, kInt32},                    // int8 * int8 = int32
                     BmmTypes{kInt8, kInt8, true, kInt16, kInt16},            // cast_type = int16
                     BmmTypes{kFloat16, kFloat16, true, kFloat32, kFloat32},  // cast_type = fp32
                     BmmTypes{kFloat32, kFloat32, true, kFloat16, kFloat16},  // cast_type = fp16
                     BmmTypes{kFloat16, kFloat32, false},
                     BmmTypes{kInt32, kFloat32, false},
                     BmmTypes{kInt32, kInt8, false},
                   })));

/// Feature: BatchMatmul infer
/// Description: primitive has no attr "transpose_a"
/// Expectation: infer will fail
TEST_F(TestBmm, test_bmm_no_transpose_a_attr_fail) {
  auto prim = std::make_shared<Primitive>(kNameBatchMatMul);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("transpose_b", MakeValue<bool>(false));
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(std::vector<int64_t>{1, 3}), kFloat32);
  auto y = abstract::MakeAbstract(std::make_shared<abstract::Shape>(std::vector<int64_t>{3, 1}), kFloat32);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_ANY_THROW(BatchMatmulInfer(nullptr, prim, {x, y}));
}

/// Feature: BatchMatmul infer
/// Description: primitive has no attr "transpose_b"
/// Expectation: infer will fail
TEST_F(TestBmm, test_bmm_no_transpose_b_attr_fail) {
  auto prim = std::make_shared<Primitive>(kNameBatchMatMul);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("transpose_a", MakeValue<bool>(false));
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(std::vector<int64_t>{1, 3}), kFloat32);
  auto y = abstract::MakeAbstract(std::make_shared<abstract::Shape>(std::vector<int64_t>{3, 1}), kFloat32);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_ANY_THROW(BatchMatmulInfer(nullptr, prim, {x, y}));
}
}  // namespace ops
}  // namespace mindspore
