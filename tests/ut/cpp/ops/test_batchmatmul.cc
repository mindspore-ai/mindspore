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

class TestBmm : public TestOps, public testing::WithParamInterface<BmmParams> {};

TEST_P(TestBmm, bmm) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameBatchMatMul);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("transpose_a", MakeValue<bool>(param.transpose_a));
  prim->set_attr("transpose_b", MakeValue<bool>(param.transpose_b));
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.x_shape), param.x_type);
  auto y = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.y_shape), param.y_type);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  if (param.is_success) {
    auto expect = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.out_shape), param.out_type);
    auto out_abstract = BatchMatmulInfer(nullptr, prim, {x, y});
    ASSERT_NE(out_abstract, nullptr);
    ASSERT_TRUE(*out_abstract == *expect);
  } else {
    ASSERT_ANY_THROW(BatchMatmulInfer(nullptr, prim, {x, y}));
  }
}

INSTANTIATE_TEST_CASE_P(
  TestBmm, TestBmm,
  testing::Values(BmmParams{false, false, {1, 3}, kFloat32, {3, 1}, kFloat32, true, {1, 1}, kFloat32},
                  BmmParams{false, false, {3, 1, 3}, kFloat32, {3, 3, 1}, kFloat32, true, {3, 1, 1}, kFloat32},
                  BmmParams{true, false, {3, 3, 1}, kFloat32, {3, 3, 1}, kFloat32, true, {3, 1, 1}, kFloat32},
                  BmmParams{false, true, {3, 1, 3}, kFloat32, {3, 1, 3}, kFloat32, true, {3, 1, 1}, kFloat32},
                  BmmParams{true, true, {3, 1, 3}, kFloat32, {3, 3, 1}, kFloat32, true, {3, 3, 3}, kFloat32},
                  BmmParams{false, false, {3, 1, 2, 4}, kFloat32, {3, 4, 2}, kFloat32, true, {3, 3, 2, 2}, kFloat32},
                  BmmParams{false, false, {1, 3, 2, 4}, kFloat32, {3, 1, 4, 2}, kFloat32, true, {3, 3, 2, 2}, kFloat32},
                  BmmParams{false, false, {3, 2, 4}, kFloat32, {4, 2}, kFloat32, true, {3, 2, 2}, kFloat32},
                  BmmParams{false, false, {2, 4}, kFloat32, {2, 2, 4, 2}, kFloat32, true, {2, 2, 2, 2}, kFloat32},
                  BmmParams{false,
                            false,
                            {6, 1, 8, 6, 1, 4},
                            kFloat32,
                            {6, 4, 1, 1, 8, 1, 4, 6},
                            kFloat32,
                            true,
                            {6, 4, 6, 1, 8, 6, 1, 6},
                            kFloat32},
                  BmmParams{false, false, {1, 3}, kFloat32, {2, 1}, kFloat32, false},
                  BmmParams{false, false, {3, 1, 3}, kFloat32, {3, 2, 1}, kFloat32, false},
                  BmmParams{true, false, {3, 2, 1}, kFloat32, {3, 3, 1}, kFloat32, false},
                  BmmParams{false, true, {3, 1, 3}, kFloat32, {3, 1, 2}, kFloat32, false},
                  BmmParams{true, true, {3, 1, 3}, kFloat32, {3, 3, 2}, kFloat32, false},
                  BmmParams{false, false, {1}, kFloat32, {1}, kFloat32, false},
                  BmmParams{false, false, {4, 1, 3}, kFloat32, {4, 3, 1}, kFloat16, false}));

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
