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
#include "ops/batch_matmul.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
class TestBmm : public UT::Common {
 public:
  TestBmm() {}
  void SetUp() {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    origin_device_target_ = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  }
  void TearDown() {
    auto context_ptr = MsContext::GetInstance();
    if (context_ptr != nullptr) {
      context_ptr->set_param<std::string>(MS_CTX_DEVICE_TARGET, origin_device_target_);
    }
  }

 private:
  std::string origin_device_target_;
};

struct TestBmmSuccParams {
  bool transpose_a;
  bool transpose_b;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector y_shape;
  TypePtr y_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestBmmSucc : public TestBmm, public testing::WithParamInterface<TestBmmSuccParams> {};

INSTANTIATE_TEST_CASE_P(test_bmm_success, TestBmmSucc,
                        testing::Values(
                          TestBmmSuccParams{
                            false,
                            false,
                            ShapeVector{1, 3},
                            kFloat32,
                            ShapeVector{3, 1},
                            kFloat32,
                            ShapeVector{1, 1},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            false,
                            false,
                            ShapeVector{3, 1, 3},
                            kFloat32,
                            ShapeVector{3, 3, 1},
                            kFloat32,
                            ShapeVector{3, 1, 1},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            true,
                            false,
                            ShapeVector{3, 3, 1},
                            kFloat32,
                            ShapeVector{3, 3, 1},
                            kFloat32,
                            ShapeVector{3, 1, 1},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            false,
                            true,
                            ShapeVector{3, 1, 3},
                            kFloat32,
                            ShapeVector{3, 1, 3},
                            kFloat32,
                            ShapeVector{3, 1, 1},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            true,
                            true,
                            ShapeVector{3, 1, 3},
                            kFloat32,
                            ShapeVector{3, 3, 1},
                            kFloat32,
                            ShapeVector{3, 3, 3},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            false,
                            false,
                            ShapeVector{3, 1, 2, 4},
                            kFloat32,
                            ShapeVector{3, 4, 2},
                            kFloat32,
                            ShapeVector{3, 3, 2, 2},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            false,
                            false,
                            ShapeVector{1, 3, 2, 4},
                            kFloat32,
                            ShapeVector{3, 1, 4, 2},
                            kFloat32,
                            ShapeVector{3, 3, 2, 2},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            false,
                            false,
                            ShapeVector{3, 2, 4},
                            kFloat32,
                            ShapeVector{4, 2},
                            kFloat32,
                            ShapeVector{3, 2, 2},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            false,
                            false,
                            ShapeVector{2, 4},
                            kFloat32,
                            ShapeVector{2, 2, 4, 2},
                            kFloat32,
                            ShapeVector{2, 2, 2, 2},
                            kFloat32,
                          },
                          TestBmmSuccParams{
                            false,
                            false,
                            ShapeVector{6, 1, 8, 6, 1, 4},
                            kFloat32,
                            ShapeVector{6, 4, 1, 1, 8, 1, 4, 6},
                            kFloat32,
                            ShapeVector{6, 4, 6, 1, 8, 6, 1, 6},
                            kFloat32,
                          }));

TEST_P(TestBmmSucc, bmm) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameBatchMatMul);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("transpose_a", MakeValue<bool>(param.transpose_a));
  prim->set_attr("transpose_b", MakeValue<bool>(param.transpose_b));
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.x_shape), param.x_type);
  auto y = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.y_shape), param.y_type);
  auto expect = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.out_shape), param.out_type);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  auto out_abstract = BatchMatmulInfer(nullptr, prim, {x, y});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

struct TestBmmFailParams {
  bool transpose_a;
  bool transpose_b;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector y_shape;
  TypePtr y_type;
};

class TestBmmFail : public TestBmm, public testing::WithParamInterface<TestBmmFailParams> {};

INSTANTIATE_TEST_CASE_P(test_bmm_fail, TestBmmFail,
                        testing::Values(
                          TestBmmFailParams{
                            false,
                            false,
                            ShapeVector{1, 3},
                            kFloat32,
                            ShapeVector{2, 1},
                            kFloat32,
                          },
                          TestBmmFailParams{
                            false,
                            false,
                            ShapeVector{3, 1, 3},
                            kFloat32,
                            ShapeVector{3, 2, 1},
                            kFloat32,
                          },
                          TestBmmFailParams{
                            true,
                            false,
                            ShapeVector{3, 2, 1},
                            kFloat32,
                            ShapeVector{3, 3, 1},
                            kFloat32,
                          },
                          TestBmmFailParams{
                            false,
                            true,
                            ShapeVector{3, 1, 3},
                            kFloat32,
                            ShapeVector{3, 1, 2},
                            kFloat32,
                          },
                          TestBmmFailParams{
                            true,
                            true,
                            ShapeVector{3, 1, 3},
                            kFloat32,
                            ShapeVector{3, 3, 2},
                            kFloat32,
                          },
                          TestBmmFailParams{
                            false,
                            false,
                            ShapeVector{1},
                            kFloat32,
                            ShapeVector{1},
                            kFloat32,
                          },
                          TestBmmFailParams{
                            false,
                            false,
                            ShapeVector{4, 1, 3},
                            kFloat32,
                            ShapeVector{4, 3, 1},
                            kFloat16,
                          }));

TEST_P(TestBmmFail, bmm) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameBatchMatMul);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("transpose_a", MakeValue<bool>(param.transpose_a));
  prim->set_attr("transpose_b", MakeValue<bool>(param.transpose_b));
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.x_shape), param.x_type);
  auto y = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.y_shape), param.y_type);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_ANY_THROW(BatchMatmulInfer(nullptr, prim, {x, y}));
}

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
