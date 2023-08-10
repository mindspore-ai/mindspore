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
#include "ops/eps.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
class TestEps : public UT::Common {
 public:
  TestEps() {}
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

struct TestEpsSuccParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestEpsSucc : public TestEps, public testing::WithParamInterface<TestEpsSuccParams> {};

INSTANTIATE_TEST_CASE_P(test_eps_success, TestEpsSucc,
                        testing::Values(
                          TestEpsSuccParams{
                            ShapeVector{2, 3},
                            kFloat16,
                            ShapeVector{2, 3},
                            kFloat16,
                          },
                          TestEpsSuccParams{
                            ShapeVector{3, 2, 4},
                            kFloat32,
                            ShapeVector{3, 2, 4},
                            kFloat32,
                          },
                          TestEpsSuccParams{
                            ShapeVector{4, 1, 3, 2},
                            kFloat64,
                            ShapeVector{4, 1, 3, 2},
                            kFloat64,
                          },
                          TestEpsSuccParams{
                            ShapeVector{6, 4, 1, 1, 8, 1, 4, 6},
                            kFloat32,
                            ShapeVector{6, 4, 1, 1, 8, 1, 4, 6},
                            kFloat32,
                          }));

TEST_P(TestEpsSucc, eps) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameEps);
  ASSERT_NE(prim, nullptr);
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.x_shape), param.x_type);
  auto expect = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.out_shape), param.out_type);
  ASSERT_NE(x, nullptr);
  auto out_abstract = EpsInfer(nullptr, prim, {x});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

struct TestEpsFailParams {
  ShapeVector x_shape;
  TypePtr x_type;
};

class TestEpsFail : public TestEps, public testing::WithParamInterface<TestEpsFailParams> {};

INSTANTIATE_TEST_CASE_P(test_eps_fail, TestEpsFail,
                        testing::Values(
                          TestEpsFailParams{
                            ShapeVector{2, 3, 4},
                            kInt32,
                          },
                          TestEpsFailParams{
                            ShapeVector{3, 2, 4, 6},
                            kUInt16,
                          }));

TEST_P(TestEpsFail, eps) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameEps);
  ASSERT_NE(prim, nullptr);
  auto x = abstract::MakeAbstract(std::make_shared<abstract::Shape>(param.x_shape), param.x_type);
  ASSERT_NE(x, nullptr);
  ASSERT_ANY_THROW(EpsInfer(nullptr, prim, {x}));
}
}  // namespace ops
}  // namespace mindspore