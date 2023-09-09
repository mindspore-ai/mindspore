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
#include "ops/sequence_op_name.h"
#include "ops/softmax.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "utils/tensor_construct_utils.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"
#include "ir/primitive.h"


namespace mindspore {
namespace ops {
struct SoftMaxParams {
  std::vector<int64_t> axis;
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestSoftMax: public UT::Common, public testing::WithParamInterface<SoftMaxParams> {
  public:
    TestSoftMax() = default;
    ~TestSoftMax() override = default;
    void SetUp() {}
    void TearDown() {}
};

TEST_P(TestSoftMax, test_softmax) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(expect, nullptr);
  auto prim = std::make_shared<Primitive>(kNameSoftmax);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("axis", MakeValue<std::vector<int64_t>>(param.axis));
  auto abstract = SoftmaxInfer(nullptr, prim, {x});
  ASSERT_NE(abstract, nullptr);
  ASSERT_TRUE(*abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestSoftMaxGroup, TestSoftMax,
                        testing::Values(SoftMaxParams{{-1}, {1, 2, 3, 4, 5}, kFloat16, {1, 2, 3, 4, 5}, kFloat16},
                                        SoftMaxParams{{-1}, {1, 2, 3, 4, 5}, kFloat32, {1, 2, 3, 4, 5}, kFloat32},
                                        SoftMaxParams{{-1}, {1, 2, 3, 4, 5}, kFloat64, {1, 2, 3, 4, 5}, kFloat64}));

}  // namespace ops
}  // namespace mindspore{}