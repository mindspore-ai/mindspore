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
#include "ops/randperm_v2.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"
#include "ir/tensor.h"

namespace mindspore {
namespace ops {
struct RandpermV2OpParams {
  ShapeVector input_shape;
  bool input_is_value_any;
  int64_t input_value;
  bool seed_is_value_any;
  int64_t seed_value;
  bool offset_is_value_any;
  int64_t offset_value;
  ShapeVector out_shape;
  TypePtr out_type;
};

class TestRandpermV2 : public TestOps, public testing::WithParamInterface<RandpermV2OpParams> {};

ValuePtr GetValuePtr(const bool is_value_any, const int64_t value) {
  if (is_value_any) {
    return kValueAny;
  } else {
    return std::make_shared<tensor::Tensor>(value, kInt64);  // n, seed, offset value must a tensor
  }
}

TEST_P(TestRandpermV2, dyn_shape) {
  const auto &param = GetParam();
  auto input_n = std::make_shared<abstract::AbstractTensor>(kInt64, param.input_shape);
  input_n->set_value(GetValuePtr(param.input_is_value_any, param.input_value));
  auto seed =
    std::make_shared<abstract::AbstractScalar>(GetValuePtr(param.seed_is_value_any, param.seed_value), kInt64);
  auto offset =
    std::make_shared<abstract::AbstractScalar>(GetValuePtr(param.offset_is_value_any, param.offset_value), kInt64);
  ASSERT_NE(input_n, nullptr);
  ASSERT_NE(seed, nullptr);
  ASSERT_NE(offset, nullptr);
  
  auto expect = std::make_shared<abstract::AbstractTensor>(param.out_type, param.out_shape);
  ASSERT_NE(expect, nullptr);

  auto prim = std::make_shared<Primitive>(kNameRandpermV2);
  prim->set_attr("dtype", param.out_type);
  auto out_abstract = opt::CppInferShapeAndType(prim, {input_n, seed, offset});
  ASSERT_NE(out_abstract, nullptr);
  ASSERT_TRUE(*out_abstract == *expect);
}

INSTANTIATE_TEST_CASE_P(TestRandpermV2Group, TestRandpermV2,
                        testing::Values(RandpermV2OpParams{{}, false, 10, false, 0, false, 0, {10}, kInt64},
                                        RandpermV2OpParams{{1}, false, 10, true, 0, true, 0, {10}, kInt64},
                                        RandpermV2OpParams{{}, true, -1, true, 0, true, 0, {-1}, kInt64},
                                        RandpermV2OpParams{{1}, false, 10, false, -1, true, 0, {10}, kInt64}));
}  // namespace ops
}  // namespace mindspore
