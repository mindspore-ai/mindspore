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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "infer/ops_func_impl/free.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/test_ops.h"

namespace mindspore {
namespace ops {

struct FreeParams {
  ShapeVector xShape;
  TypePtr xType;
  ShapeVector outputShape;
  TypePtr outputType;
};

class TestFree : public TestOps, public testing::WithParamInterface<FreeParams> {};

// Feature: Free operator type and shape inference
// Description: Test Free operator's InferType and InferShape with dynamic shapes
// Expectation: Returns correct output type and shape matching input
TEST_P(TestFree, dynShape) {
  const auto &param = GetParam();
  auto xTensor = std::make_shared<abstract::AbstractTensor>(param.xType, param.xShape);

  auto expectType = std::make_shared<TensorType>(param.outputType);
  auto expectShape = std::make_shared<abstract::Shape>(param.outputShape);

  FreeFuncImpl freeFuncImpl;
  auto primitive = std::make_shared<Primitive>("Free");

  auto realType = freeFuncImpl.InferType(primitive, {xTensor});
  ASSERT_TRUE(*realType == *expectType);
  auto realShape = freeFuncImpl.InferShape(primitive, {xTensor});
  ASSERT_TRUE(*realShape == *expectShape);
}

INSTANTIATE_TEST_CASE_P(TestFree, TestFree,
                        testing::Values(FreeParams{{3, 4, 5}, kFloat32, {3, 4, 5}, kFloat32},
                                        FreeParams{{2, 3, 4}, kFloat16, {2, 3, 4}, kFloat16},
                                        FreeParams{{-3, 4, 5}, kBFloat16, {-3, 4, 5}, kBFloat16},
                                        FreeParams{{1, 2, 3, 4, 5}, kFloat32, {1, 2, 3, 4, 5}, kFloat32},
                                        FreeParams{{-1, -2, -3, 4, 5}, kFloat16, {-1, -2, -3, 4, 5}, kFloat16},
                                        FreeParams{{3, 4}, kBFloat16, {3, 4}, kBFloat16}));

}  // namespace ops
}  // namespace mindspore
