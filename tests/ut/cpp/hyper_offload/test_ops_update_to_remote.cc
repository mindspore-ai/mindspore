/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/update_to_remote.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {

struct UpdateToRemoteParams {
  ShapeVector xShape;
  TypePtr xType;
  ValuePtr sync;
  ShapeVector outputShape;
  TypePtr outputType;
};

class TestUpdateToRemote : public TestOps, public testing::WithParamInterface<UpdateToRemoteParams> {};

// Feature: UpdateToRemote operator type and shape inference
// Description: Test UpdateToRemote operator's InferType and InferShape with dynamic shapes
// Expectation: Returns correct output type and shape matching input
TEST_P(TestUpdateToRemote, dynShape) {
  const auto &param = GetParam();
  auto xTensor = std::make_shared<abstract::AbstractTensor>(param.xType, param.xShape);
  auto syncAbstract = param.sync->ToAbstract();

  auto expectShape = std::make_shared<abstract::Shape>(param.outputShape);
  auto expectType = std::make_shared<TensorType>(param.outputType);

  UpdateToRemoteFuncImpl updateToRemoteFuncImpl;
  auto primitive = std::make_shared<Primitive>("UpdateToRemote");

  auto realType = updateToRemoteFuncImpl.InferType(primitive, {xTensor, syncAbstract});
  ASSERT_TRUE(*realType == *expectType);

  auto realShape = updateToRemoteFuncImpl.InferShape(primitive, {xTensor, syncAbstract});
  ASSERT_TRUE(*realShape == *expectShape);
}

INSTANTIATE_TEST_CASE_P(
  TestUpdateToRemote, TestUpdateToRemote,
  testing::Values(
    UpdateToRemoteParams{{1}, kFloat32, CreateScalar<bool>(true), {1}, kFloat32},
    UpdateToRemoteParams{{6}, kFloat32, CreateScalar<bool>(true), {6}, kFloat32},
    UpdateToRemoteParams{{6, 7}, kFloat32, CreateScalar<bool>(true), {6, 7}, kFloat32},
    UpdateToRemoteParams{{6, 7, 8}, kFloat32, CreateScalar<bool>(true), {6, 7, 8}, kFloat32},
    UpdateToRemoteParams{{6, 7, 8, 9}, kFloat32, CreateScalar<bool>(true), {6, 7, 8, 9}, kFloat32},
    UpdateToRemoteParams{{6, 7, 8, 9, 10}, kFloat32, CreateScalar<bool>(true), {6, 7, 8, 9, 10}, kFloat32},
    UpdateToRemoteParams{{1}, kFloat32, CreateScalar<bool>(false), {1}, kFloat32},
    UpdateToRemoteParams{{6}, kFloat32, CreateScalar<bool>(false), {6}, kFloat32},
    UpdateToRemoteParams{{6, 7}, kFloat32, CreateScalar<bool>(false), {6, 7}, kFloat32},
    UpdateToRemoteParams{{6, 7, 8}, kFloat32, CreateScalar<bool>(false), {6, 7, 8}, kFloat32},
    UpdateToRemoteParams{{6, 7, 8, 9}, kFloat32, CreateScalar<bool>(false), {6, 7, 8, 9}, kFloat32},
    UpdateToRemoteParams{{6, 7, 8, 9, 10}, kFloat32, CreateScalar<bool>(false), {6, 7, 8, 9, 10}, kFloat32},
    UpdateToRemoteParams{{6, 7, 8, 9}, kInt16, CreateScalar<bool>(true), {6, 7, 8, 9}, kInt16},
    UpdateToRemoteParams{{6, 7, 8, 9}, kInt32, CreateScalar<bool>(true), {6, 7, 8, 9}, kInt32},
    UpdateToRemoteParams{{6, 7, 8, 9}, kInt64, CreateScalar<bool>(true), {6, 7, 8, 9}, kInt64}));

}  // namespace ops
}  // namespace mindspore
