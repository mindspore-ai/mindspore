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
#include "ops/test_ops_cmp_utils.h"
#include "ir/dtype/number.h"
#include "ops/ops_func_impl/reflection_pad_1d.h"
#include "ops/ops_func_impl/reflection_pad_2d.h"
#include "ops/ops_func_impl/reflection_pad_3d.h"
#include "ops/ops_func_impl/replication_pad_1d.h"
#include "ops/ops_func_impl/replication_pad_2d.h"
#include "ops/ops_func_impl/replication_pad_3d.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/test_value_utils.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace ops {
struct PadReflectAndReplicateParams {
  ShapeVector input_shape;
  TypePtr input_dtype;
  ValuePtr    padding;
  ShapeVector output_shape;
  TypePtr output_dtype;
};
static std::map<std::string, OpFuncImplPtr> pad_func_impl = {
  {kNameReflectionPad1D, std::make_shared<ReflectionPad1DFuncImpl>()},
  {kNameReflectionPad2D, std::make_shared<ReflectionPad2DFuncImpl>()},
  {kNameReflectionPad3D, std::make_shared<ReflectionPad3DFuncImpl>()},
  {kNameReplicationPad1D, std::make_shared<ReplicationPad1DFuncImpl>()},
  {kNameReplicationPad2D, std::make_shared<ReplicationPad2DFuncImpl>()},
  {kNameReplicationPad3D, std::make_shared<ReplicationPad3DFuncImpl>()},
};
class TestPadReflectAndReplicate : public TestOps, public testing::WithParamInterface<std::tuple<const char *, PadReflectAndReplicateParams>> {};

TEST_P(TestPadReflectAndReplicate, dyn_shape) {
  const auto &pad_mode = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  auto pad_op_itr = pad_func_impl.find(pad_mode);
  ASSERT_TRUE(pad_op_itr != pad_func_impl.end());
  auto pad_op_impl = pad_op_itr->second;
  ASSERT_NE(pad_op_impl, nullptr);

  auto x = std::make_shared<abstract::AbstractTensor>(param.input_dtype, param.input_shape);
  ASSERT_NE(x, nullptr);
  auto padding = param.padding->ToAbstract();
  ASSERT_NE(padding, nullptr);
  auto prim = std::make_shared<Primitive>(pad_mode);
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto inferred_shape = pad_op_impl->InferShape(prim, {x, padding});
  ShapeCompare(inferred_shape, expect_shape);
}

namespace {
auto dyn_rank = abstract::TensorShape::kShapeRankAny;
auto dyn_dim = abstract::TensorShape::kShapeDimAny;
auto PadDynTestCase1D = testing::ValuesIn(
  {PadReflectAndReplicateParams{{2, 3, 4}, kFloat32, CreatePyIntTuple({1, 1}), {2, 3, 6}, kFloat32},
   PadReflectAndReplicateParams{{2, dyn_dim, 4}, kFloat32, CreatePyIntTuple({1, 1}), {2, dyn_dim, 6}, kFloat32},
   PadReflectAndReplicateParams{{dyn_rank}, kFloat32, CreatePyIntTuple({1, 1}), {dyn_rank}, kFloat32},
   PadReflectAndReplicateParams{{2, 3, 4}, kFloat32, kValueAny, {2, 3, dyn_dim}, kFloat32}});
auto PadDynTestCase2D = testing::ValuesIn(
  {PadReflectAndReplicateParams{{2, 3, 4, 5}, kFloat32, CreatePyIntTuple({1, 1, 1, 1}),
             {2, 3, 6, 7}, kFloat32},
   PadReflectAndReplicateParams{{2, dyn_dim, 4, 5}, kFloat32, CreatePyIntTuple({1, 1, 1, 1}),
             {2, dyn_dim, 6, 7}, kFloat32},
   PadReflectAndReplicateParams{{dyn_rank}, kFloat32, CreatePyIntTuple({1, 1, 1, 1}), {dyn_rank}, kFloat32},
   PadReflectAndReplicateParams{{2, dyn_dim, 4, 5}, kFloat32, kValueAny,
             {2, dyn_dim, dyn_dim, dyn_dim}, kFloat32}});
auto PadDynTestCase3D = testing::ValuesIn(
  {PadReflectAndReplicateParams{{2, 3, 4, 5}, kFloat32, CreatePyIntTuple({1, 1, 1, 1, 1, 1}),
             {2, 5, 6, 7}, kFloat32},
   PadReflectAndReplicateParams{{2, dyn_dim, 4, 5}, kFloat32, CreatePyIntTuple({1, 1, 1, 1, 1, 1}),
             {2, dyn_dim, 6, 7}, kFloat32},
   PadReflectAndReplicateParams{{dyn_rank}, kFloat32, CreatePyIntTuple({1, 1, 1, 1, 1, 1}),{dyn_rank}, kFloat32},
   PadReflectAndReplicateParams{{2, dyn_dim, 4, 5}, kFloat32, CreatePyIntTuple({1, 1, 1, 1, 1, 1}),
             {2, dyn_dim, 6, 7}, kFloat32}});
}

INSTANTIATE_TEST_CASE_P(TestReflectionPad1DGroup, TestPadReflectAndReplicate,
                        testing::Combine(testing::ValuesIn({kNameReflectionPad1D}), PadDynTestCase1D));
INSTANTIATE_TEST_CASE_P(TestReflectionPad2DGroup, TestPadReflectAndReplicate,
                        testing::Combine(testing::ValuesIn({kNameReflectionPad2D}), PadDynTestCase2D));
INSTANTIATE_TEST_CASE_P(TestReflectionPad3DGroup, TestPadReflectAndReplicate,
                        testing::Combine(testing::ValuesIn({kNameReflectionPad3D}), PadDynTestCase3D));
INSTANTIATE_TEST_CASE_P(TestReplicationPad1DGroup, TestPadReflectAndReplicate,
                        testing::Combine(testing::ValuesIn({kNameReplicationPad1D}), PadDynTestCase1D));
INSTANTIATE_TEST_CASE_P(TestReplicationPad2DGroup, TestPadReflectAndReplicate,
                        testing::Combine(testing::ValuesIn({kNameReplicationPad2D}), PadDynTestCase2D));
INSTANTIATE_TEST_CASE_P(TestReplicationPad3DGroup, TestPadReflectAndReplicate,
                        testing::Combine(testing::ValuesIn({kNameReplicationPad3D}), PadDynTestCase3D));
}  // namespace ops
}  // namespace mindspore
