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
#include <exception>
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "include/backend/common/pass_manager/helper.h"
#include "ops/test_ops.h"
#include "infer/ops_func_impl/add_rmsnorm_dynamic_quant.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace ops {

struct AddRmsNormDynamicQuantShape {
  std::vector<int64_t> x1_shape;
  std::vector<int64_t> x2_shape;
  std::vector<int64_t> gamma_shape;
  std::vector<int64_t> smooth_scale1_shape;
  std::vector<int64_t> smooth_scale2_shape;
  ValuePtr eps;

  std::vector<int64_t> quant_out1_shape;
  std::vector<int64_t> quant_out2_shape;
  std::vector<int64_t> add_out_shape;
  std::vector<int64_t> scale_out1_shape;
  std::vector<int64_t> scale_out2_shape;
};

struct AddRmsNormDynamicQuantType {
  TypePtr x1_type;
  TypePtr x2_type;
  TypePtr gamma_type;
  TypePtr smooth_scale1_type;
  TypePtr smooth_scale2_type;

  TypePtr quant_out1_type;
  TypePtr quant_out2_type;
  TypePtr add_out_type;
  TypePtr scale_out1_type;
  TypePtr scale_out2_type;
};

class TestAddRmsNormDynamicQuant
    : public TestOps,
      public testing::WithParamInterface<std::tuple<AddRmsNormDynamicQuantShape, AddRmsNormDynamicQuantType>> {};

class TestAddRmsNormDynamicQuantException
    : public TestOps,
      public testing::WithParamInterface<std::tuple<AddRmsNormDynamicQuantShape, AddRmsNormDynamicQuantType>> {};

TEST_P(TestAddRmsNormDynamicQuant, AddRmsNormDynamicQuant_dyn_shape) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  AddRmsNormDynamicQuantFuncImpl AddRmsNormDynamicQuant_func_impl;
  auto x1 = std::make_shared<abstract::AbstractTensor>(dtype_param.x1_type, shape_param.x1_shape);
  auto x2 = std::make_shared<abstract::AbstractTensor>(dtype_param.x2_type, shape_param.x2_shape);
  auto gamma = std::make_shared<abstract::AbstractTensor>(dtype_param.gamma_type, shape_param.gamma_shape);
  auto smooth_scale1 =
    std::make_shared<abstract::AbstractTensor>(dtype_param.smooth_scale1_type, shape_param.smooth_scale1_shape);
  auto smooth_scale2 =
    std::make_shared<abstract::AbstractTensor>(dtype_param.smooth_scale2_type, shape_param.smooth_scale2_shape);

  auto expect_quant_out1_shape = std::make_shared<abstract::TensorShape>(shape_param.quant_out1_shape);
  auto expect_quant_out2_shape = std::make_shared<abstract::TensorShape>(shape_param.quant_out2_shape);
  auto expect_add_out_shape = std::make_shared<abstract::TensorShape>(shape_param.add_out_shape);
  auto expect_scale_out1_shape = std::make_shared<abstract::TensorShape>(shape_param.scale_out1_shape);
  auto expect_scale_out2_shape = std::make_shared<abstract::TensorShape>(shape_param.scale_out2_shape);

  auto expect_quant_out1_type = std::make_shared<TensorType>(dtype_param.quant_out1_type);
  auto expect_quant_out2_type = std::make_shared<TensorType>(dtype_param.quant_out2_type);
  auto expect_add_out_type = std::make_shared<TensorType>(dtype_param.add_out_type);
  auto expect_scale_out1_type = std::make_shared<TensorType>(dtype_param.scale_out1_type);
  auto expect_scale_out2_type = std::make_shared<TensorType>(dtype_param.scale_out2_type);

  std::vector<BaseShapePtr> shapes_list{expect_quant_out1_shape, expect_quant_out2_shape, expect_add_out_shape,
                                        expect_scale_out1_shape, expect_scale_out2_shape};
  auto expect_shapes = std::make_shared<abstract::TupleShape>(shapes_list);
  std::vector<TypePtr> types_list = {expect_quant_out1_type, expect_quant_out2_type, expect_add_out_type,
                                     expect_scale_out1_type, expect_scale_out2_type};
  auto expect_types = std::make_shared<Tuple>(types_list);

  auto prim = std::make_shared<Primitive>(kNameAddRmsNormDynamicQuant);
  auto out_dtypes = AddRmsNormDynamicQuant_func_impl.InferType(
    prim, {x1, x2, gamma, smooth_scale1, smooth_scale2, shape_param.eps->ToAbstract()});
  ASSERT_TRUE(*out_dtypes == *expect_types);
  auto out_shapes = AddRmsNormDynamicQuant_func_impl.InferShape(
    prim, {x1, x2, gamma, smooth_scale1, smooth_scale2, shape_param.eps->ToAbstract()});
  ASSERT_TRUE(*out_shapes == *expect_shapes);
}

TEST_P(TestAddRmsNormDynamicQuantException, AddRmsNormDynamicQuant_exception) {
  const auto &shape_param = std::get<0>(GetParam());
  const auto &dtype_param = std::get<1>(GetParam());

  AddRmsNormDynamicQuantFuncImpl AddRmsNormDynamicQuant_func_impl;
  auto prim = std::make_shared<Primitive>(kNameAddRmsNormDynamicQuant);
  auto x1 = std::make_shared<abstract::AbstractTensor>(dtype_param.x1_type, shape_param.x1_shape);
  auto x2 = std::make_shared<abstract::AbstractTensor>(dtype_param.x2_type, shape_param.x2_shape);
  auto gamma = std::make_shared<abstract::AbstractTensor>(dtype_param.gamma_type, shape_param.gamma_shape);
  auto smooth_scale1 =
    std::make_shared<abstract::AbstractTensor>(dtype_param.smooth_scale1_type, shape_param.smooth_scale1_shape);
  auto smooth_scale2 =
    std::make_shared<abstract::AbstractTensor>(dtype_param.smooth_scale2_type, shape_param.smooth_scale2_shape);

  bool raise_exception = false;
  try {
    (void)AddRmsNormDynamicQuant_func_impl.InferShape(
      prim, {x1, x2, gamma, smooth_scale1, smooth_scale2, shape_param.eps->ToAbstract()});
  } catch (std::exception &e) {
    raise_exception = true;
  }

  ASSERT_TRUE(raise_exception);
}

auto AddRmsNormDynamicQuantOpShapeTestCases =
  testing::ValuesIn({AddRmsNormDynamicQuantShape{{1, 4, 6},
                                                 {1, 4, 6},
                                                 {6},
                                                 {6},
                                                 {6},
                                                 CreateScalar<float>(0.000001),
                                                 {1, 4, 6},
                                                 {1, 4, 6},
                                                 {1, 4, 6},
                                                 {1, 4},
                                                 {1, 4}},
                     AddRmsNormDynamicQuantShape{{-1, -1, -1},
                                                 {-1, -1, -1},
                                                 {6},
                                                 {6},
                                                 {6},
                                                 CreateScalar<float>(0.000001),
                                                 {-1, -1, -1},
                                                 {-1, -1, -1},
                                                 {-1, -1, -1},
                                                 {-1, -1},
                                                 {-1, -1}},
                     AddRmsNormDynamicQuantShape{
                       {-2}, {-2}, {-2}, {-2}, {-2}, CreateScalar<float>(0.000001), {-2}, {-2}, {-2}, {-2}, {-2}}});

auto AddRmsNormDynamicQuantOpTypeTestCases = testing::ValuesIn({
  AddRmsNormDynamicQuantType{kFloat16, kFloat16, kFloat16, kFloat16, kFloat16, kInt8, kInt8, kFloat16, kFloat32,
                             kFloat32},
  AddRmsNormDynamicQuantType{kBFloat16, kBFloat16, kBFloat16, kBFloat16, kBFloat16, kInt8, kInt8, kBFloat16, kFloat32,
                             kFloat32},
});

auto AddRmsNormDynamicQuantOpInvalidShapeTestCases = testing::ValuesIn(
  {AddRmsNormDynamicQuantShape{{1, 4, 6}, {1, 1, 6}, {6}, {6}, {6}, CreateScalar<float>(0.000001), {}, {}, {}, {}, {}},
   AddRmsNormDynamicQuantShape{
     {1, 4, 6}, {1, 4, 6}, {6, 6}, {6}, {6}, CreateScalar<float>(0.000001), {}, {}, {}, {}, {}},
   AddRmsNormDynamicQuantShape{
     {1, 4, 6}, {1, 4, 6}, {6}, {6, 6}, {6}, CreateScalar<float>(0.000001), {}, {}, {}, {}, {}},
   AddRmsNormDynamicQuantShape{
     {1, 3, 6}, {3, 6}, {6}, {6}, {6}, CreateScalar<float>(0.000001), {}, {}, {}, {}, {}},
   AddRmsNormDynamicQuantShape{{4}, {4}, {4}, {4}, {4}, CreateScalar<float>(0.000001), {}, {}, {}, {}, {}},
   AddRmsNormDynamicQuantShape{{3, 4}, {3, 4}, {4}, {5}, {4}, CreateScalar<float>(0.000001), {}, {}, {}, {}, {}},
   AddRmsNormDynamicQuantShape{
     {1, 4, 6}, {1, 4, 6}, {5}, {-2}, {-2}, CreateScalar<float>(0.000001), {}, {}, {}, {}, {}}});

INSTANTIATE_TEST_CASE_P(TestAddRmsNormDynamicQuant, TestAddRmsNormDynamicQuant,
                        testing::Combine(AddRmsNormDynamicQuantOpShapeTestCases,
                                         AddRmsNormDynamicQuantOpTypeTestCases));

INSTANTIATE_TEST_CASE_P(TestAddRmsNormDynamicQuantException, TestAddRmsNormDynamicQuantException,
                        testing::Combine(AddRmsNormDynamicQuantOpInvalidShapeTestCases,
                                         AddRmsNormDynamicQuantOpTypeTestCases));
}  // namespace ops
}  // namespace mindspore
