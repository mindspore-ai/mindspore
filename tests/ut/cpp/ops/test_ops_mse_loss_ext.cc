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
#include <memory>
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "infer/ops_func_impl/mse_loss_ext.h"
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace ops {
namespace {
struct MSELossExtParams {
  ShapeVector input_shape;
  TypePtr input_type;
  ShapeVector target_shape;
  TypePtr target_type;
  AbstractBasePtr reduction;
  ShapeVector output_shape;
  TypePtr output_type;
};

AbstractBasePtr CreateInt(const int &value) { return CreatePyInt(value)->ToAbstract(); }

}  // namespace

class TestMSELossExt : public TestOps,
                       public testing::WithParamInterface<std::tuple<const char *, MSELossExtParams>> {};

TEST_P(TestMSELossExt, dyn_shape) {
  const auto &op_name = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  auto op_impl = std::make_shared<MSELossExtFuncImpl>();
  ASSERT_NE(op_impl, nullptr);

  auto prim = std::make_shared<Primitive>(op_name);
  ASSERT_NE(prim, nullptr);
  auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
  ASSERT_NE(input, nullptr);
  auto target = std::make_shared<abstract::AbstractTensor>(param.target_type, param.target_shape);
  ASSERT_NE(target, nullptr);

  auto input_args = std::vector<AbstractBasePtr>{input, target, param.reduction};

  auto expect_shape = std::make_shared<abstract::TensorShape>(param.output_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.output_type);
  ASSERT_NE(expect_type, nullptr);

  auto out_shape = op_impl->InferShape(prim, input_args);
  auto out_type = op_impl->InferType(prim, input_args);

  ShapeCompare(out_shape, expect_shape);
  TypeCompare(out_type, expect_type);
}

class TestMSELossExtSimpleInfer : public TestOps,
                                  public testing::WithParamInterface<std::tuple<const char *, MSELossExtParams>> {};

TEST_P(TestMSELossExtSimpleInfer, simple_infer) {
  const auto &op_name = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());
  auto op_impl = std::make_shared<MSELossExtFuncImpl>();
  ASSERT_NE(op_impl, nullptr);

  auto prim = std::make_shared<Primitive>(op_name);
  ASSERT_NE(prim, nullptr);
  auto input = tensor::from_spec(param.input_type->type_id(), param.input_shape, device::DeviceType::kCPU);
  ASSERT_NE(input, nullptr);
  auto target = tensor::from_spec(param.target_type->type_id(), param.target_shape, device::DeviceType::kCPU);
  ASSERT_NE(input, nullptr);
  ValuePtrList input_values;
  input_values.emplace_back(input);
  input_values.emplace_back(target);
  input_values.emplace_back(param.reduction->GetValue());

  auto expect_shape = ShapeArray{param.output_shape};
  auto expect_type = TypePtrList{param.output_type};

  auto output_shape = op_impl->InferShape(prim, input_values);
  auto output_type = op_impl->InferType(prim, input_values);

  ShapeCompare(output_shape, expect_shape);
  TypeCompare(output_type, expect_type);
}

// enum Reduction : int64_t {REDUCTION_SUM = 0, MEAN = 1, NONE = 2};
auto MSELossExtTestCase = testing::ValuesIn({
  MSELossExtParams{{3, 1, 5, 1}, kFloat16, {2, 1, 3}, kFloat16, CreateInt(2), {3, 2, 5, 3}, kFloat16},
  MSELossExtParams{{3, 1, 5, 2}, kBFloat16, {2, 5, 1}, kBFloat16, CreateInt(1), {}, kBFloat16},
  MSELossExtParams{{2, 5, 1}, kFloat32, {3, 1, 5, 2}, kFloat32, CreateInt(0), {}, kFloat32},
  MSELossExtParams{{3, 1, -1, 6}, kFloat32, {2, 3, 1}, kFloat32, CreateInt(2), {3, 2, 3, 6}, kFloat32},
  MSELossExtParams{{2, 3, 1}, kFloat32, {3, 1, -1, 6}, kFloat32, CreateInt(2), {3, 2, 3, 6}, kFloat32},
  MSELossExtParams{{3, 1, -1, 6}, kFloat32, {2, -1, 1}, kFloat32, CreateInt(2), {3, 2, -1, 6}, kFloat32},
  MSELossExtParams{{2, -1, 1}, kFloat32, {3, 1, -1, 6}, kFloat32, CreateInt(2), {3, 2, -1, 6}, kFloat32},
  MSELossExtParams{{2, -1, 1}, kFloat32, {}, kFloat32, CreateInt(2), {2, -1, 1}, kFloat32},
  MSELossExtParams{{}, kFloat32, {3, 1, -1, 6}, kFloat32, CreateInt(2), {3, 1, -1, 6}, kFloat32},
  MSELossExtParams{{3, 4, 5}, kFloat32, {-2}, kFloat32, CreateInt(2), {-2}, kFloat32},
  MSELossExtParams{{-2}, kFloat32, {3, 4, 5}, kFloat32, CreateInt(2), {-2}, kFloat32},
  MSELossExtParams{{-2}, kFloat32, {-2}, kFloat32, CreateInt(2), {-2}, kFloat32},
});

INSTANTIATE_TEST_CASE_P(TestMSELossExtGroup, TestMSELossExt,
                        testing::Combine(testing::ValuesIn({kNameMSELossExt}), MSELossExtTestCase));

auto MSELossExtSimpleInferTestCase = testing::ValuesIn(
  {MSELossExtParams{{3, 1, 5, 1}, kFloat16, {2, 1, 3}, kFloat16, CreateInt(2), {3, 2, 5, 3}, kFloat16},
   MSELossExtParams{{3, 1, 5, 1}, kFloat32, {}, kFloat32, CreateInt(2), {3, 1, 5, 1}, kFloat32},
   MSELossExtParams{{}, kFloat32, {2, 1, 3}, kFloat32, CreateInt(2), {2, 1, 3}, kFloat32},
   MSELossExtParams{{3, 1, 5, 2}, kBFloat16, {2, 5, 1}, kBFloat16, CreateInt(1), {}, kBFloat16},
   MSELossExtParams{{2, 5, 1}, kFloat32, {3, 1, 5, 2}, kFloat32, CreateInt(0), {}, kFloat32}});

INSTANTIATE_TEST_CASE_P(TestMSELossExtGroup, TestMSELossExtSimpleInfer,
                        testing::Combine(testing::ValuesIn({kNameMSELossExt}), MSELossExtSimpleInferTestCase));

}  // namespace ops
}  // namespace mindspore
