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
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "infer/ops_func_impl/less.h"
#include "ops/test_ops.h"
#include "ops/utils/general_infer_utils.h"
#include "ir/tensor_new.h"
#include "ops/test_value_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "ops/op_def.h"
#include "ops/infer_info/abstract_infer_info_adapter.h"

namespace mindspore {
namespace ops {
struct LessOpParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector y_shape;
  TypePtr y_type;
  ShapeVector output_shape;
  TypePtr output_type;
};

class TestLess : public TestOps, public testing::WithParamInterface<LessOpParams> {};

TEST_P(TestLess, less_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("Less");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  ASSERT_NE(x, nullptr);
  auto y = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  ASSERT_NE(y, nullptr);
  std::vector<abstract::AbstractBasePtr> input_args{std::move(x), std::move(y)};
  auto infer_impl = std::make_shared<LessFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = infer_impl->InferShape(primitive, input_args);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = infer_impl->InferType(primitive, input_args);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = std::make_shared<TensorType>(param.output_type);
  ASSERT_NE(expect_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(*infer_type == *expect_type);
}

INSTANTIATE_TEST_CASE_P(TestLessGroup, TestLess,
                        testing::Values(LessOpParams{{2, 3, 4}, kFloat32, {2, 1, 4}, kFloat32, {2, 3, 4}, kBool},
                                        LessOpParams{{2, 3, 4}, kFloat32, {-1, -1, -1}, kFloat32, {2, 3, 4}, kBool},
                                        LessOpParams{{2, 3, 4}, kFloat32, {1}, kFloat32, {2, 3, 4}, kBool}));

struct LessInferValueParams {
  tensor::TensorPtr x;
  tensor::TensorPtr y;
  tensor::TensorPtr out;
};

class TestLessInferValue : public TestOps, public testing::WithParamInterface<LessInferValueParams> {};

TEST_P(TestLessInferValue, infer_value_cases) {
  const auto &param = GetParam();
  ASSERT_NE(param.x, nullptr);
  ASSERT_NE(param.y, nullptr);
  auto x_abs = param.x->ToAbstract();
  auto y_abs = param.y->ToAbstract();
  auto primitive = std::make_shared<Primitive>("Less");
  ASSERT_NE(primitive, nullptr);

  auto input_args = abstract::AbstractBasePtrList{x_abs, y_abs};
  auto value_opt = abstract::InferValueByFuncImpl(primitive, input_args);
  ASSERT_TRUE(value_opt.has_value());
  auto infer_out = value_opt.value();
  ASSERT_NE(infer_out, nullptr);
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

static tensor::TensorPtr CreateTensorF32(const ShapeVector &shape, const std::vector<float> &vals) {
  return CreateTensor<float>(kNumberTypeFloat32, shape, vals);
}

static tensor::TensorPtr CreateTensorF16(const ShapeVector &shape, const std::vector<float16> &vals) {
  return CreateTensor<float16>(kNumberTypeFloat16, shape, vals);
}

static tensor::TensorPtr CreateTensorI32(const ShapeVector &shape, const std::vector<int32_t> &vals) {
  return CreateTensor<int32_t>(kNumberTypeInt32, shape, vals);
}

static tensor::TensorPtr CreateTensorI64(const ShapeVector &shape, const std::vector<int64_t> &vals) {
  return CreateTensor<int64_t>(kNumberTypeInt64, shape, vals);
}

static tensor::TensorPtr CreateTensorBool(const ShapeVector &shape, const std::vector<bool> &vals) {
  std::vector<uint8_t> int_vals(vals.begin(), vals.end());
  return CreateTensor<uint8_t>(kNumberTypeBool, shape, int_vals);
}

INSTANTIATE_TEST_CASE_P(
  TestLessInferValue, TestLessInferValue,
  testing::Values(
    // same type
    LessInferValueParams{CreateTensorF32({2}, {1.0, 2.0}), CreateTensorF32({2}, {3.0, 1.0}),
                         CreateTensorBool({2}, {true, false})},
    LessInferValueParams{CreateTensorF16({2}, {float16(1.0), float16(2.0)}),
                         CreateTensorF16({2}, {float16(3.0), float16(1.0)}), CreateTensorBool({2}, {true, false})},
    LessInferValueParams{CreateTensorI32({2}, {1, 2}), CreateTensorI32({2}, {3, 1}),
                         CreateTensorBool({2}, {true, false})},
    // mixed type
    LessInferValueParams{CreateTensorF32({2}, {1.0, 2.0}), CreateTensorI32({2}, {3, 1}),
                         CreateTensorBool({2}, {true, false})},
    LessInferValueParams{CreateTensorF16({2}, {float16(1.0), float16(2.0)}), CreateTensorI32({2}, {3, 1}),
                         CreateTensorBool({2}, {true, false})},
    LessInferValueParams{CreateTensorF32({2}, {1.0, 2.0}), CreateTensorF16({2}, {float16(3.0), float16(1.0)}),
                         CreateTensorBool({2}, {true, false})},
    LessInferValueParams{CreateTensorI32({2}, {1, 2}), CreateTensorI64({2}, {3, 1}),
                         CreateTensorBool({2}, {true, false})}));
}  // namespace ops
}  // namespace mindspore
