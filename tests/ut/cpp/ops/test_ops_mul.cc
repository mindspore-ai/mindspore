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
#include "infer/ops_func_impl/mul.h"
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
class TestMul : public TestOps, public testing::WithParamInterface<BroadcastOpParams> {};

TEST_P(TestMul, mul_dyn_shape) {
  auto primitive = std::make_shared<Primitive>("Mul");
  ASSERT_NE(primitive, nullptr);
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto y = std::make_shared<abstract::AbstractTensor>(param.y_type, param.y_shape);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  auto x_info = std::make_unique<AbstractInferInfoAdapter>(x, "Mul", "x");
  auto y_info = std::make_unique<AbstractInferInfoAdapter>(y, "Mul", "y");
  InferInfoPtrList input_infos;
  input_infos.push_back(std::move(x_info));
  input_infos.push_back(std::move(y_info));
  auto infer_impl = std::make_shared<MulFuncImpl>();
  ASSERT_NE(infer_impl, nullptr);
  auto infer_shape = std::make_shared<abstract::Shape>(infer_impl->InferShape(primitive, input_infos)[0]);
  ASSERT_NE(infer_shape, nullptr);
  auto infer_type = TypeIdToType(infer_impl->InferType(primitive, input_infos)[0]);
  ASSERT_NE(infer_type, nullptr);

  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  ASSERT_NE(expect_shape, nullptr);
  auto expect_type = param.out_type;
  ASSERT_NE(expect_type, nullptr);
  ASSERT_TRUE(*infer_shape == *expect_shape);
  ASSERT_TRUE(infer_type == expect_type);
}

INSTANTIATE_TEST_CASE_P(
  TestMulGroup, TestMul,
  testing::Values(BroadcastOpParams{{2, 1}, kFloat32, {1, 1, 4}, kFloat32, {1, 2, 4}, kFloat32},
                  BroadcastOpParams{{-1, 3}, kFloat32, {-1, 1}, kFloat32, {-1, 3}, kFloat32},
                  BroadcastOpParams{{-1, -1}, kFloat32, {-1, -1, -1}, kFloat32, {-1, -1, -1}, kFloat32},
                  BroadcastOpParams{{-1, 1, 4}, kFloat32, {1, -1, 4}, kFloat32, {-1, -1, 4}, kFloat32},
                  BroadcastOpParams{{-1, 2, 3}, kFloat32, {2, -1, 3}, kFloat32, {2, 2, 3}, kFloat32},
                  BroadcastOpParams{{-2}, kFloat32, {2, 3}, kFloat32, {-2}, kFloat32}));

struct MulInferValueParams {
  tensor::TensorPtr x;
  tensor::TensorPtr y;
  tensor::TensorPtr out;
  bool expect_throw{false};
};

class TestMulInferValue : public TestOps, public testing::WithParamInterface<MulInferValueParams> {};

TEST_P(TestMulInferValue, infer_value_cases) {
  const auto &param = GetParam();
  ASSERT_NE(param.x, nullptr);
  ASSERT_NE(param.y, nullptr);
  auto x_abs = param.x->ToAbstract();
  auto y_abs = param.y->ToAbstract();
  auto primitive = std::make_shared<Primitive>("Mul");
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

INSTANTIATE_TEST_CASE_P(
  TestMulInferValue, TestMulInferValue,
  testing::Values(
    // same type
    MulInferValueParams{CreateTensorF32({2}, {1.0, 2.0}), CreateTensorF32({2}, {3.0, 4.0}),
                        CreateTensorF32({2}, {3.0, 8.0})},
    MulInferValueParams{CreateTensorF16({2}, {float16(1.0), float16(2.0)}),
                        CreateTensorF16({2}, {float16(3.0), float16(4.0)}),
                        CreateTensorF16({2}, {float16(3.0), float16(8.0)})},
    MulInferValueParams{CreateTensorI32({2}, {1, 2}), CreateTensorI32({2}, {3, 4}), CreateTensorI32({2}, {3, 8})},
    // mixed type
    MulInferValueParams{CreateTensorF32({2}, {1.0, 2.0}), CreateTensorI32({2}, {3, 4}),
                        CreateTensorF32({2}, {3.0, 8.0})},
    MulInferValueParams{CreateTensorF16({2}, {float16(1.0), float16(2.0)}), CreateTensorI32({2}, {3, 4}),
                        CreateTensorF16({2}, {float16(3.0), float16(8.0)})},
    MulInferValueParams{CreateTensorF32({2}, {1.0, 2.0}), CreateTensorF16({2}, {float16(3.0), float16(4.0)}),
                        CreateTensorF32({2}, {3.0, 8.0})}));
}  // namespace ops
}  // namespace mindspore
