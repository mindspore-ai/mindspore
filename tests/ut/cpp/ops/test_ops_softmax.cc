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
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "primitive/nn_op_name.h"
#include "primitive/op_name.h"
#include "infer/ops_func_impl/softmax.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "utils/ms_context.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
struct SoftmaxParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ValuePtr axis;
  ShapeVector out_shape;
};

class TestSoftmax : public TestOps, public testing::WithParamInterface<SoftmaxParams> {};

TEST_P(TestSoftmax, dyn_shape) {
  const auto &param = GetParam();
  auto x = std::make_shared<abstract::AbstractTensor>(param.x_type, param.x_shape);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(x, nullptr);
  ASSERT_NE(axis, nullptr);

  auto expect = std::make_shared<abstract::TensorShape>(param.out_shape);
  auto prim = std::make_shared<Primitive>(kSoftmaxOpName);
  auto infer_impl = std::make_shared<SoftmaxFuncImpl>();
  auto out_shape = infer_impl->InferShape(prim, {x, axis});
  ASSERT_NE(out_shape, nullptr);
  ASSERT_TRUE(*out_shape == *expect);
}

INSTANTIATE_TEST_CASE_P(
  TestSoftmax, TestSoftmax,
  testing::Values(SoftmaxParams{ShapeVector{2, 4, 8}, kFloat32, CreateTuple({1}), ShapeVector{2, 4, 8}},
                  SoftmaxParams{ShapeVector{2, 4, 8}, kFloat32, CreateTuple({kValueAny}), ShapeVector{2, 4, 8}},
                  SoftmaxParams{ShapeVector{2, 4, 8}, kFloat32, CreateTuple({1}), ShapeVector{2, 4, 8}},
                  SoftmaxParams{ShapeVector{2, 4, 8}, kFloat32, CreateTuple({kValueAny}), ShapeVector{2, 4, 8}},
                  SoftmaxParams{ShapeVector{-1, 2, -1}, kFloat32, CreateTuple({1}), ShapeVector{-1, 2, -1}},
                  SoftmaxParams{ShapeVector{-1, 2, -1}, kFloat32, CreateTuple({kValueAny}), ShapeVector{-1, 2, -1}},
                  SoftmaxParams{ShapeVector{-1, 2, -1}, kFloat32, CreateTuple({1}), ShapeVector{-1, 2, -1}},
                  SoftmaxParams{ShapeVector{-1, 2, -1}, kFloat32, CreateTuple({kValueAny}), ShapeVector{-1, 2, -1}},
                  SoftmaxParams{ShapeVector{-2}, kFloat32, CreateTuple({1}), ShapeVector{-2}},
                  SoftmaxParams{ShapeVector{-2}, kFloat32, CreateTuple({kValueAny}), ShapeVector{-2}},
                  SoftmaxParams{ShapeVector{-2}, kFloat32, CreateTuple({1}), ShapeVector{-2}},
                  SoftmaxParams{ShapeVector{-2}, kFloat32, CreateTuple({kValueAny}), ShapeVector{-2}}));

struct SoftmaxInferValueParams {
  tensor::TensorPtr input;
  ValuePtr axis;
  tensor::TensorPtr out;
};

class TestSoftmaxInferValue : public TestOps, public testing::WithParamInterface<SoftmaxInferValueParams> {};

TEST_P(TestSoftmaxInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  ASSERT_NE(input, nullptr);

  ASSERT_NE(param.axis, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);

  auto input_args = abstract::AbstractBasePtrList{input, axis};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimSoftmax, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "Softmax have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "Softmax can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestSoftmaxInferValue, TestSoftmaxInferValue,
  testing::Values(
    SoftmaxInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{2, 2, 2, 2}),
                            CreateScalar<int64_t>(0),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{0.5, 0.5, 0.5, 0.5})},
    SoftmaxInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{1, 2, 1, 1, 2, 1}),
                            CreateScalar<int64_t>(0),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 3}, std::vector<float>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5})},
    SoftmaxInferValueParams{CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{3, 3, 3, 3}),
                            CreateScalar<int64_t>(1),
                            CreateTensor<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{0.5, 0.5, 0.5, 0.5})}));
}  // namespace ops
}  // namespace mindspore
