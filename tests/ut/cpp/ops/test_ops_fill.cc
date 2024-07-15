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

#include "ops/ops_func_impl/fill_scalar.h"
#include "ops/ops_func_impl/fill_tensor.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/test_ops_cmp_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
struct FillOpParam {
  ValuePtr shape_value;
  ValuePtr fill_value;
  ValuePtr dtype_value;
  ShapeVector output_shape;
  TypePtr output_dtype;
};

// Test FillScalar InferShape and InferType
class TestFillScalar : public TestOps, public testing::WithParamInterface<FillOpParam> {};

TEST_P(TestFillScalar, fill_scalar_dyn_shape) {
  const auto param = GetParam();
  auto shape_list = param.shape_value->cast<ValueTuplePtr>();
  ASSERT_NE(shape_list, nullptr);
  auto shape_dim = shape_list->size();
  auto in_shape = param.shape_value->ToAbstract();
  auto in_shape_seq = in_shape->cast<abstract::AbstractSequencePtr>();
  if (shape_dim == 0) {
    in_shape_seq->CheckAndConvertToDynamicLenSequence();
  }
  auto fill_abs = param.fill_value->ToAbstract();
  auto dtype_abs = param.dtype_value->ToAbstract();
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);

  DoFuncImplInferAndCompare<FillScalarFuncImpl>(
      kNameFillScalar, {in_shape, fill_abs, dtype_abs}, expect_shape, expect_type);
  // skip the dynamic length input case
  if (shape_dim != 0) {
    DoFuncImplSimpleInferAndCompare<FillScalarFuncImpl>(
        kNameFillScalar, {param.shape_value, param.fill_value, param.dtype_value}, {param.output_shape}, {param.output_dtype});
  }
}

auto fill_scalar_test_cases = testing::Values(
    FillOpParam{CreatePyIntTuple({2, 2, 3}), CreateScalar<int64_t>(1), CreateScalar<int64_t>(kNumberTypeInt64), {2, 2, 3}, kInt64},
    FillOpParam{CreatePyIntTuple({3, 4}), CreateScalar<float>(2), CreateScalar<int64_t>(kNumberTypeFloat32), {3, 4}, kFloat32},
    FillOpParam{CreatePyIntTuple({kValueAny, 2, 3}), CreateScalar<int64_t>(3), CreateScalar<int64_t>(kNumberTypeFloat64), {-1, 2, 3}, kFloat64},
    FillOpParam{CreatePyIntTuple({}), CreateScalar<int>(4), CreateScalar<int64_t>(kNumberTypeInt32), {-2}, kInt32});

INSTANTIATE_TEST_CASE_P(TestFillScalar, TestFillScalar, fill_scalar_test_cases);


// Test FillTensor InferShape and InferType
class TestFillTensor : public TestOps, public testing::WithParamInterface<FillOpParam> {};

TEST_P(TestFillTensor, fill_tensor_dyn_shape) {
  const auto param = GetParam();
  auto shape_list = param.shape_value->cast<ValueTuplePtr>();
  ASSERT_NE(shape_list, nullptr);
  auto shape_dim = shape_list->size();
  auto in_shape = param.shape_value->ToAbstract();
  auto in_shape_seq = in_shape->cast<abstract::AbstractSequencePtr>();
  if (shape_dim == 0) {
    in_shape_seq->CheckAndConvertToDynamicLenSequence();
  }
  auto fill_value = param.fill_value->ToAbstract();
  auto dtype_value = param.dtype_value->ToAbstract();
  auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
  auto expect_type = std::make_shared<TensorType>(param.output_dtype);

  DoFuncImplInferAndCompare<FillTensorFuncImpl>(kNameFillTensor, {in_shape, fill_value, dtype_value}, expect_shape, expect_type);
}

tensor::TensorPtr CreateTensor(float value, TypePtr dtype) {
  auto tensor = std::make_shared<tensor::Tensor>(value, dtype);
  return tensor;
}

auto fill_tensor_test_cases = testing::Values(
    FillOpParam{CreatePyIntTuple({2, 2, 3}), CreateTensor(1, kInt64), CreateScalar<int64_t>(kNumberTypeInt64), {2, 2, 3}, kInt64},
    FillOpParam{CreatePyIntTuple({3, 4}), CreateTensor(2, kFloat32), CreateScalar<int64_t>(kNumberTypeFloat32), {3, 4}, kFloat32},
    FillOpParam{CreatePyIntTuple({kValueAny, 2, 3}), CreateTensor(3, kFloat64), CreateScalar<int64_t>(kNumberTypeFloat64), {-1, 2, 3}, kFloat64},
    FillOpParam{CreatePyIntTuple({}), CreateTensor(4, kInt32), CreateScalar<int64_t>(kNumberTypeInt32), {-2}, kInt32});

INSTANTIATE_TEST_CASE_P(TestFillTensor, TestFillTensor, fill_tensor_test_cases);
}  // namespace ops
}  // namespace mindspore
