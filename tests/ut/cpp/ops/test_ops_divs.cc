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

#include "ops/utils/general_infer_utils.h"
#include <vector>
#include <memory>
#include "ir/tensor_new.h"
#include "common/common_test.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_ops_dyn_cases.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{1, 2, 3, 4, 5}, kNumberTypeInt32},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(2.)},
    })
    .FeedExpectedOutput({{1, 2, 3, 4, 5}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({
      // input
      InferInfoParam{ShapeVector{-1, -1, -1, -1, -1, -1}, kNumberTypeInt64},
      // other
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    })
    .FeedExpectedOutput({{-1, -1, -1, -1, -1, -1}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Divs, GeneralInferTest, testing::ValuesIn(prepare_params()));

template <typename T>
tensor::TensorPtr CreateTensorPtr(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
  return tensor;
}

struct DivsInferValueParams {
  tensor::TensorPtr input;
  ValuePtr other;
  tensor::TensorPtr out;
};

class TestDivsInferValue : public TestOps, public testing::WithParamInterface<DivsInferValueParams> {};

TEST_P(TestDivsInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();
  ASSERT_NE(param.input, nullptr);
  auto input = param.input->ToAbstract();
  auto other = param.other->ToAbstract();

  auto input_args = abstract::AbstractBasePtrList{input, other};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimDivs, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "Divs have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "Divs can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestDivsInferValue, TestDivsInferValue,
  testing::Values(
    DivsInferValueParams{
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{2, 4, 6.8, 8}),
      CreatePyInt(2),
      CreateTensorPtr<float>(kNumberTypeFloat32, ShapeVector{2, 2}, std::vector<float>{1, 2, 3.4, 4})
    },
    DivsInferValueParams{
      CreateTensorPtr<double>(kNumberTypeFloat64, ShapeVector{2, 2}, std::vector<double>{2, 4, 6.8, 8}),
      CreatePyInt(2),
      CreateTensorPtr<double>(kNumberTypeFloat64, ShapeVector{2, 2}, std::vector<double>{1, 2, 3.4, 4})
    },
    DivsInferValueParams{
      CreateTensorPtr<int32_t>(kNumberTypeInt32, ShapeVector{2, 2}, std::vector<int32_t>{2, 4, 6, 8}),
      CreatePyInt(2),
      CreateTensorPtr<double>(kNumberTypeFloat64, ShapeVector{2, 2}, std::vector<double>{1, 2, 3, 4})
    }
  )
);
}  // namespace mindspore::ops
