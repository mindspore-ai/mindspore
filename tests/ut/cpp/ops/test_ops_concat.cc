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
#include "ops/test_ops.h"
#include "ops/test_ops_cmp_utils.h"
#include "ops/test_value_utils.h"
#include "ir/tensor_new.h"
#include "abstract/ops/primitive_infer_map.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, 2, 4}, {3, 5, 4}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{3, 7, 4}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, -1, 5}, {-1, -1, -1}, {3, 4, -1}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-2}, {-2}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-2}, {-2}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{2, 3, 4}}, kNumberTypeFloat32, kValueAny, true},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{2, -1, 4}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{2, 3, 4}}, kNumberTypeFloat32, kValueAny, true},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-1, -1}}, kNumberTypeFloat32, kValueAny, true},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-1, -1}}, kNumberTypeFloat32, kValueAny, true},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-2}}, kNumberTypeFloat32, kValueAny, true},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-2}}, kNumberTypeFloat32, kValueAny, true},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, 4, 5}, {3, 4, 5}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, 4, 5}, {-1, 4, 5}, {3, 4, -1}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{2, 3, 4}, {2, -1, -1}, {-1, -1, 5}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)}})
    .FeedExpectedOutput({{2, 3, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-2}, {2, -1, -1}, {-1, 4, -1}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)}})
    .FeedExpectedOutput({{2, 4, -1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{-1, 6, 3}, {5, -1, 4}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)}})
    .FeedExpectedOutput({{5, 6, 7}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, 4, 5}, {3, 4, 4}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny}})
    .FeedExpectedOutput({{3, 4, 9}}, {kNumberTypeFloat32});
  //  cases expect exception
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{2, 3, 4}}, kNumberTypeFloat32, kValueAny, true},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)}})
    .CaseShouldThrow();
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, 2, 4}, {3, 5, 4}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)}})
    .CaseShouldThrow();
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, 2, 4}, {3, 5, 4}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-4)}})
    .CaseShouldThrow();
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{3, 2, 4}, {3, 5, 4}}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)}})
    .CaseShouldThrow();
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Concat, GeneralInferTest, testing::ValuesIn(prepare_params()));

struct ConcatInferValueParams {
  std::vector<tensor::TensorPtr> input_tensors;
  ValuePtr axis;
  tensor::TensorPtr out;
};

static tensor::TensorPtr CreateTensor(const ShapeVector &shape, std::vector<float> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(kNumberTypeFloat32, shape, data_ptr, kNumberTypeFloat32);
  return tensor;
}

class TestConcatInferValue : public TestOps, public testing::WithParamInterface<ConcatInferValueParams> {};

TEST_P(TestConcatInferValue, dyn_shape_infer_value) {
  const auto &param = GetParam();

  auto input_tensors = param.input_tensors;
  AbstractBasePtrList input_elements;
  for (auto tensor : input_tensors) {
    ASSERT_NE(tensor, nullptr);
    auto x = tensor->ToAbstract();
    ASSERT_NE(x, nullptr);
    input_elements.push_back(x);
  }

  auto tensors = std::make_shared<abstract::AbstractTuple>(input_elements);
  ASSERT_NE(tensors, nullptr);

  ASSERT_NE(param.axis, nullptr);
  auto axis = param.axis->ToAbstract();
  ASSERT_NE(axis, nullptr);

  abstract::AbstractBasePtrList input_args = {tensors, axis};
  auto value_opt = abstract::InferValueByFuncImpl(prim::kPrimConcat, input_args);
  if (!value_opt.has_value()) {
    MS_LOG(ERROR) << "Concat have no infer value implement!";
    ASSERT_TRUE(false);
  }
  auto infer_out = value_opt.value();
  if (infer_out == nullptr) {
    MS_LOG(ERROR) << "Concat can not infer value with inputs: " << input_args;
    ASSERT_TRUE(false);
  }
  auto infer_tensor = infer_out->cast<tensor::TensorPtr>();
  ASSERT_NE(infer_tensor, nullptr);
  ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));
}

INSTANTIATE_TEST_CASE_P(
  TestConcatInferValue, TestConcatInferValue,
  testing::Values(ConcatInferValueParams{{CreateTensor(ShapeVector{2}, std::vector<float>{1, 2}),
                                          CreateTensor(ShapeVector{2}, std::vector<float>{3, 4})},
                                         CreateScalar<int64_t>(0),
                                         CreateTensor(ShapeVector{4}, std::vector<float>{1, 2, 3, 4})},
                  ConcatInferValueParams{
                    {CreateTensor(ShapeVector{2, 2}, std::vector<float>{1, 2, 3, 4}),
                     CreateTensor(ShapeVector{2, 3}, std::vector<float>{5, 6, 7, 8, 9, 10})},
                    CreateScalar<int64_t>(1),
                    CreateTensor(ShapeVector{2, 5}, std::vector<float>{1, 2, 5, 6, 7, 3, 4, 8, 9, 10})}));
}  // namespace mindspore::ops
