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

#include <vector>
#include <tuple>
#include <memory>
#include <string>

#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/anf.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "infer/ctc_loss_v2.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace ops {
struct CTCLossV2Params {
  // input shapes
  ShapeVector log_probs_shape;
  ShapeVector targets_shape;
  ShapeVector input_lengths_shape;
  ShapeVector target_lengths_shape;
  // attr
  int64_t blank;
  // input target_lengths value/type
  ValuePtr target_lengths_value;
  TypePtr target_lengths_type;
  // expect output shapes
  ShapeVector neg_log_shape;
  ShapeVector log_alpha_shape;
  // true: expect run success, false: expect exception
  bool is_success;
};

struct CTCLossV2Types {
  // input types
  TypePtr log_probs_type;
  TypePtr targets_type;
  TypePtr input_lengths_type;
  // expect output types
  TypePtr neg_log_type;
  TypePtr log_alpha_type;
  bool is_success;
};

class TestCTCLossV2 : public TestOps,
                      public testing::WithParamInterface<std::tuple<CTCLossV2Params, CTCLossV2Types>> {};

TEST_P(TestCTCLossV2, CTCLossV2) {
  auto context_ptr = MsContext::GetInstance();
  UT_CHECK_NULL(context_ptr);
  context_ptr->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  const auto &input_params = std::get<0>(GetParam());
  const auto &input_types = std::get<1>(GetParam());

  auto log_probs = std::make_shared<abstract::AbstractTensor>(input_types.log_probs_type, input_params.log_probs_shape);
  auto targets = std::make_shared<abstract::AbstractTensor>(input_types.targets_type, input_params.targets_shape);
  auto input_lengths =
    std::make_shared<abstract::AbstractTensor>(input_types.input_lengths_type, input_params.input_lengths_shape);
  UT_CHECK_NULL(log_probs);
  UT_CHECK_NULL(targets);
  UT_CHECK_NULL(input_lengths);
  auto target_lengths = abstract::MakeAbstract(std::make_shared<abstract::Shape>(input_params.target_lengths_shape),
                                               input_params.target_lengths_type);
  UT_CHECK_NULL(target_lengths);
  if (input_params.target_lengths_value != nullptr) {
    target_lengths->set_value(input_params.target_lengths_value);
  }

  auto prim = std::make_shared<Primitive>(kNameCTCLossV2);
  UT_CHECK_NULL(prim);
  prim->set_attr(kAttrBlank, MakeValue<int64_t>(input_params.blank));
  if (input_params.is_success && input_types.is_success) {
    auto infer_result_abs = CTCLossV2Infer(nullptr, prim, {log_probs, targets, input_lengths, target_lengths});
    UT_CHECK_NULL(infer_result_abs);
    auto expect_neg_log_shape = std::make_shared<abstract::Shape>(input_params.neg_log_shape);
    auto expect_log_alpha_shape = std::make_shared<abstract::Shape>(input_params.log_alpha_shape);
    UT_CHECK_NULL(expect_neg_log_shape);
    UT_CHECK_NULL(expect_log_alpha_shape);
    auto expect_shape = std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{expect_neg_log_shape, expect_log_alpha_shape});
    UT_CHECK_NULL(expect_shape);
    auto expect_type =
      std::make_shared<Tuple>(std::vector<TypePtr>{std::make_shared<TensorType>(input_types.neg_log_type),
                                                   std::make_shared<TensorType>(input_types.log_alpha_type)});
    UT_CHECK_NULL(expect_type);
    ShapeCompare(infer_result_abs->GetShape(), expect_shape);
    TypeCompare(infer_result_abs->GetType(), expect_type);
  } else {
    ASSERT_ANY_THROW(CTCLossV2Infer(nullptr, prim, {log_probs, targets, input_lengths, target_lengths}));
  }
}

INSTANTIATE_TEST_CASE_P(TestCTCLossV2, TestCTCLossV2,
                        testing::Combine(
                          testing::ValuesIn({
                            CTCLossV2Params{{8, 8, 9}, {8, 8}, {8}, {8}, 7, nullptr, kInt32, {8}, {8, 8, -1}, true},
                            CTCLossV2Params{{-2}, {8, 8}, {8}, {8}, 0, nullptr, kInt32, {-2}, {-2}, true},
                            CTCLossV2Params{{8, 8, 9}, {8, -1}, {8}, {8}, 6, nullptr, kInt32, {8}, {8, 8, -1}, true},
                            CTCLossV2Params{{8, 8, 9},
                                            {8, 8},
                                            {8},
                                            {8},
                                            6,
                                            CreateTensor<int>(kNumberTypeInt32, {8}, {1, 2, 3, 4, 5, 6, 6, 6}),
                                            kInt32,
                                            {8},
                                            {8, 8, 16},
                                            true},
                            CTCLossV2Params{{8, 8, 9},
                                            {8, 8},
                                            {8},
                                            {8},
                                            8,
                                            CreateScalarTuple<int64_t>({1, 2, 3, 4, 5, 6, 6, 6}),
                                            kInt64,
                                            {8},
                                            {8, 8, 16},
                                            true},
                            CTCLossV2Params{{8, 8, 9}, {8, 8}, {8, 9}, {8}, 7, nullptr, kInt32, {8}, {8, 8, -1}, false},
                            CTCLossV2Params{{8, 8, 9}, {8, 8}, {8}, {8}, 10, nullptr, kInt32, {8}, {8, 8, -1}, false},
                          }),
                          testing::ValuesIn({
                            CTCLossV2Types{kFloat32, kInt32, kInt32, kFloat32, kFloat32, true},
                            CTCLossV2Types{kFloat64, kInt32, kInt32, kFloat64, kFloat64, true},
                            CTCLossV2Types{kFloat16, kInt32, kInt32, kFloat32, kFloat32, false},
                            CTCLossV2Types{kFloat32, kInt32, kInt8, kFloat32, kFloat32, false},
                          })));

}  // namespace ops
}  // namespace mindspore
