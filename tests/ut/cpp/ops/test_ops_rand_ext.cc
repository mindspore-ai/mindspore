/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <string>
#include <vector>
#include "ops/utils/general_infer_utils.h"
#include "ops/test_ops.h"
#include "ops/op_def.h"
#include "include/op_enum.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(DEVICE_META_LOWER)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(kNumberTypeFloat64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(DEVICE_META_LOWER)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(DEVICE_CPU)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  return generator.Generate();
}

InferInfoParam AlignInferInfoParam(InferInfoParam param) {
  const bool is_sequence_shape = std::holds_alternative<ShapeArray>(param.shape);
  const bool is_sequence_type = std::holds_alternative<TypeIdList>(param.type);
  const bool is_sequence_value = std::holds_alternative<ValuePtrList>(param.value);
  if (!is_sequence_shape && !is_sequence_type && !is_sequence_value) {
    return param;
  }
  if (!is_sequence_shape) {
    param.shape = ShapeArray{std::get<ShapeVector>(param.shape)};
  }
  if (!is_sequence_type) {
    param.type = TypeIdList{std::get<TypeId>(param.type)};
  }
  if (!is_sequence_value) {
    param.value = ValuePtrList{std::get<ValuePtr>(param.value)};
  }

  auto &shapes = std::get<ShapeArray>(param.shape);
  auto &types = std::get<TypeIdList>(param.type);
  auto &values = std::get<ValuePtrList>(param.value);
  const size_t size = std::max(shapes.size(), std::max(types.size(), values.size()));
  if (shapes.size() == 1) {
    shapes = ShapeArray(size, shapes[0]);
  }
  if (types.size() == 1) {
    types = TypeIdList(size, types[0]);
  }
  if (values.size() == 1) {
    values = ValuePtrList(size, values[0]);
  }
  return param;
}

InferInfoPtrList BuildInferInfos(const std::string &op_type, const std::vector<InferInfoParam> &params,
                                 bool use_value) {
  InferInfoPtrList infos;
  const auto &op_def = GetOpDef(op_type);
  MS_EXCEPTION_IF_NULL(op_def);
  for (size_t i = 0; i < params.size(); ++i) {
    auto aligned_param = AlignInferInfoParam(params[i]);
    const auto &arg_name = op_def->args_[i].arg_name_;
    infos.push_back(use_value ? param_to_value_info(aligned_param, op_type, arg_name)
                              : param_to_abstract_info(aligned_param, op_type, arg_name));
  }
  return infos;
}
}  // namespace

INSTANTIATE_TEST_CASE_P(RandExt, GeneralInferTest, testing::ValuesIn(prepare_params()));

class TestRandExtContext : public TestOps {};

/// Feature: RandExt infer type validation with implicit device context.
/// Description: Infer RandExt type when device input is None and runtime context is Meta.
/// Expectation: Inference succeeds and returns float32 for both abstract and value paths.
TEST_F(TestRandExtContext, infer_type_accepts_meta_context_when_device_is_none) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kMetaDevice);

  std::vector<InferInfoParam> params{
    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
  };
  auto prim = std::make_shared<Primitive>("RandExt");
  const auto &op_def = GetOpDef("RandExt");
  ASSERT_NE(op_def, nullptr);

  auto abstract_infos = BuildInferInfos("RandExt", params, false);
  auto value_infos = BuildInferInfos("RandExt", params, true);

  EXPECT_NO_THROW({
    const auto &abstract_types = op_def->func_impl_.InferType(prim, abstract_infos);
    ASSERT_EQ(abstract_types.size(), 1);
    EXPECT_EQ(abstract_types[0], kNumberTypeFloat32);
  });
  EXPECT_NO_THROW({
    const auto &value_types = op_def->func_impl_.InferType(prim, value_infos);
    ASSERT_EQ(value_types.size(), 1);
    EXPECT_EQ(value_types[0], kNumberTypeFloat32);
  });
}

/// Feature: RandExt infer type validation with implicit device context.
/// Description: Infer RandExt type when device input is None and runtime context is CPU.
/// Expectation: Inference succeeds because backend device validation is deferred to execution.
TEST_F(TestRandExtContext, infer_type_accepts_cpu_context_when_device_is_none) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kCPUDevice);

  std::vector<InferInfoParam> params{
    InferInfoParam{ShapeArray{{}, {}}, kNumberTypeInt64,
                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(3)}},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
  };
  auto prim = std::make_shared<Primitive>("RandExt");
  const auto &op_def = GetOpDef("RandExt");
  ASSERT_NE(op_def, nullptr);

  auto abstract_infos = BuildInferInfos("RandExt", params, false);
  auto value_infos = BuildInferInfos("RandExt", params, true);

  EXPECT_NO_THROW({
    const auto &abstract_types = op_def->func_impl_.InferType(prim, abstract_infos);
    ASSERT_EQ(abstract_types.size(), 1);
    EXPECT_EQ(abstract_types[0], kNumberTypeFloat32);
  });
  EXPECT_NO_THROW({
    const auto &value_types = op_def->func_impl_.InferType(prim, value_infos);
    ASSERT_EQ(value_types.size(), 1);
    EXPECT_EQ(value_types[0], kNumberTypeFloat32);
  });
}

/// Feature: RandLikeExt infer type validation with implicit device context.
/// Description: Infer RandLikeExt type when device input is None and runtime context is Meta.
/// Expectation: Abstract inference succeeds and preserves the input float32 dtype.
TEST_F(TestRandExtContext, rand_like_infer_type_accepts_meta_context_when_device_is_none) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kMetaDevice);

  std::vector<InferInfoParam> params{
    InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
  };
  auto prim = std::make_shared<Primitive>("RandLikeExt");
  const auto &op_def = GetOpDef("RandLikeExt");
  ASSERT_NE(op_def, nullptr);

  auto abstract_infos = BuildInferInfos("RandLikeExt", params, false);

  EXPECT_NO_THROW({
    const auto &abstract_types = op_def->func_impl_.InferType(prim, abstract_infos);
    ASSERT_EQ(abstract_types.size(), 1);
    EXPECT_EQ(abstract_types[0], kNumberTypeFloat32);
  });
}

/// Feature: RandLikeExt infer type validation with implicit device context.
/// Description: Infer RandLikeExt type when device input is None and runtime context is CPU.
/// Expectation: Abstract inference succeeds because backend device validation is deferred to execution.
TEST_F(TestRandExtContext, rand_like_infer_type_accepts_cpu_context_when_device_is_none) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kCPUDevice);

  std::vector<InferInfoParam> params{
    InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, mindspore::kNone},
  };
  auto prim = std::make_shared<Primitive>("RandLikeExt");
  const auto &op_def = GetOpDef("RandLikeExt");
  ASSERT_NE(op_def, nullptr);

  auto abstract_infos = BuildInferInfos("RandLikeExt", params, false);

  EXPECT_NO_THROW({
    const auto &abstract_types = op_def->func_impl_.InferType(prim, abstract_infos);
    ASSERT_EQ(abstract_types.size(), 1);
    EXPECT_EQ(abstract_types[0], kNumberTypeFloat32);
  });
}
}  // namespace mindspore::ops
