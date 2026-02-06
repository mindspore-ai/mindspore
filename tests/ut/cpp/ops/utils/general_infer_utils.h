/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_TESTS_UT_CPP_OPS_UTILS_GENERAL_INFER_UTILS
#define MINDSPORE_TESTS_UT_CPP_OPS_UTILS_GENERAL_INFER_UTILS

#include <iostream>
#include <memory>
#include <string>

#include "common/common_test.h"
#include "ops/utils/general_infer_param.h"
#include "ops/test_value_utils.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "ops/op_def.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
InferInfoPtr param_to_abstract_info(InferInfoParam param, const std::string &op_type, const std::string &arg_name);

InferInfoPtr param_to_value_info(InferInfoParam param, const std::string &op_type, const std::string &arg_name);

std::pair<InferInfoPtrList, InferInfoPtrList> params_to_infos(GeneralInferParam params, const OpDef op_def);

class GeneralInferTest : public testing::TestWithParam<GeneralInferParam> {};

static std::vector<GeneralInferParam> single_input_eltwise_op_params() {
  GeneralInferParamGenerator generator;
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 6}, kNumberTypeFloat32}})
      .FeedExpectedOutput({{2, 3, 6}}, {kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{6}, kNumberTypeFloat16}})
      .FeedExpectedOutput({{6}}, {kNumberTypeFloat16});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{1, 1, 1, 1, 1, 1, 1, 8}, kNumberTypeFloat32}})
      .FeedExpectedOutput({{1, 1, 1, 1, 1, 1, 1, 8}}, {kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, -1}, kNumberTypeFloat64}})
      .FeedExpectedOutput({{2, -1}}, {kNumberTypeFloat64});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32}})
      .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  return generator.Generate();
}

static auto single_input_eltwise_op_default_cases = testing::ValuesIn(single_input_eltwise_op_params());
}  // namespace mindspore::ops
#endif  // MINDSPORE_TESTS_UT_CPP_OPS_UTILS_GENERAL_INFER_UTILS
