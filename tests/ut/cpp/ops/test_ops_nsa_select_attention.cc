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

#include "ops/utils/general_infer_utils.h"
#include "op_enum.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  // Case 1: Basic fixed shapes - attention_out=[query[0], query[1], value[2]], softmax=[query[0], query[1], 8]
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{8, 1024, 128}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{8, 512, 128}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{8, 512, 64}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{8, 1024, 16}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.125)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({1024, 1024})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({512, 512})},})
    .FeedExpectedOutput({{8, 1024, 64}, {8, 1024, 8}, {8, 1024, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  // Case 2: Partial dynamic dims - query partially dynamic, value fixed
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 128}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, -1, 128}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, 512, 64}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2, -1, 16}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0625)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(128)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{2, -1, 512}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({1024, 1024})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({512, 512})},})
    .FeedExpectedOutput({{2, -1, 64}, {2, -1, 8}, {2, -1, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  // Case 3: Fully dynamic dims - both query and value are fully dynamic
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.125)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({512, 512})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({256, 256})},})
    .FeedExpectedOutput({{-1, -1, -1}, {-1, -1, 8}, {-1, -1, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  // Case 4: Dynamic rank - query has dynamic rank, the first two dims of attention_out are kShapeDimAny
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4, 512, 128}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{4, 512, 64}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.125)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({512, 512})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({256, 256})},})
    .FeedExpectedOutput({{-1, -1, 64}, {-1, -1, 8}, {-1, -1, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  // Case 5: Dynamic rank on value - the third dim of attention_out is kShapeDimAny
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 512, 128}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 256, 128}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2, 512, 8}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.25)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{2, 512, 256}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({512, 512})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({256, 256})},})
    .FeedExpectedOutput({{2, 512, -1}, {2, 512, 8}, {2, 512, 8}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  // Case 6: Mixed dynamic - query partially dynamic, value has dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, -1, 128}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{1, 64, 128}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{1, -1, 4}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({128,})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({64,})},})
    .FeedExpectedOutput({{1, -1, -1}, {1, -1, 8}, {1, -1, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(NsaSelectAttention, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
