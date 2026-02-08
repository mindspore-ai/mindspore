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
#include "ops/utils/general_infer_param.h"
#include "ops/op_def.h"
#include "ir/value.h"

namespace mindspore::ops {
namespace {
constexpr ShapeValueDType kShapeRankAny = mindspore::abstract::Shape::kShapeRankAny;
constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;

// Shapes: input [T, N, D], weight [block, N]; Scalars: block, stride; Tuple: actual_seq_len

static std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator gen;
  // Case 0: valid
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32), CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{4, 4, 128}}, {kNumberTypeFloat16});

  // Case 1: dynamic T
  gen.FeedInputArgs({InferInfoParam{ShapeVector{kShapeDimAny, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32), CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{4, 4, 128}}, {kNumberTypeFloat16});

  // Case 2: dynamic N
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, kShapeDimAny, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32), CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{4, kShapeDimAny, 128}}, {kNumberTypeFloat16});

  // Case 3: dynamic D
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, 4, kShapeDimAny}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32), CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{4, 4, kShapeDimAny}}, {kNumberTypeFloat16});

  // Case 4: dynamic rank input
  gen.FeedInputArgs({InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32), CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{kShapeRankAny}}, {kNumberTypeFloat16});

  // Case 5: actual_seq_len unknown -> first dim unknown
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{kValueAny}}})
    .FeedExpectedOutput({ShapeVector{kShapeDimAny, 4, 128}}, {kNumberTypeFloat16});

  // Case 6: block/stride unknown (ValueAny), seq known -> first dim unknown
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{kShapeDimAny, 4, 128}}, {kNumberTypeFloat16});

  // Case 7: block known, stride unknown, seq known -> first dim unknown
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{kShapeDimAny, 4, 128}}, {kNumberTypeFloat16});

  // Case 8: both block/stride known, seq contains unknown value -> first dim unknown
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16), kValueAny}}})
    .FeedExpectedOutput({ShapeVector{kShapeDimAny, 4, 128}}, {kNumberTypeFloat16});

  // Case 9: both block/stride known, seq known -> exact first dim
  gen.FeedInputArgs({InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{32, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                     InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{0, 4, 128}}, {kNumberTypeFloat16});

  return gen.Generate();
}
}  // namespace

// NsaCompress GeneralInfer UT
INSTANTIATE_TEST_CASE_P(NsaCompress, GeneralInferTest, testing::ValuesIn(prepare_params()));

namespace {
static std::vector<GeneralInferParam> prepare_params_grad() {
  GeneralInferParamGenerator gen;

  // G0: valid shapes; grad shape is arbitrary here
  gen.FeedInputArgs({InferInfoParam{ShapeVector{4, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{64, 4, 128}, ShapeVector{16, 4}},
                        {kNumberTypeFloat16, kNumberTypeFloat16});

  // G1: input dynamic rank
  gen.FeedInputArgs({InferInfoParam{ShapeVector{4, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{kShapeRankAny}, ShapeVector{16, 4}},
                        {kNumberTypeFloat16, kNumberTypeFloat16});

  // G2: weight dynamic rank
  gen.FeedInputArgs({InferInfoParam{ShapeVector{4, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{64, 4, 128}, ShapeVector{kShapeRankAny}},
                        {kNumberTypeFloat16, kNumberTypeFloat16});

  // G3: input dynamic N
  gen.FeedInputArgs({InferInfoParam{ShapeVector{4, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{64, kShapeDimAny, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{64, kShapeDimAny, 128}, ShapeVector{16, 4}},
                        {kNumberTypeFloat16, kNumberTypeFloat16});

  // G4: input dynamic D
  gen.FeedInputArgs({InferInfoParam{ShapeVector{4, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{64, 4, kShapeDimAny}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{64, 4, kShapeDimAny}, ShapeVector{16, 4}},
                        {kNumberTypeFloat16, kNumberTypeFloat16});

  // G5: actual_seq_len unknown value
  gen.FeedInputArgs({InferInfoParam{ShapeVector{4, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{64, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{16, 4}, kNumberTypeFloat16},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{kValueAny}}})
    .FeedExpectedOutput({ShapeVector{64, 4, 128}, ShapeVector{16, 4}},
                        {kNumberTypeFloat16, kNumberTypeFloat16});

  // G6: both input and weight dynamic rank
  gen.FeedInputArgs({InferInfoParam{ShapeVector{4, 4, 128}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                     InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64, ValuePtrList{CreateScalar<int64_t>(16)}},
                     InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                                    ValuePtrList{CreateScalar<int64_t>(16), CreateScalar<int64_t>(32),
                                                 CreateScalar<int64_t>(48), CreateScalar<int64_t>(64)}}})
    .FeedExpectedOutput({ShapeVector{kShapeRankAny}, ShapeVector{kShapeRankAny}},
                        {kNumberTypeFloat16, kNumberTypeFloat16});

  return gen.Generate();
}
}  // namespace

// NsaCompressGrad GeneralInfer UT
INSTANTIATE_TEST_CASE_P(NsaCompressGrad, GeneralInferTest, testing::ValuesIn(prepare_params_grad()));

}  // namespace mindspore::ops
