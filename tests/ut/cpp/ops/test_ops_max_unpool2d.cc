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

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 1, 2, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, 1, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{1, 1, 3, 3}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, -1, 2, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, -1, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{1, -1, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{-1, 3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-1, 3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, 3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{1, 3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{-1, 3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-1, 3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, 3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}}})
    .FeedExpectedOutput({{1, 3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, 1, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{1, 1, 3, 3}}, {kNumberTypeFloat64});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 2, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{1, 3, 3}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{-1, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{-1, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{3, 3, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}}})
    .FeedExpectedOutput({{3, -1, -1}}, {kNumberTypeFloat64});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, 2}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, 2, 2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(1)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{1, 3, 3}}, {kNumberTypeFloat64});
  // Static input + usable output_size + kernel_size unknown at infer time: the (min, max) range implied by the
  // attributes cannot be derived, so the supplied output_size must be used directly instead of indexing an empty
  // stride vector.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 3, 4, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{1, 3, 4, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, ValuePtrList{kValueAny, kValueAny}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(8), CreateScalar<int64_t>(8)}}})
    .FeedExpectedOutput({{1, 3, 8, 8}}, {kNumberTypeFloat64});
  // `indices` with a rank different from the input is invalid: the output shape is written with the input's last dim
  // index, so a smaller rank must be rejected instead of writing past the end of the shape vector.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 3, 4, 4}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{3, 4, 4}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .CaseShouldThrow();
  // `indices` with an unknown rank must not be used as the base of the output shape, the input shape is used instead.
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, -1, -1, -1}, kNumberTypeFloat64},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(2)}},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64,
                                   ValuePtrList{CreateScalar<int64_t>(0), CreateScalar<int64_t>(0)}},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone}})
    .FeedExpectedOutput({{1, -1, -1, -1}}, {kNumberTypeFloat64});
  return generator.Generate();
}
}  // namespace
INSTANTIATE_TEST_CASE_P(MaxUnpool2DExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
