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
  // Case 1: Basic fixed shapes - gradient outputs match corresponding input shapes
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{8, 1024, 64}, kNumberTypeBFloat16},  // grad
                    InferInfoParam{ShapeVector{8, 1024, 128}, kNumberTypeBFloat16}, // query
                    InferInfoParam{ShapeVector{8, 512, 128}, kNumberTypeBFloat16},  // key
                    InferInfoParam{ShapeVector{8, 512, 64}, kNumberTypeBFloat16},   // value
                    InferInfoParam{ShapeVector{8, 1024, 64}, kNumberTypeBFloat16},  // attention_out
                    InferInfoParam{ShapeVector{8, 1024, 8}, kNumberTypeFloat32},    // softmax_max
                    InferInfoParam{ShapeVector{8, 1024, 8}, kNumberTypeFloat32},    // softmax_sum
                    InferInfoParam{ShapeVector{8, 1024, 16}, kNumberTypeInt32},     // topk_indices
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.125)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},})
    .FeedExpectedOutput({{8, 1024, 128}, {8, 512, 128}, {8, 512, 64}},  // query_grad, key_grad, value_grad
                        {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});
  // Case 2: Partial dynamic dims - query partially dynamic, key/value fixed
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, 64}, kNumberTypeFloat16},    // grad
                    InferInfoParam{ShapeVector{2, -1, 128}, kNumberTypeFloat16},   // query
                    InferInfoParam{ShapeVector{2, 512, 128}, kNumberTypeFloat16},  // key 
                    InferInfoParam{ShapeVector{2, 512, 64}, kNumberTypeFloat16},   // value
                    InferInfoParam{ShapeVector{2, -1, 64}, kNumberTypeFloat16},    // attention_out
                    InferInfoParam{ShapeVector{2, -1, 8}, kNumberTypeFloat32},     // softmax_max
                    InferInfoParam{ShapeVector{2, -1, 8}, kNumberTypeFloat32},     // softmax_sum
                    InferInfoParam{ShapeVector{2, -1, 16}, kNumberTypeInt32},      // topk_indices
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.0625)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(128)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{2, -1, 512}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({1024, 1024})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({512, 512})},})
    .FeedExpectedOutput({{2, -1, 128}, {2, 512, 128}, {2, 512, 64}},
                        {kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
  // Case 3: Fully dynamic dims - all inputs are fully dynamic
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16}, // grad
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16}, // query
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16}, // key
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16}, // value
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeBFloat16}, // attention_out
                    InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},   // softmax_max
                    InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},   // softmax_sum
                    InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeInt32},    // topk_indices
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.125)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},})
    .FeedExpectedOutput({{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
                        {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});
  // Case 4: Dynamic rank - all main inputs have dynamic rank
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},         // grad
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},         // query
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},         // key
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},         // value
                    InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},         // attention_out
                    InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},   // softmax_max
                    InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},   // softmax_sum
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},            // topk_indices
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.125)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(64)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},})
    .FeedExpectedOutput({{-2}, {-2}, {-2}},
                        {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});
  // Case 5: Mixed dynamic types - Float32 with attention mask
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 512, -1}, kNumberTypeFloat32},   // grad
                    InferInfoParam{ShapeVector{2, 512, 128}, kNumberTypeFloat32},  // query
                    InferInfoParam{ShapeVector{2, 256, 128}, kNumberTypeFloat32},  // key
                    InferInfoParam{ShapeVector{2, 256, -1}, kNumberTypeFloat32},   // value
                    InferInfoParam{ShapeVector{2, 512, -1}, kNumberTypeFloat32},   // attention_out
                    InferInfoParam{ShapeVector{2, 512, 8}, kNumberTypeFloat32},    // softmax_max
                    InferInfoParam{ShapeVector{2, 512, 8}, kNumberTypeFloat32},    // softmax_sum
                    InferInfoParam{ShapeVector{2, 512, 8}, kNumberTypeInt32},      // topk_indices
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.25)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(32)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)},
                    InferInfoParam{ShapeVector{2, 512, 256}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({512, 512})},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({256, 256})},})
    .FeedExpectedOutput({{2, 512, 128}, {2, 256, 128}, {2, 256, -1}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  // Case 6: Mixed dynamic - query partially dynamic, key/value with different shapes
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, -1, -1}, kNumberTypeBFloat16},   // grad
                    InferInfoParam{ShapeVector{1, -1, 128}, kNumberTypeBFloat16},  // query
                    InferInfoParam{ShapeVector{1, 64, 128}, kNumberTypeBFloat16},  // key
                    InferInfoParam{ShapeVector{1, 64, -1}, kNumberTypeBFloat16},   // value
                    InferInfoParam{ShapeVector{1, -1, -1}, kNumberTypeBFloat16},   // attention_out
                    InferInfoParam{ShapeVector{1, -1, 8}, kNumberTypeFloat32},     // softmax_max
                    InferInfoParam{ShapeVector{1, -1, 8}, kNumberTypeFloat32},     // softmax_sum
                    InferInfoParam{ShapeVector{1, -1, 4}, kNumberTypeInt32},       // topk_indices
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(0.5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(16)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
                    InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},})
    .FeedExpectedOutput({{1, -1, 128}, {1, 64, 128}, {1, 64, -1}},
                        {kNumberTypeBFloat16, kNumberTypeBFloat16, kNumberTypeBFloat16});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(NsaSelectAttentionGrad, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
