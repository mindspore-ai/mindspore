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

#include <vector>
#include <memory>

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "primitive/op_name.h"
#include "infer/ops_func_impl/flash_attention_score.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"
#include "ops/utils/general_infer_param.h"
#include "ops/op_def.h"
#include "ir/value.h"

namespace mindspore {
namespace ops {

// Forward declarations
static std::vector<GeneralInferParam> prepare_params_flash_attention();

constexpr ShapeValueDType kShapeRankAny = mindspore::abstract::Shape::kShapeRankAny;
constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;

static std::vector<GeneralInferParam> prepare_params_flash_attention() {
  GeneralInferParamGenerator gen;

  // Case 0: BSH, fp16, keep_prob=1.0, outputs: [B,N1,S1,8], [B,N1,S1,8], [1], [B,S1,H1]
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8;  // H1=N1*D=32
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // real_shift=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // drop_mask=None (keep_prob=1)
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // padding_mask=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // attn_mask=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // prefix=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // actual_seq_qlen=None
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},  // actual_seq_kvlen=None
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // BSH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // sparse_mode=0
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 1: BSH, bfloat16, keep_prob=1.0
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeBFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeBFloat16, kNumberTypeBFloat16});
  }

  // Case 2: BSH with attn_mask bool [S1, S1]
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{S1, S1}, kNumberTypeBool},  // attn_mask bool
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 3: BSH, known rank with dynamic dims [-1,-1,-1]
  {
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},  // BSH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{-1, 1, -1, 8}, ShapeVector{-1, 1, -1, 8}, ShapeVector{1},
                            ShapeVector{-1, -1, -1}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 4: TND, fp16, actual_seq_qlen/kvlen as tuple (ValuePtrList cumulative)
  {
    const int64_t T = 128, N = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                        ValuePtrList{CreateScalar<int64_t>(32), CreateScalar<int64_t>(64),
                                     CreateScalar<int64_t>(96), CreateScalar<int64_t>(128)}},
         InferInfoParam{ShapeArray{{}}, kNumberTypeInt64,
                        ValuePtrList{CreateScalar<int64_t>(32), CreateScalar<int64_t>(64),
                                     CreateScalar<int64_t>(96), CreateScalar<int64_t>(128)}},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},  // TND
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{T, N, 8}, ShapeVector{T, N, 8}, ShapeVector{1}, ShapeVector{T, N, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 5: BNSD, fp16
  {
    const int64_t B = 2, N1 = 4, S1 = 16, D = 8, N2 = 2;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, N1, S1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, N2, S1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, N2, S1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},  // BNSD
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, N1, S1, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 6: SBH, fp16
  {
    const int64_t S1 = 16, B = 2, N1 = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{S1, B, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{S1, B, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{S1, B, N1 * D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},  // SBH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{S1, B, N1 * D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 7: BSND, fp16
  {
    const int64_t B = 2, S1 = 16, N1 = 4, D = 8, N2 = 2;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{B, S1, N1, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N2, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{B, S1, N2, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N1)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},  // BSND
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{B, N1, S1, 8}, ShapeVector{B, N1, S1, 8}, ShapeVector{1},
                            ShapeVector{B, S1, N1, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 8: TND, fp16, softmax [T,N,8]; attention [T,N,D]
  {
    const int64_t T = 128, N = 4, D = 8;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, N, D}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         // actual_seq_qlen / kvlen will be provided as scalars below
         // actual_seq_qlen / kvlen (scalars or arrays of cumulative)
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(N)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},  // TND
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{T, N, 8}, ShapeVector{T, N, 8}, ShapeVector{1}, ShapeVector{T, N, D}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 9: TH, fp16, softmax [T, head_num*8]; attention [T, H]
  {
    const int64_t T = 128, H = 64, head_num = 2;
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{T, H}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, H}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{T, H}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(T)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(head_num)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},  // TH
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{T, head_num * 8}, ShapeVector{T, head_num * 8}, ShapeVector{1},
                            ShapeVector{T, H}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  // Case 10: Dynamic rank query, BSH
  {
    gen.FeedInputArgs({
         InferInfoParam{ShapeVector{kShapeRankAny}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{2, 16, 32}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kMetaTypeNone, mindspore::kNone},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0f)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
         InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
       })
       .FeedExpectedOutput({ShapeVector{kShapeDimAny, kShapeDimAny, kShapeDimAny, 8}, ShapeVector{kShapeDimAny, kShapeDimAny, kShapeDimAny, 8},
                            ShapeVector{1}, ShapeVector{kShapeRankAny}},
                           {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  }

  return gen.Generate();
}
INSTANTIATE_TEST_CASE_P(FlashAttentionScore, GeneralInferTest,
                        testing::ValuesIn(prepare_params_flash_attention()));

}  // namespace ops
}  // namespace mindspore
