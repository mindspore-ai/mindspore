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

#include "infer/ops_func_impl/mla.h"
#include <set>
#include <string>
#include <utility>

#include "primitive/nn_ops.h"
#include "include/op_enum.h"
#include "utils/check_convert_utils.h"
#include "mindapi/helper.h"

namespace mindspore {
namespace ops {
static constexpr auto kMLAQshapeRank = 3;
static constexpr auto kMLAKVshapeRank = 4;
static constexpr auto kMLABlockSizeDim = 1;
static constexpr auto kMLABlockTablesRank = 2;
static constexpr auto kMLAMaskRank = 2;
static constexpr auto kMLADeqScaleRank = 1;
static constexpr auto kMLAMaskFreeLastDim = 128;
static constexpr auto kMLAQKVnopeHiddenSize = 512;
static constexpr auto kMLAQKropeHiddenSize = 64;
static constexpr auto kMLAQheadMax = 128;
static constexpr auto kMLABlockSizeheadMax = 128;

#define ALIGN_16(v) (((v) & (16 - 1)) == 0)

static void CheckParam(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  auto kv_heads = input_infos[kMlaInputNumKVHeadIndex]->GetScalarValue<int64_t>();
  if (kv_heads.has_value()) {
    MS_CHECK_VALUE(kv_heads.value() == 1, CheckAndConvertUtils::FormatCommMsg(
                                            "For MLA The kv_head_num must be 1 , but got : ", kv_heads.value()));
  }

  auto q_heads = input_infos[kMlaInputNumHeadIndex]->GetScalarValue<int64_t>();
  if (q_heads.has_value()) {
    MS_CHECK_VALUE(q_heads.value() <= kMLAQheadMax,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The head_num must be <= ", kMLAQheadMax,
                                                       ", but got : ", q_heads.value()));
    MS_CHECK_VALUE(ALIGN_16(q_heads.value()),
                   CheckAndConvertUtils::FormatCommMsg("For MLA The head_num must be the multiple of 16, but got : ",
                                                       q_heads.value()));

    if (q_heads.value() == kMLAQheadMax) {
      auto q_nope_type = input_infos[kMlaInputQnopeIndex]->GetType();
      if (q_nope_type == kNumberTypeInt8) {
        MS_LOG(EXCEPTION) << "For MLA int8 is not support when head_num=128.";
      }
    }
  }
}

static void CheckShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  auto q_nope_shape = input_infos[kMlaInputQnopeIndex]->GetShape();
  auto q_rope_shape = input_infos[kMlaInputQropeIndex]->GetShape();
  auto ctkv_shape = input_infos[kMlaInputKvCacheIndex]->GetShape();
  auto block_tables_shape = input_infos[kMlaInputBlockTablesIndex]->GetShape();
  auto q_len_shape = input_infos[kMlaInputQueryLensIndex]->GetShape();
  auto context_len_shape = input_infos[kMlaInputContextLensIndex]->GetShape();

  if (!input_infos[kMlaInputQnopeIndex]->IsDynamic()) {
    MS_CHECK_VALUE(q_nope_shape.size() == kMLAQshapeRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of q_nope must be ", kMLAQshapeRank,
                                                       ", but got shape: ", q_nope_shape));
    MS_CHECK_VALUE(q_nope_shape[q_nope_shape.size() - 1] == kMLAQKVnopeHiddenSize,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of q_nope must be ", kMLAQKVnopeHiddenSize,
                                                       ", but got shape: ", q_nope_shape));
  }

  if (!input_infos[kMlaInputQropeIndex]->IsDynamic()) {
    MS_CHECK_VALUE(q_rope_shape.size() == kMLAQshapeRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of q_rope must be ", kMLAQshapeRank,
                                                       ", but got shape: ", q_rope_shape));
    MS_CHECK_VALUE(q_rope_shape[q_rope_shape.size() - 1] == kMLAQKropeHiddenSize,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of q_rope must be ", kMLAQKropeHiddenSize,
                                                       ", but got shape: ", q_rope_shape));
  }

  if (!input_infos[kMlaInputKvCacheIndex]->IsDynamic()) {
    auto q_heads = input_infos[kMlaInputNumHeadIndex]->GetScalarValue<int64_t>();
    if (q_heads.has_value()) {
      if (q_heads.value() == kMLAQheadMax) {
        if (ctkv_shape[kMLABlockSizeDim] != kMLAQheadMax) {
          MS_LOG(EXCEPTION) << "For MLA the block_size must be 128 when head_num is 128, but got block_size: "
                            << ctkv_shape[kMLABlockSizeDim];
        }
      }
    }
  }

  if (!input_infos[kMlaInputBlockTablesIndex]->IsDynamic()) {
    MS_CHECK_VALUE(block_tables_shape.size() == kMLABlockTablesRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of block_tables must be ", kMLABlockTablesRank,
                                                       ", but got shape: ", block_tables_shape));
  }

  if (!input_infos[kMlaInputAttnMaskIndex]->IsNone() && !input_infos[kMlaInputAttnMaskIndex]->IsDynamic()) {
    auto mask_shape = input_infos[kMlaInputAttnMaskIndex]->GetShape();
    auto mask_mode_value = input_infos[kMlaInputMaskModeIndex]->GetScalarValue<int64_t>();
    if (!mask_mode_value.has_value()) {
      MS_EXCEPTION(ValueError) << "For MLA mask_mode must be constant but got variable.";
    }

    auto mask_mode = mask_mode_value.value();
    if (mask_mode == MLAMode::MASK_SPEC || mask_mode == MLAMode::MASK_FREE) {
      MS_CHECK_VALUE(mask_shape.size() == kMLAMaskRank,
                     CheckAndConvertUtils::FormatCommMsg("For MLA The rank of mask must be ", kMLAMaskRank,
                                                         ", but got shape: ", mask_shape));
    }

    if (mask_mode == MLAMode::MASK_FREE) {
      MS_CHECK_VALUE(mask_shape[mask_shape.size() - 1] == kMLAMaskFreeLastDim,
                     CheckAndConvertUtils::FormatCommMsg("For MLA The last dim of mask must be ", kMLAMaskFreeLastDim,
                                                         ", when mask_mode is MASK_FREE but got shape: ", mask_shape));
    }
  }

  if (!input_infos[kMlaInputDeqScaleQkIndex]->IsNone()) {
    auto deq_scale_qk_shape = input_infos[kMlaInputDeqScaleQkIndex]->GetShape();
    MS_CHECK_VALUE(deq_scale_qk_shape.size() == kMLADeqScaleRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of deq_scale_qk must be ", kMLADeqScaleRank,
                                                       ", but got shape: ", deq_scale_qk_shape));
  }

  if (!input_infos[kMlaInputDeqScalePvIndex]->IsNone()) {
    auto deq_scale_pv_shape = input_infos[kMlaInputDeqScalePvIndex]->GetShape();

    MS_CHECK_VALUE(deq_scale_pv_shape.size() == kMLADeqScaleRank,
                   CheckAndConvertUtils::FormatCommMsg("For MLA The rank of deq_scale_pv must be ", kMLADeqScaleRank,
                                                       ", but got shape: ", deq_scale_pv_shape));
  }

  MS_CHECK_VALUE(q_len_shape.size() == kMLADeqScaleRank,
                 CheckAndConvertUtils::FormatCommMsg("For MLA The rank of q_seq_lens must be ", kMLADeqScaleRank,
                                                     ", but got shape: ", q_len_shape));
  MS_CHECK_VALUE(context_len_shape.size() == kMLADeqScaleRank,
                 CheckAndConvertUtils::FormatCommMsg("For MLA The rank of context_lengths must be ", kMLADeqScaleRank,
                                                     ", but got shape: ", context_len_shape));
  if (!input_infos[kMlaInputQueryLensIndex]->IsDynamic() && !input_infos[kMlaInputContextLensIndex]->IsDynamic()) {
    MS_CHECK_VALUE(context_len_shape[0] == q_len_shape[0],
                   CheckAndConvertUtils::FormatCommMsg(
                     "For MLA The shape of context_lengths and q_seq_lens must be same but got context_len_shape: ",
                     context_len_shape, ", q_len_shape: ", q_len_shape));
  }
}

ShapeArray MlaFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &q_nope_info = input_infos[kMlaInputQnopeIndex];
  auto q_nope_shape = q_nope_info->GetShape();
  auto is_ring_value = input_infos[kMlaInputIsRingIndex]->GetScalarValue<int64_t>();
  if (!is_ring_value.has_value()) {
    MS_EXCEPTION(ValueError) << "For MLA, the ring must be a constant, but got a variable.";
  }

  auto is_ring = is_ring_value.value();
  if (is_ring != 0) {
    MS_EXCEPTION(ValueError) << "For MLA, ir_ring must be 0 now, but got: " << is_ring;
  }

  CheckShape(primitive, input_infos);

  CheckParam(primitive, input_infos);

  ShapeVector lse_out_shape{0};
  return {q_nope_shape, lse_out_shape};
}

std::vector<TypeId> MlaFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto q_nope_type = input_infos[kMlaInputQnopeIndex]->GetType();
  auto q_rope_type = input_infos[kMlaInputQropeIndex]->GetType();

  return {q_rope_type, q_nope_type};
}
}  // namespace ops
}  // namespace mindspore
