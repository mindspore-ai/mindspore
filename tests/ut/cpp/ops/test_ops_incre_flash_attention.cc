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

#include <vector>
#include <memory>

#include "ops/test_ops.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/ops_func_impl/incre_flash_attention.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace ops {
#define I64(x) (static_cast<int64_t>((x)))
struct IncreFlashAttentionParams {
    ShapeVector q_shape;
    ShapeVector kv_shape;
    ShapeVector mask_shape;
    ShapeVector pse_shift_shape;
    bool enable_actual_seq_lengths;
    ValuePtr actual_seq_lengths_list;
    ShapeVector actual_seq_lengths_shape;
    TypePtr data_type;
    TypePtr mask_data_type;
    ValuePtr num_heads_value;
    ValuePtr input_layout_value;
    ValuePtr scale_value_value;
    ValuePtr num_key_value_heads_value;
    ValuePtr block_size_value;
    ValuePtr inner_precise_value;
    bool enable_post_quant;
    ShapeVector out_shape;
    TypePtr out_type;
};

class TestIncreFlashAttention : public TestOps, public testing::WithParamInterface<IncreFlashAttentionParams> {};

TEST_P(TestIncreFlashAttention, dyn_shape)
{
    const auto &param = GetParam();
    auto incre_flash_attention_func_impl = std::make_shared<IncreFlashAttentionFuncImpl>();
    auto prim = std::make_shared<Primitive>("IncreFlashAttention");
    auto none = std::make_shared<abstract::AbstractNone>();


    ShapeVector q_shape = param.q_shape;
    ShapeVector kv_shape = param.kv_shape;
    ShapeVector mask_shape = param.mask_shape;
    ShapeVector pse_shift_shape = param.pse_shift_shape;

    bool enable_actual_seq_lengths = param.enable_actual_seq_lengths;
    ValuePtr actual_seq_lengths_list = param.actual_seq_lengths_list;
    ShapeVector actual_seq_lengths_shape = param.actual_seq_lengths_shape;
    TypePtr data_type = param.data_type;
    TypePtr mask_data_type = param.mask_data_type;
    bool enable_post_quant = param.enable_post_quant;
    auto q = std::make_shared<abstract::AbstractTensor>(data_type, q_shape);
    ASSERT_NE(q, nullptr);
    auto k = std::make_shared<abstract::AbstractTensor>(data_type, kv_shape);
    ASSERT_NE(k, nullptr);
    auto v = std::make_shared<abstract::AbstractTensor>(data_type, kv_shape);
    ASSERT_NE(v, nullptr);
    auto attn_mask = std::make_shared<abstract::AbstractTensor>(mask_data_type, mask_shape);
    ASSERT_NE(attn_mask, nullptr);
    auto pse_shift = std::make_shared<abstract::AbstractTensor>(data_type, pse_shift_shape);
    ASSERT_NE(pse_shift, nullptr);
    abstract::AbstractBasePtr actual_seq_lengths = none;
    if (enable_actual_seq_lengths) {
      TypePtr actual_seq_lengths_dtype = kInt64;
      auto actual_seq_lengths = std::make_shared<abstract::AbstractTensor>(actual_seq_lengths_dtype,
                                                                           actual_seq_lengths_shape);
      actual_seq_lengths->set_value(actual_seq_lengths_list);
    }

    ASSERT_NE(param.num_heads_value, nullptr);
    abstract::AbstractBasePtr num_heads = param.num_heads_value->ToAbstract();
    ASSERT_NE(param.input_layout_value, nullptr);
    abstract::AbstractBasePtr input_layout = param.input_layout_value->ToAbstract();
    ASSERT_NE(param.scale_value_value, nullptr);
    abstract::AbstractBasePtr scale_value = param.scale_value_value->ToAbstract();
    ASSERT_NE(param.num_key_value_heads_value, nullptr);
    abstract::AbstractBasePtr num_key_value_heads = param.num_key_value_heads_value->ToAbstract();
    ASSERT_NE(param.block_size_value, nullptr);
    abstract::AbstractBasePtr block_size = param.block_size_value->ToAbstract();
    ASSERT_NE(param.inner_precise_value, nullptr);
    abstract::AbstractBasePtr inner_precise = param.inner_precise_value->ToAbstract();

    auto expect_out_shape = std::make_shared<abstract::Shape>(param.out_shape);
    auto expect_out_dtype = std::make_shared<TensorType>(param.out_type);
    ASSERT_NE(expect_out_shape, nullptr);
    ASSERT_NE(expect_out_dtype, nullptr);

    // execute
    auto input_none = std::make_shared<abstract::AbstractNone>();
    auto input_scalar = std::make_shared<abstract::AbstractScalar>();
    abstract::AbstractBasePtr quant_scale2 = input_none;
    if (enable_post_quant) {
      ShapeVector quant_scale2_shape{1};
      TypePtr quant_scale2_dtype = kFloat32;
      quant_scale2 = std::make_shared<abstract::AbstractTensor>(quant_scale2_dtype, quant_scale2_shape);
    }

    std::vector<AbstractBasePtr> input_args = {q,
        k,
        v,
        attn_mask,
        actual_seq_lengths,
        pse_shift,
        input_none,
        input_none,
        input_none,
        quant_scale2,
        input_none,
        input_none,
        input_none,
        input_none,
        input_none,
        num_heads,
        input_layout,
        scale_value,
        num_key_value_heads,
        block_size,
        inner_precise
        };
    auto out_shape = incre_flash_attention_func_impl->InferShape(prim, input_args);
    auto out_dtype = incre_flash_attention_func_impl->InferType(prim, input_args);

    MS_LOG(DEBUG) << "============ out_shape: " << out_shape->ToString();
    MS_LOG(DEBUG) << "============ expect_shape: " << expect_out_shape->ToString();
    MS_LOG(DEBUG) << "============ out_dtype: " << out_dtype->ToString();
    MS_LOG(DEBUG) << "============ expect_dtype: " << expect_out_dtype->ToString();

    // verify output
    ASSERT_NE(out_shape, nullptr);
    ASSERT_TRUE(*out_shape == *expect_out_shape);
    ASSERT_NE(out_dtype, nullptr);
    ASSERT_TRUE(out_dtype == expect_out_dtype->element());
}

const int BSH = 0;
const int BNSD = 1;

ShapeVector GetQShape(int B, int N, int S, int D, int kvN, int input_layout){
  if (input_layout == BSH) {
    return ShapeVector{B, 1, N * D};
  }
  return ShapeVector{B, N, 1, D};
}

ShapeVector GetKVShape(int B, int N, int S, int D, int kvN, int input_layout){
  if (input_layout == BNSD) {
    return ShapeVector{B, S, kvN * D};
  }
  return ShapeVector{B, kvN, S, D};
}

ShapeVector GetMaskShape(int B, int N, int S, int D, int kvN) {
  return ShapeVector{B, 1, 1, S};
}

ShapeVector GetPseShiftShape(int B, int N, int S, int D, int kvN) {
  return ShapeVector{1, N, 1, S};
}

INSTANTIATE_TEST_CASE_P(TestIncreFlashAttentionGroup, TestIncreFlashAttention,
    testing::Values(
        IncreFlashAttentionParams{
          // B, N, S, D, kvN
          GetQShape(1, 5, 4096, 128, 1, 0),
          GetKVShape(1, 5, 4096, 128, 1, 0),
          GetMaskShape(1, 5, 4096, 128, 1),
          GetPseShiftShape(1, 5, 4096, 128, 1),
          true,
          CreateList({4096}),
          {1},
          kFloat16,
          kFloat16,
          CreateScalar<int64_t>(1),
          CreateScalar<int64_t>(0),
          CreateScalar<float>(1.0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(1),
          false,
          {1, 1, 640},
          kFloat16
        },
        IncreFlashAttentionParams{
          // B, N, S, D, kvN
          GetQShape(1, 5, 4096, 128, 1, 0),
          GetKVShape(1, 5, 4096, 128, 1, 0),
          GetMaskShape(1, 5, 4096, 128, 1),
          GetPseShiftShape(1, 5, 4096, 128, 1),
          true,
          CreateList({4096}),
          {1},
          kFloat16,
          kFloat16,
          CreateScalar<int64_t>(1),
          CreateScalar<int64_t>(0),
          CreateScalar<float>(1.0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(1),
          true,
          {1, 1, 640},
          kInt8
        },
        IncreFlashAttentionParams{
          // B, N, S, D, kvN
          GetQShape(1, 5, 4096, 128, 1, 1),
          GetKVShape(1, 5, 4096, 128, 1, 1),
          GetMaskShape(1, 5, 4096, 128, 1),
          GetPseShiftShape(1, 5, 4096, 128, 1),
          true,
          CreateList({4096}),
          {1},
          kBFloat16,
          kBFloat16,
          CreateScalar<int64_t>(1),
          CreateScalar<int64_t>(1),
          CreateScalar<float>(1.0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(1),
          false,
          {1, 5, 1, 128},
          kBFloat16
        },
        IncreFlashAttentionParams{
          // B, N, S, D, kvN
          {-1, -1, -1},
          {-1, -1, -1},
          {-1, -1, -1, -1},
          {-1, -1, -1, -1},
          true,
          CreateList({-1}),
          {1},
          kFloat16,
          kFloat16,
          CreateScalar<int64_t>(1),
          CreateScalar<int64_t>(0),
          CreateScalar<float>(1.0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(1),
          false,
          {-1, 1, -1},
          kFloat16
        },
        IncreFlashAttentionParams{
          // B, N, S, D, kvN
          {-1, -1, -1, -1},
          {-1, -1, -1, -1},
          {-1, -1, -1, -1},
          {-1, -1, -1, -1},
          true,
          CreateList({-1}),
          {1},
          kBFloat16,
          kBFloat16,
          CreateScalar<int64_t>(1),
          CreateScalar<int64_t>(1),
          CreateScalar<float>(1.0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(0),
          CreateScalar<int64_t>(1),
          false,
          {-1, -1, 1, -1},
          kBFloat16
        }
        ));

}  // namespace ops
}  // namespace mindspore