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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_OTHER_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_OTHER_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_OP(KVCacheMgr)
  .INPUT(past, TensorType({DT_FLOAT16}))
  .INPUT(cur, TensorType({DT_FLOAT16}))
  .INPUT(index, TensorType({DT_INT32}))
  .OUTPUT(past, TensorType({DT_FLOAT16}))
  .OP_END_FACTORY_REG(KVCacheMgr)

REG_OP(DecoderKvCache)
  .INPUT(cache, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
  .INPUT(update, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
  .INPUT(valid_seq_len, TensorType({DT_INT64}))
  .INPUT(batch_index, TensorType({DT_INT64}))
  .INPUT(seq_len_axis, TensorType({DT_INT64}))
  .INPUT(new_max_seq_len, TensorType({DT_INT64}))
  .INPUT(cur_max_seq_len, TensorType({DT_INT64}))
  .OUTPUT(out, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
  .OP_END_FACTORY_REG(DecoderKvCache)

REG_OP(PromptKvCache)
  .INPUT(cache, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
  .INPUT(update, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
  .INPUT(valid_seq_len, TensorType({DT_INT64}))
  .INPUT(batch_index, TensorType({DT_INT64}))
  .INPUT(seq_len_axis, TensorType({DT_INT64}))
  .INPUT(new_max_seq_len, TensorType({DT_INT64}))
  .INPUT(cur_max_seq_len, TensorType({DT_INT64}))
  .OUTPUT(out, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
  .OP_END_FACTORY_REG(PromptKvCache)

REG_CUST_OP(NoRepeatNGram)
  .INPUT(state_seq, TensorType({DT_INT32}))
  .INPUT(log_probs, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .OUTPUT(out, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
  .REQUIRED_ATTR(ngram_size, Int)
  .CUST_OP_END_FACTORY_REG(NoRepeatNGram)

REG_CUST_OP(GenerateEodMask)
    .INPUT(inputs_ids, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(position_ids, TensorType({DT_INT64, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(n_pos, Int)
    .REQUIRED_ATTR(eod_token_id, Int)
    .REQUIRED_ATTR(n_step, ListInt)
    .REQUIRED_ATTR(n_error_mode, String)
    .CUST_OP_END_FACTORY_REG(GenerateEodMask)


}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_OTHER_OPS_H_
