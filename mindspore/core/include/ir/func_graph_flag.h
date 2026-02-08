/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_FUNC_GRAPH_FLAG_H_
#define MINDSPORE_CORE_IR_FUNC_GRAPH_FLAG_H_

#include <string>

namespace mindspore {
const char FUNC_GRAPH_FLAG_IGNORE_VALUE[] = "ignore_value";
const char FUNC_GRAPH_FLAG_VMAP_TRANSFORMED[] = "vmap_transformed";
const char FUNC_GRAPH_FLAG_DEFER_INLINE[] = "defer_inline";
const char FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP[] = "primal_of_bprop";
const char FUNC_GRAPH_FLAG_SPARSE_BPROP[] = "sparse_bprop";
const char FUNC_GRAPH_FLAG_NO_INLINE[] = "no_inline";
const char FUNC_GRAPH_FLAG_CELL_REUSE[] = "cell_reuse";
const char FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER[] = "lazy_inline_order";
const char FUNC_GRAPH_FLAG_CELL_LAZY_INLINE_ORDER_CLONE[] = "lazy_inline_order_clone";
const char FUNC_GRAPH_FLAG_AFTER_BLOCK[] = "after_block";
const char FUNC_GRAPH_FLAG_CORE[] = "core";
const char FUNC_GRAPH_FLAG_K_GRAPH[] = "k_graph";
const char FUNC_GRAPH_ATTR_GRAPH_KERNEL[] = "graph_kernel";
const char FUNC_GRAPH_ATTR_KERNEL_PACKET[] = "kernel_packet_node";
const char FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON[] = "unsupport_higher_order_grad_reason";
const char FUNC_GRAPH_FLAG_SPECIALIZE_PARAMETER[] = "spec_param";
const char FUNC_GRAPH_OUTPUT_NO_RECOMPUTE[] = "output_no_recompute";
const char FUNC_GRAPH_RECOMPUTE_K_GRAPH[] = "recompute_k_graph";
const char FUNC_GRAPH_RECOMPUTE_GRAD_GRAPH[] = "recompute_grad_graph";
const char FUNC_GRAPH_NOT_RECOMPUTE_K_GRAPH[] = "not_recompute_k_graph";
const char FUNC_GRAPH_FLAG_FORCE_INLINE[] = "force_inline";
const char FUNC_GRAPH_FLAG_DUMP[] = "dump";
const char FUNC_GRAPH_FLAG_DYNAMIC_SHAPE[] = "dynamic_shape";
const char FUNC_GRAPH_FLAG_NO_RECURSIVE[] = "no_recursive";
const char FUNC_GRAPH_FLAG_ARGS_NO_EXPAND[] = "args_no_expand";
const char FUNC_GRAPH_FLAG_PROXY_GRAPH[] = "proxy_graph";
const char FUNC_GRAPH_FLAG_NO_CHILD_GRAPH[] = "no_child_graph";
const char FUNC_GRAPH_FLAG_AMP_STRATEGY[] = "amp_strategy";
const char FUNC_GRAPH_FLAG_ROLLED_HEADER[] = "rolled_header";
const char FUNC_GRAPH_FLAG_FORWARD_PRE_HOOK[] = "forward_pre_hook";

const char kFuncGraphFlagUndetermined[] = "undeterminate";
const char kFuncGraphFlagBackPropEntry[] = "back_prop_entry";
const char kFuncGraphFlagReAutoMonad[] = "re_auto_monad";
const char kFuncGraphFlagRecursive[] = "recursive";
const char kFuncGraphFlagMetaFuncGraphBprop[] = "meta_fg_bprop";
const char kFuncGraphFlagAddedForwardU[] = "added_forward_u";

// saved tensors hooks flag
const char FUNC_GRAPH_FLAG_PACK_HOOK[] = "_saved_tensors_pack_hook";
const char FUNC_GRAPH_FLAG_UNPACK_HOOK[] = "_saved_tensors_unpack_hook";

const char kFuncGraphFlagStreamId[] = "stream_id";
const char kFuncGraphFlagStreamLimitId[] = "stream_limit_id";
const char kFuncGraphFlagCubeNum[] = "cube_num";
const char kFuncGraphFlagVectorNum[] = "vector_num";
const char kFuncGraphFlagStreamCtxAfter[] = "stream_ctx_after";
const char kFuncGraphFlagStreamLimitCtxAfter[] = "stream_limit_ctx_after";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_FUNC_GRAPH_FLAG_H_
