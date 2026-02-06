/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_OPTIMIZER_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_OPTIMIZER_H_

#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/common/pass_manager/pass_manager.h"
#include "backend/common/pass/getitem_tuple.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "backend/common/pass/dynamic_sequence_ops_adaptation.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "backend/common/pass/adjust_depend_for_parallel_optimizer_recompute_all_gather.h"
#include "include/backend/common/pass_manager/common_backend_optimization.h"
#include "plugin/gpu/graph_optimizer/pass/base/replace_addn_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/adam_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/adam_weight_decay_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/alltoall_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/apply_momentum_scale_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/apply_momentum_weight_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/apply_momentum_weight_scale_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/batch_norm_add_relu_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/batch_norm_add_relu_grad_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/batch_norm_relu_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/batch_norm_relu_grad_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/batch_norm_silu_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/batch_norm_silu_grad_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/bias_dropout_add_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/clip_by_norm_fission.h"
#include "plugin/gpu/graph_optimizer/pass/train/concat_outputs_for_all_gather.h"
#include "plugin/gpu/graph_optimizer/pass/train/insert_format_transform_op.h"
#include "plugin/gpu/graph_optimizer/pass/train/matmul_biasadd_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/neighbor_exchange_v2_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/post_batch_norm_add_relu_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/remove_format_transform_pair.h"
#include "plugin/gpu/graph_optimizer/pass/train/replace_momentum_cast_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/train/remove_redundant_format_transform.h"
#include "plugin/gpu/graph_optimizer/pass/inference/combine_optimizer_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/inference/combine_cast_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/inference/cudnn_inplace_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/inference/insert_cast_gpu.h"
#include "plugin/gpu/graph_optimizer/pass/inference/print_reduce_fusion.h"
#include "plugin/gpu/graph_optimizer/pass/inference/reduce_precision_fusion.h"

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_OPTIMIZER_H_
