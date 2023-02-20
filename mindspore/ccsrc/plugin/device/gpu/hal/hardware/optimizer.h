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

#include "backend/common/optimizer/helper.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass_manager.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "backend/common/pass/adjust_depend_for_parallel_optimizer_recompute_all_gather.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "plugin/device/gpu/optimizer/adam_weight_decay_fusion.h"
#include "plugin/device/gpu/optimizer/adam_fusion.h"
#include "plugin/device/gpu/optimizer/alltoall_fusion.h"
#include "plugin/device/gpu/optimizer/apply_momentum_weight_scale_fusion.h"
#include "plugin/device/gpu/optimizer/apply_momentum_scale_fusion.h"
#include "plugin/device/gpu/optimizer/apply_momentum_weight_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_relu_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_relu_grad_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_add_relu_fusion.h"
#include "plugin/device/gpu/optimizer/post_batch_norm_add_relu_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_add_relu_grad_fusion.h"
#include "plugin/device/gpu/optimizer/combine_momentum_fusion.h"
#include "plugin/device/gpu/optimizer/combine_cast_fusion.h"
#include "plugin/device/gpu/optimizer/cudnn_inplace_fusion.h"
#include "plugin/device/gpu/optimizer/insert_format_transform_op.h"
#include "plugin/device/gpu/optimizer/replace_momentum_cast_fusion.h"
#include "plugin/device/gpu/optimizer/replace_addn_fusion.h"
#include "plugin/device/gpu/optimizer/print_reduce_fusion.h"
#include "plugin/device/gpu/optimizer/remove_format_transform_pair.h"
#include "plugin/device/gpu/optimizer/remove_redundant_format_transform.h"
#include "plugin/device/gpu/optimizer/reduce_precision_fusion.h"
#include "plugin/device/gpu/optimizer/relu_v2_pass.h"
#include "plugin/device/gpu/optimizer/add_relu_v2_fusion.h"
#include "plugin/device/gpu/optimizer/add_relu_grad_v2_fusion.h"
#include "common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "plugin/device/gpu/optimizer/concat_outputs_for_all_gather.h"
#include "backend/common/pass/getitem_tuple.h"
#include "plugin/device/gpu/optimizer/matmul_biasadd_fusion.h"
#include "plugin/device/gpu/optimizer/bce_with_logits_loss_fusion.h"
#include "plugin/device/gpu/optimizer/insert_cast_gpu.h"
#include "plugin/device/gpu/optimizer/neighbor_exchange_v2_fusion.h"
#include "plugin/device/gpu/optimizer/bias_dropout_add_fusion.h"
#include "plugin/device/gpu/optimizer/clip_by_norm_fission.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "backend/common/pass/dynamic_sequence_ops_adaptation.h"

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_OPTIMIZER_H_
