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

#include "backend/optimizer/common/helper.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/pass_manager.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/gpu/adam_weight_decay_fusion.h"
#include "backend/optimizer/gpu/adam_fusion.h"
#include "backend/optimizer/gpu/apply_momentum_weight_scale_fusion.h"
#include "backend/optimizer/gpu/apply_momentum_scale_fusion.h"
#include "backend/optimizer/gpu/apply_momentum_weight_fusion.h"
#include "backend/optimizer/gpu/batch_norm_relu_fusion.h"
#include "backend/optimizer/gpu/batch_norm_relu_grad_fusion.h"
#include "backend/optimizer/gpu/batch_norm_add_relu_fusion.h"
#include "backend/optimizer/gpu/post_batch_norm_add_relu_fusion.h"
#include "backend/optimizer/gpu/batch_norm_add_relu_grad_fusion.h"
#include "backend/optimizer/gpu/combine_momentum_fusion.h"
#include "backend/optimizer/gpu/combine_cast_fusion.h"
#include "backend/optimizer/gpu/cudnn_inplace_fusion.h"
#include "backend/optimizer/gpu/insert_format_transform_op.h"
#include "backend/optimizer/gpu/replace_momentum_cast_fusion.h"
#include "backend/optimizer/gpu/replace_addn_fusion.h"
#include "backend/optimizer/gpu/print_reduce_fusion.h"
#include "backend/optimizer/gpu/remove_format_transform_pair.h"
#include "backend/optimizer/gpu/remove_redundant_format_transform.h"
#include "backend/optimizer/gpu/reduce_precision_fusion.h"
#include "backend/optimizer/gpu/relu_v2_pass.h"
#include "backend/optimizer/gpu/add_relu_v2_fusion.h"
#include "backend/optimizer/gpu/add_relu_grad_v2_fusion.h"
#include "backend/optimizer/graph_kernel/graph_kernel_optimization.h"
#include "backend/optimizer/pass/communication_op_fusion.h"
#include "backend/optimizer/gpu/concat_outputs_for_all_gather.h"
#include "backend/optimizer/pass/getitem_tuple.h"
#include "backend/optimizer/gpu/matmul_biasadd_fusion.h"
#include "backend/optimizer/gpu/bce_with_logits_loss_fusion.h"
#include "backend/optimizer/gpu/insert_cast_gpu.h"

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_OPTIMIZER_H_
