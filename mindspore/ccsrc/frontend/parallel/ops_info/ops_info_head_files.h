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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_OPS_INFO_HEAD_FILES_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_OPS_INFO_HEAD_FILES_H_

#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/ops_info/arithmetic_info.h"
#include "frontend/parallel/ops_info/batch_parallel_info.h"
#include "frontend/parallel/ops_info/bias_add_info.h"
#include "frontend/parallel/ops_info/comparison_function_info.h"
#include "frontend/parallel/ops_info/dropout_do_mask_info.h"
#include "frontend/parallel/ops_info/elementary_function_info.h"
#include "frontend/parallel/ops_info/gather_v2_info.h"
#include "frontend/parallel/ops_info/get_next_info.h"
#include "frontend/parallel/ops_info/l2_normalize_info.h"
#include "frontend/parallel/ops_info/layer_norm_info.h"
#include "frontend/parallel/ops_info/loss_info.h"
#include "frontend/parallel/ops_info/matmul_info.h"
#include "frontend/parallel/ops_info/onehot_info.h"
#include "frontend/parallel/ops_info/prelu_info.h"
#include "frontend/parallel/ops_info/reduce_method_info.h"
#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/ops_info/transpose_info.h"
#include "frontend/parallel/ops_info/unsorted_segment_op_info.h"
#include "frontend/parallel/ops_info/virtual_dataset_info.h"
#include "frontend/parallel/ops_info/gather_v2_p_info.h"
#include "frontend/parallel/ops_info/tile_info.h"
#include "frontend/parallel/ops_info/strided_slice_info.h"
#include "frontend/parallel/ops_info/slice_info.h"
#include "frontend/parallel/ops_info/concat_info.h"
#include "frontend/parallel/ops_info/split_info.h"
#include "frontend/parallel/ops_info/tensordot_info.h"
#include "frontend/parallel/ops_info/range_info.h"
#include "frontend/parallel/ops_info/pack_info.h"
#include "frontend/parallel/ops_info/broadcast_to_info.h"
#include "frontend/parallel/ops_info/unique_info.h"
#include "frontend/parallel/ops_info/uniform_candidate_sampler_info.h"
#include "frontend/parallel/ops_info/reluv2_info.h"
#include "frontend/parallel/ops_info/select_info.h"
#include "frontend/parallel/ops_info/gathernd_info.h"
#include "frontend/parallel/ops_info/topk_info.h"
#include "frontend/parallel/ops_info/scatter_update_info.h"
#include "frontend/parallel/ops_info/virtual_output_info.h"
#include "frontend/parallel/ops_info/conv2d_info.h"
#include "frontend/parallel/ops_info/batchnorm_info.h"
#include "frontend/parallel/ops_info/maxpool_info.h"
#include "frontend/parallel/ops_info/gatherd_info.h"
#include "frontend/parallel/ops_info/matmul_dds_info.h"
#include "frontend/parallel/ops_info/dsd_matmul_info.h"
#include "frontend/parallel/ops_info/uniform_real_info.h"
#include "frontend/parallel/ops_info/resizebilinear_info.h"

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_HEAD_FILES_H_
