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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_INFERENCE_WEIGHT_PREPROCESS_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_INFERENCE_WEIGHT_PREPROCESS_UTILS_H_

#include <vector>
#include <functional>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/api/data_type.h"
#include "mindspore/core/utils/ms_context.h"
#include "mindspore/core/ops/framework_ops.h"

namespace mindspore {
namespace opt {

constexpr auto n_lens_str = "n_lens";
constexpr auto is_fixed_weight_str = "is_fixed_weight";

tensor::TensorPtr GetParamFromLoad(const CNodePtr &load, const bool unused);

bool CheckFusionValid(const CNodePtr &matmul, int64_t *k, const int trans_a_pos, const int trans_b_pos,
                      const std::vector<TypeId> &valid_dtypes);

std::shared_ptr<ValueNode> CreateWeightTensor(TypeId type_id, const std::vector<int64_t> &weight_shape,
                                              const std::vector<void *> &data_c_list,
                                              const std::vector<int64_t> &n_len_list, const int64_t &k_len,
                                              const std::shared_ptr<Type> &w_dtype, const bool &need_rank_offset,
                                              const uint32_t &global_rank_id);

void SortWeightNodeList(AnfNodePtrList *node_list);

std::shared_ptr<ValueNode> ConvertWeightsToNewType(const AnfNodePtr &weight_node);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_INFERENCE_WEIGHT_PREPROCESS_UTILS_H_
