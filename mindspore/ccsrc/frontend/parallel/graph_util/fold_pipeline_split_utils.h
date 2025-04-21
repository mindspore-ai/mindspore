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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_FOLD_PIPELINE_SPLIT_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_FOLD_PIPELINE_SPLIT_UTILS_H_

#include <utility>
#include <vector>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"

namespace mindspore {
namespace parallel {
void FoldPipelineReorder(const FuncGraphPtr &root);
void ReorderForBackwardOtherSeg(const PipelinePair &backward_start_pair, const PipelinePair &backward_end_pair,
                                int64_t micro_max, int64_t stage_num, const FuncGraphPtr &root);
void InsertVirtualFoldPipelineEndNode(const AnfNodePtr &temp_node, const FuncGraphManagerPtr &manager);
bool CompFuncBySegAscending(const AnfNodePtr &node1, const AnfNodePtr &node2);
bool CompFuncBySegDescending(const AnfNodePtr &node1, const AnfNodePtr &node2);
AnfNodePtr GetPreNode(const AnfNodePtr &node);
PipelinePair Deduplicate(const std::vector<AnfNodePtr> &node_vector, const FuncGraphPtr &root, int64_t micro_max,
                         int64_t seg_max, bool is_train);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_FOLD_PIPELINE_SPLIT_UTILS_H_
