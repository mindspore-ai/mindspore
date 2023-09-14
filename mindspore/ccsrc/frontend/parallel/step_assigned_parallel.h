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
#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_ASSIGNED_PARALLEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_ASSIGNED_PARALLEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "frontend/optimizer/opt.h"
#include "frontend/parallel/status.h"
#include "ir/anf.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"

namespace mindspore {
namespace parallel {
// main step of Auto-parallel
bool StepAssignedParallel(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager, size_t device_num,
                          size_t rank_id, bool sapp);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_ASSIGNED_PARALLEL_H_
