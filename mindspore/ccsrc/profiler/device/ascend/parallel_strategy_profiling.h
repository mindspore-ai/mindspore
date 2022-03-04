/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_PARALLEL_STRATEGY_PROFILING_H
#define MINDSPORE_PARALLEL_STRATEGY_PROFILING_H

#include <string>

#include "ir/func_graph.h"
#include "base/base.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace profiler {
namespace ascend {
BACKEND_EXPORT void DumpProfileParallelStrategy(const FuncGraphPtr &func_graph);
void SaveParallelStrategyToFile();
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_PARALLEL_STRATEGY_PROFILING_H
