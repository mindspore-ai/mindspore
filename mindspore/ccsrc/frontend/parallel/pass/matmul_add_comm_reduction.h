/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MATMUL_ADD_COMM_REDUCTION_H
#define MINDSPORE_MATMUL_ADD_COMM_REDUCTION_H

#include "ir/anf.h"
#include "include/frontend/optimizer/optimizer.h"

namespace mindspore {
namespace parallel {
// MatMul allReduce structure fusion
bool MatmulAddCommReduction(const FuncGraphPtr &graph, const opt::OptimizerPtr &);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_MATMUL_ADD_COMM_REDUCTION_H
