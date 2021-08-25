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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_USS_ATOMIC_ADD_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_USS_ATOMIC_ADD_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/graph_kernel/add_atomic_clean.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
/*
 * output = SubGraph(input_x, segment_ids) {
 *   %0 = UnsortedSegmentSum(%para1, %para2)
 *   return %0
 * }
 * ---------------------------------------------------------------->
 * // Clean output with zero.
 * output = broadcast_to(0.0) // attrs{"shape": [shape of origin output.]}
 * fake_out = SubGraph'(input_x, segment_ids, output) {
 *   %0 = UnsortedSegmentSum(%para1, %para2)
 *   %1 = InplaceAssign(%para3, %0, %0) // attrs{"fake_output":true}
 *   return %1
 * }
 */
class UssAtomicAdd : public AtomicCleanInsertter {
 public:
  UssAtomicAdd() : AtomicCleanInsertter("unsorted_segment_sum_atomic_add_process") {}
  ~UssAtomicAdd() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
using UssAtomicAddPtr = std::shared_ptr<UssAtomicAdd>;
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_USS_ATOMIC_ADD_H_
