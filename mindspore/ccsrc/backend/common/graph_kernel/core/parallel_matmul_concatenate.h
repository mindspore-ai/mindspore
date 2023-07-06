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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_MATMUL_CONCATENATE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_MATMUL_CONCATENATE_H_

#include <vector>
#include <string>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/parallel_op_concatenate.h"

namespace mindspore::graphkernel {
using BMNK = std::tuple<int64_t, int64_t, int64_t, int64_t>;
using MMAttr = std::pair<bool, bool>;
class ParallelMatMulConcatenater : public ParallelOpConcatenater {
 public:
  explicit ParallelMatMulConcatenater(uint64_t min_num_branches, const std::string &layout)
      : ParallelOpConcatenater("MatMul", min_num_branches, layout) {}

 protected:
  virtual bool CanOpsBeCombined(const AnfNodePtr a, const AnfNodePtr b);
  virtual bool IsSupportedOp(const AnfNodePtr n);
  virtual AnfNodePtr MakeCombinedOp(const Group &branches);
  bool IsArgCompatible(const AnfNodePtr a, const AnfNodePtr b) override;

 private:
  ConcatenatePlan Analyse(const Group &branches) const;
};

AnfNodePtr ConcatParallelMatMul(AnfNodePtr root, uint64_t min_num_branches, const std::string &layout,
                                const FuncGraphPtr &func_graph = nullptr);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_MATMUL_CONCATENATE_H_
