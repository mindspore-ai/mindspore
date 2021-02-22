/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_SOLVER_CORE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_SOLVER_CORE_H_

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/optimizer/somas/somas_solver_alg.h"
#include "backend/optimizer/somas/somas_solver_pre.h"

namespace mindspore {
namespace somas {

class SomasSolverCore {
 public:
  /// Interface Function: receive parameters, creates the model to solve and then save the result
  SomasSolverCore(const TensorsDescMap &tensors, const std::vector<DynamicBitSet> *constraints, uint32_t sol,
                  bool isMultiThreadValid = true)
      : best_sol_(0),
        sort_strategy_(kGreaterSizeSmallerIndex),
        branching_strategy_(kBest),
        sol_count_(sol),
        algorithm_(kManyObjects),
        tensors_(tensors),
        constraints_(*constraints),
        upperbound_(SIZE_MAX),
        verify_(false),
        all_(true),
        is_multi_thread_valid_(isMultiThreadValid) {}
  ~SomasSolverCore() = default;

  Status MemoryAllocationSolver();
  Status Verify();
  bool Verify(const size_t &);
  void VerifySolution(const bool verify) { verify_ = verify; }
  void SortTensors();
  void BuildBlocks();
  void Clean();
  void SetBestSolution() { RestoreSolution(best_sol_); }
  void RestoreSolution(uint32_t sol_id);
  void SetSortingStrategy(SortingType sort_strategy) { sort_strategy_ = sort_strategy; }
  void SetFittingStrategy(FittingType branching_strategy) { branching_strategy_ = branching_strategy; }
  void SetAlgorithmStrategy(AlgorithmType algorithm_strategy) { algorithm_ = algorithm_strategy; }
  void SetAllStrategies(bool all) { all_ = all; }
  const size_t &GetUpperbound() const { return upperbound_; }
  const size_t &Getlifelongmemory() const { return lifelong_memory_; }

  uint32_t best_sol_{0};
  SortingType sort_strategy_;
  FittingType branching_strategy_;
  uint32_t sol_count_{0};
  AlgorithmType algorithm_;
  size_t timing_{0};

 private:
  const TensorsDescMap &tensors_;
  vector<BlockTensor> block_tensors_;
  const std::vector<DynamicBitSet> &constraints_;
  size_t upperbound_{0};
  size_t lifelong_memory_{0};
  bool verify_{false};
  bool all_{false};
  bool is_multi_thread_valid_{true};

  size_t FindSolutions();
  size_t Search(const std::shared_ptr<FootPrint> &pFootprint);
  void AppendLifelongTensors();
  void Destroy(std::shared_ptr<FootPrint> &);
};
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_SOLVER_CORE_H_
