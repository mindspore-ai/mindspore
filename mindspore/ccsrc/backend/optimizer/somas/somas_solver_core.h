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
  SomasSolverCore(const std::unordered_map<size_t, SomasSolverTensorDescPtr> &tensors,
                  const std::vector<DynamicBitSet> *constraints)
      : tensors_(tensors),
        constraints_(*constraints),
        upperbound_(SIZE_MAX),
        timing_(0),
        lifelongmemory_(0),
        verify_(false),
        all_(true),
        best_sol_(0),
        best_sort_(kGreaterSizeSmallerIndex),
        best_branching_(kBest),
        sort_strategy_(kGreaterSizeSmallerIndex),
        branching_strategy_(kBest),
        sol_count_(0),
        algorithm_(kManyObjects) {}
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

 private:
  std::unordered_map<size_t, SomasSolverTensorDescPtr> tensors_;
  vector<BlockTensor> block_tensors_;
  std::vector<DynamicBitSet> constraints_;
  size_t upperbound_{0};
  size_t timing_{0};
  size_t lifelongmemory_{0};
  bool verify_{false};
  bool all_{false};
  uint32_t best_sol_{0};
  SortingType best_sort_;
  FittingType best_branching_;
  SortingType sort_strategy_;
  FittingType branching_strategy_;
  uint32_t sol_count_{0};
  AlgorithmType algorithm_;

  size_t FindSolutions();
  size_t Search(const std::shared_ptr<FootPrint> &pFootprint);
  void AppendLifelongTensors();
  void Destroy(std::shared_ptr<FootPrint> &);

  const std::string sorting_[6] = {"size(>), index(<)",
                                   "size(>), index(>)",
                                   "size(>), constraints(<), index(<)",
                                   "size(>), constraints(<), index(>)",
                                   "size(>), constraints(>), index(<)",
                                   "size(>), constraints(>), index(>)"};
  const std::string branching_[4] = {"bestfit", "smallest", "largest", "worstfit"};
  const std::string algorithm_type_[2] = {"Shared Objects", "Single Object"};
};
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_SOLVER_CORE_H_
