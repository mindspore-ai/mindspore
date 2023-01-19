/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd

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

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include "include/common/thread_pool.h"

#include "backend/common/somas/somas_solver_core.h"
#include "backend/common/somas/somas_solver_pre.h"
#include "include/common/debug/common.h"

namespace mindspore {
namespace somas {
constexpr auto kSolBytesThreshold = 100 * 1024 * 1024;
constexpr auto kSolNumThresholdMultiThread = 8;
Status SomasSolverPre::CheckTensors(const TensorsDescMap *pTensors, uint32_t index1, uint32_t index2) const {
  auto tensors = *pTensors;
  if (tensors[index1] == nullptr) {
    MS_LOG(EXCEPTION) << "NULL tensor received in continuous constraint (tensor index " << index1
                      << "), there may be kGraphInput or kGraphOutput in the input tensors or output tensors of the "
                         "fused communication op.";
  }
  if (tensors[index2] == nullptr) {
    MS_LOG(EXCEPTION) << "NULL tensor received in continuous constraint (tensor index " << index2
                      << "), there may be kGraphInput or kGraphOutput in the input tensors or output tensors of the "
                         "fused communication op.";
  }

  if (tensors[index1]->right_) {
    MS_LOG(WARNING) << "Warning:tensor " << index1
                    << " already has a right tensor (id: " << tensors[index1]->right_->index_;
  }
  if (tensors[index2]->left_) {
    MS_LOG(WARNING) << "Warning:tensor " << index2
                    << " already has a left tensor (id: " << tensors[index2]->left_->index_;
  }
  return SUCCESS;
}
Status SomasSolverPre::AddContiguousInfoInMap(const vector<vector<size_t>> &continuous_v,
                                              TensorsDescMap *pTensors) const {
  auto &tensors = *pTensors;
  // creating S Lists
  for (auto &aux : continuous_v) {
    for (size_t i = 0; i < aux.size() - 1; i++) {
      auto index1 = aux[i];
      auto index2 = aux[i + 1];
      if (CheckTensors(pTensors, SizeToUint(index1), SizeToUint(index2)) == FAILED) {
        return FAILED;
      }
      tensors[index1]->right_ = tensors[index2];
      tensors[index2]->left_ = tensors[index1];
    }
  }
  return SUCCESS;
}
Status SomasSolverPre::AddContiguousInfoInMultiMaps(const vector<vector<size_t>> &continuous_v,
                                                    vector<TensorsDescMap> *vecTensorsMap,
                                                    const TensorsDescMap *pTensors) const {
  // creating S Lists
  for (auto &aux : continuous_v) {
    for (size_t i = 0; i < aux.size() - 1; i++) {
      auto index1 = aux[i];
      auto index2 = aux[i + 1];
      if (CheckTensors(pTensors, SizeToUint(index1), SizeToUint(index2)) == FAILED) {
        return FAILED;
      }
      for (size_t sol = 0; sol < vecTensorsMap->size(); sol++) {
        auto &tensors_sol = (*vecTensorsMap)[sol];
        tensors_sol[index1]->right_ = tensors_sol[index2];
        tensors_sol[index2]->left_ = tensors_sol[index1];
      }
    }
  }
  return SUCCESS;
}
vector<TensorsDescMap> SomasSolverPre::CreateTensorsMaps(const TensorsDescMap &tensors, size_t total_sol) const {
  vector<TensorsDescMap> vecTensorsMap(total_sol);
  vecTensorsMap[0] = tensors;
  for (auto &pairT : tensors) {
    for (size_t sol = 1; sol < total_sol; sol++) {
      SomasSolverTensorDesc newDesc = *(pairT.second.get());
      SomasSolverTensorDescPtr newDescPtr = std::make_shared<SomasSolverTensorDesc>(newDesc);
      (void)vecTensorsMap[sol].emplace(pairT.first, newDescPtr);
    }
  }
  return vecTensorsMap;
}
void FindBest(size_t total_sol, const vector<std::shared_ptr<SomasSolverCore>> &solvers, BestInfo *best_info) {
  MS_EXCEPTION_IF_NULL(best_info);
  for (size_t sol = 0; sol < total_sol; sol++) {
    auto &solver = solvers[sol];
    auto &upperbound = solver->GetUpperbound();
    if (upperbound > best_info->worst) {
      best_info->worst = upperbound;
    }
    if (upperbound >= best_info->best) {
      continue;
    }
    if (best_info->best_algo == kManyObjects && solver->algorithm_ == kSingleObject &&
        best_info->best - upperbound <= kSolBytesThreshold) {
      continue;
    }
    best_info->best = upperbound;
    best_info->best_sol = sol;
    best_info->best_algo = solver->algorithm_;
    best_info->best_timing = LongToSize(solver->timing_);
  }
}
Status SomasSolverPre::Solving(const session::KernelGraph &graph, TensorsDescMap *ptensors,
                               const std::vector<DynamicBitSet> *pConstraints,
                               const vector<vector<size_t>> &continuous_v, bool bVerifySolution, bool ball,
                               SortingType sorting, FittingType fitting, AlgorithmType algorithm) {
  Status ret = SUCCESS;
  try {
    TensorsDescMap &tensors = *ptensors;
    constexpr size_t numSortingTypes = static_cast<size_t>(kNumSortingTypes);
    constexpr size_t numFittingTypes = static_cast<size_t>(kNumFittingTypes);
    constexpr size_t numAlgorithmTypes = static_cast<size_t>(kNumAlgorithmTypes);
    constexpr size_t total_sol = numSortingTypes * numFittingTypes * numAlgorithmTypes;
    const double giga = 1024. * 1024. * 1024.;

    vector<std::shared_ptr<SomasSolverCore>> solvers;
    std::vector<common::Task> tasks;
    vector<TensorsDescMap> vecTensorsMap = CreateTensorsMaps(tensors, total_sol);
    if (AddContiguousInfoInMultiMaps(continuous_v, &vecTensorsMap, ptensors) == FAILED) {
      return FAILED;
    }
    auto start = std::chrono::system_clock::now();
    for (size_t algorithm_strategy = 0, sol = 0; algorithm_strategy < numAlgorithmTypes; algorithm_strategy++) {
      for (size_t sort_strategy = 0; sort_strategy < numSortingTypes; sort_strategy++) {
        for (size_t branching_strategy = 0; branching_strategy < numFittingTypes; branching_strategy++) {
          std::shared_ptr<SomasSolverCore> pSolver =
            std::make_shared<SomasSolverCore>(vecTensorsMap[sol], pConstraints, sol);
          pSolver->SetAlgorithmStrategy(AlgorithmType(algorithm_strategy));
          pSolver->SetSortingStrategy(SortingType(sort_strategy));
          pSolver->SetFittingStrategy(FittingType(branching_strategy));
          pSolver->VerifySolution(bVerifySolution);
          auto task = [pSolver]() {
            return pSolver->MemoryAllocationSolver() == SUCCESS ? common::SUCCESS : common::FAIL;
          };
          tasks.emplace_back(task);
          solvers.emplace_back(pSolver);
          sol++;
        }
      }
    }
    common::ThreadPool::GetInstance().SyncRun(tasks);
    BestInfo best_info;
    FindBest(total_sol, solvers, &best_info);
    auto end = std::chrono::system_clock::now();
    size_t total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto &best_solver = solvers[best_info.best_sol];
    for (auto &tensor : tensors) {
      *(tensor.second.get()) = *(vecTensorsMap[best_info.best_sol][tensor.first]);
    }
    max_offset_ = best_solver->GetUpperbound();
    constexpr float kFloatPresent = 100.0;
    MS_LOG(INFO) << "SOMAS SOLVER RESUME:";
    MS_LOG(INFO) << "Best Solution:[" << 1 + best_info.best_sol << "/" << total_sol << "] ";
    MS_LOG(INFO) << "Best result:" << best_info.best << " Bytes " << (best_info.best) / (giga) << " GB ("
                 << (best_info.best - best_solver->Getlifelongmemory()) / (giga) << " GB + "
                 << best_solver->Getlifelongmemory() / (giga) << " GB from lifelong tensors)";
    MS_LOG(INFO) << "Best timing:" << best_info.best_timing << " ms";
    MS_LOG(INFO) << "Best algorithm: " << algorithmTypeNames[best_solver->algorithm_];
    MS_LOG(INFO) << "Best sorting strategy: " << sortingNames[best_solver->sort_strategy_];
    MS_LOG(INFO) << "Best offset strategy: " << branchingNames[best_solver->branching_strategy_];
    MS_LOG(INFO) << "Time elapsed: " << total_time << " ms";
    MS_LOG(INFO) << "Spread:"
                 << static_cast<double>((best_info.worst - best_info.best) /
                                        static_cast<double>(best_info.best * kFloatPresent))
                 << " %%";
    Log(graph, tensors, pConstraints, continuous_v);
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "SomasSolver::Solving FAILED: " << e.what();
  }
  return ret;
}

void SomasSolverPre::Log(const session::KernelGraph &graph, const TensorsDescMap &tensors,
                         const std::vector<DynamicBitSet> *pConstraints,
                         const vector<vector<size_t>> &continuous_v) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    SolverInputLog(graph, tensors, continuous_v);
    SolverOutputLog(graph, tensors);
    TensorRelationLog(pConstraints, graph);
  }
}

void SomasSolverPre::TensorRelationLog(const std::vector<DynamicBitSet> *pConstraints,
                                       const session::KernelGraph &graph) const {
  MS_LOG(INFO) << "SomasSolver::Log Writing somas_tensor_relation.ir..";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs_path = context_ptr->GetSaveGraphsPath();
  std::string filename =
    GetSaveGraphsPathName("somas_tensor_relation_" + std::to_string(graph.graph_id()) + ".ir", save_graphs_path);
  std::ostringstream oss;
  for (size_t tid1 = 0; tid1 < pConstraints->size(); tid1++) {
    oss << 't' << tid1 << ' ';
    for (size_t tid2 = 0; tid2 < (*pConstraints)[tid1].bit_size_; tid2++) {
      oss << 'H' << std::hex << (*pConstraints)[tid1].bit_[tid2];
    }
    oss << std::endl << std::dec;
  }
  (void)Common::SaveStringToFile(filename, oss.str());
  MS_LOG(INFO) << "SomasSolver somas_tensor_relation Log done";
}

void SomasSolverPre::SolverInputLog(const session::KernelGraph &graph, const TensorsDescMap &tensors,
                                    const vector<vector<size_t>> &continuous_v) const {
  MS_LOG(INFO) << "SomasSolver::Log Writing somas_solver_input..";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs_path = context_ptr->GetSaveGraphsPath();
  std::string filename =
    GetSaveGraphsPathName("somas_solver_input_" + std::to_string(graph.graph_id()) + ".ir", save_graphs_path);
  std::ostringstream oss;
  for (auto &t : tensors) {
    oss << "T " << t.second->index_ << " " << t.second->size_ << " " << t.second->lifelong_ << std::endl;
  }

  for (auto &s : continuous_v) {
    oss << "S";
    for (auto idx : s) {
      oss << " " << idx;
    }
    oss << std::endl;
  }
  (void)Common::SaveStringToFile(filename, oss.str());
  MS_LOG(INFO) << "SomasSolver input Log done";
}

void SomasSolverPre::SolverOutputLog(const session::KernelGraph &graph, const TensorsDescMap &tensors) const {
  MS_LOG(INFO) << "SomasSolver::Log Writing somas_solver_output_..";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs_path = context_ptr->GetSaveGraphsPath();
  std::string out_filename =
    GetSaveGraphsPathName("somas_solver_output_" + std::to_string(graph.graph_id()) + ".ir", save_graphs_path);
  std::ostringstream oss;
  constexpr size_t contiguous_left = 1;
  constexpr size_t contiguous_mid = 2;
  constexpr size_t contiguous_right = 3;
  for (auto &t : tensors) {
    SomasSolverTensorDescPtr tensor = t.second;
    int continuous = 0;
    if (tensor->left_ == nullptr && tensor->right_ != nullptr) {
      continuous = contiguous_left;
    } else if (tensor->left_ != nullptr && tensor->right_ != nullptr) {
      continuous = contiguous_mid;
    } else if (tensor->left_ != nullptr && tensor->right_ == nullptr) {
      continuous = contiguous_right;
    }
    const size_t alignment = 512;
    bool size_aligned = tensor->size_ % alignment == 0;
    bool offset_aligned = tensor->offset_ % alignment == 0;

    oss << std::endl
        << "tensor_id=" << tensor->index_ << "\tsize=" << tensor->size_ << "\toffset=" << tensor->offset_
        << "\tcontinuous=" << continuous << "\tsize_aligned=" << size_aligned << "\toffset_aligned=" << offset_aligned;
  }
  (void)Common::SaveStringToFile(out_filename, oss.str());
  MS_LOG(INFO) << "SomasSolver output Log done";
}
}  // namespace somas
}  // namespace mindspore
