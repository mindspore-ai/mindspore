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

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "backend/optimizer/somas/somas_solver_core.h"
#include "backend/optimizer/somas/somas_solver_pre.h"
#include "debug/common.h"

namespace mindspore {
namespace somas {
Status SomasSolverPre::Solving(const session::KernelGraph *graph,
                               std::unordered_map<size_t, SomasSolverTensorDescPtr> *ptensors,
                               const std::vector<DynamicBitSet> *pConstraints,
                               const vector<vector<size_t>> &continuous_v, bool bVerifySolution, bool ball,
                               SortingType sorting, FittingType fitting, AlgorithmType algorithm) {
  Status retval = SUCCESS;

  try {
    std::unordered_map<size_t, SomasSolverTensorDescPtr> &tensors = *ptensors;
    MS_LOG(INFO) << "Filling in constraints matrix..";
    uint32_t continuous_cnt = 0;
    // creating S Lists
    for (auto &aux : continuous_v) {
      for (uint32_t i = 0; i < aux.size() - 1; i++) {
        uint32_t index1 = aux[i];
        uint32_t index2 = aux[i + 1];
        if (NULL == tensors[index1]) {
          MS_LOG(WARNING) << "NULL tensor received in continuous constraint (tensor index " << index1 << ")";
          return FAILED;
        }
        if (NULL == tensors[index2]) {
          MS_LOG(WARNING) << "NULL tensor received in continuous constraint (tensor index " << index2 << ")";
          return FAILED;
        }

        if (tensors[index1]->right_)
          MS_LOG(WARNING) << "Warning:tensor " << index1
                          << " already has a right tensor (id: " << tensors[index1]->right_->index_;
        if (tensors[index2]->left_)
          MS_LOG(WARNING) << "Warning:tensor " << index2
                          << " already has a left tensor (id: " << tensors[index2]->left_->index_;

        tensors[index1]->right_ = tensors[index2];
        tensors[index2]->left_ = tensors[index1];
        continuous_cnt++;
      }
    }
    continuous_cnt++;

    std::shared_ptr<SomasSolverCore> pSolver = std::make_shared<SomasSolverCore>(tensors, pConstraints);
    pSolver->SetAlgorithmStrategy(algorithm);
    pSolver->SetSortingStrategy(sorting);
    pSolver->SetFittingStrategy(fitting);
    pSolver->SetAllStrategies(ball);
    pSolver->VerifySolution(bVerifySolution);

    if (SUCCESS == (pSolver->MemoryAllocationSolver())) {
      max_offset_ = pSolver->GetUpperbound();
      const double giga = 1024. * 1024. * 1024.;
      MS_LOG(INFO) << "SomasSolver::Solving SUCCESS";
      MS_LOG(INFO) << "SomasSolver::Solving RESULT: " << max_offset_ << " (" << max_offset_ / (giga) << " GB)";
    }
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
    if (save_graphs) {
      Log(graph, tensors, pConstraints, continuous_v);
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "SomasSolver::Solving FAILED: " << e.what();
    retval = FAILED;
  }
  return retval;
}

void SomasSolverPre::Log(const session::KernelGraph *graph,
                         const unordered_map<size_t, SomasSolverTensorDescPtr> &tensors,
                         const std::vector<DynamicBitSet> *pConstraints, const vector<vector<size_t>> &continuous_v) {
  SolverInputLog(graph, tensors, pConstraints, continuous_v);
  SolverOutputLog(graph, tensors);
}

void SomasSolverPre::SolverInputLog(const session::KernelGraph *graph,
                                    const unordered_map<size_t, SomasSolverTensorDescPtr> &tensors,
                                    const std::vector<DynamicBitSet> *pConstraints,
                                    const vector<vector<size_t>> &continuous_v) {
  MS_LOG(INFO) << "SomasSolver::Log Writing somas-input.txt..";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs_path = context_ptr->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  std::string filename = save_graphs_path + "/" + "somas_solver_input_" + std::to_string(graph->graph_id()) + ".ir";
  if (filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << filename << " is too long.";
    return;
  }
  auto real_path = Common::GetRealPath(filename);
  if (!real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << filename;
    return;
  }

  ChangeFileMode(real_path.value(), S_IRWXU);
  std::ofstream ofs(real_path.value());

  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open log file '" << real_path.value() << "' failed!";
    return;
  }

  for (auto &t : tensors) {
    ofs << "T " << t.second->index_ << " " << t.second->size_ << " " << t.second->lifelong_ << std::endl;
  }

  for (auto &t1 : tensors) {
    for (auto &t2 : tensors) {
      size_t idx1 = t1.first;
      size_t idx2 = t2.first;
      if ((idx1 != idx2) && (*pConstraints)[idx1].IsBitTrue(idx2) == false) {
        ofs << "C " << idx1 << " " << idx2 << std::endl;
      }
    }
  }
  for (auto &s : continuous_v) {
    ofs << "S";
    for (auto idx : s) {
      ofs << " " << idx;
    }
    ofs << std::endl;
  }
  ofs.close();

  MS_LOG(INFO) << "SomasSolver input Log done";
}

void SomasSolverPre::SolverOutputLog(const session::KernelGraph *graph,
                                     const unordered_map<size_t, SomasSolverTensorDescPtr> &tensors) const {
  MS_LOG(INFO) << "SomasSolver::Log Writing somas output...";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs_path = context_ptr->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  std::string out_filename =
    save_graphs_path + "/" + "somas_solver_output_" + std::to_string(graph->graph_id()) + ".ir";
  if (out_filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << out_filename << " is too long.";
    return;
  }
  auto out_real_path = Common::GetRealPath(out_filename);
  if (!out_real_path.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << out_filename;
    return;
  }

  ChangeFileMode(out_real_path.value(), S_IRWXU);
  std::ofstream ofs_out(out_real_path.value());

  if (!ofs_out.is_open()) {
    MS_LOG(ERROR) << "Open log file '" << out_real_path.value() << "' failed!";
    return;
  }

  for (auto &t : tensors) {
    SomasSolverTensorDescPtr tensor = t.second;
    int continuous = 0;
    if (tensor->left_ == NULL && tensor->right_ != NULL)
      continuous = 1;
    else if (tensor->left_ != NULL && tensor->right_ != NULL)
      continuous = 2;
    else if (tensor->left_ != NULL && tensor->right_ == NULL)
      continuous = 3;
    const size_t alignment = 512;
    bool size_aligned = tensor->size_ % alignment == 0;
    bool offset_aligned = tensor->offset_ % alignment == 0;

    ofs_out << std::endl
            << "tensor_id=" << tensor->index_ << "\tsize=" << tensor->size_ << "\toffset=" << tensor->offset_
            << "\tcontinuous=" << continuous << "\tsize_aligned=" << size_aligned
            << "\toffset_aligned=" << offset_aligned;
  }
  ofs_out.close();

  MS_LOG(INFO) << "SomasSolver output Log done";
}
}  // namespace somas
}  // namespace mindspore
