/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd

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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "backend/optimizer/somas/somas_node.h"
#include "backend/optimizer/somas/somas_solver_pre.h"
#include "backend/optimizer/somas/somas_stream.h"
#include "backend/optimizer/somas/somas_parameter.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace somas {
class Somas {
 public:
  // Constructors/Destructors
  Somas() = default;
  Somas(const Somas &) = delete;
  Somas &operator=(const Somas &) = delete;
  ~Somas() = default;

  bool Allocate(const session::KernelGraph *graph);
  size_t GetTotalMemSize() { return mem_offset_; }
  void set_mem_base_addr(uint8_t *mem_base_addr) { mem_base_addr_ = mem_base_addr; }
  uint8_t *GetNodeOutputPtr(const AnfNodePtr &node, size_t index) const;
  uint8_t *GetNodeWorkSpacePtr(const AnfNodePtr &node, size_t index) const;

  std::string SomasInfo(bool calc_hash = false);
  std::string SomasMemory();
  void DumpSomasInfoIR(const string filename);
  void DumpSomasMemoryIR(const string filename);

  static bool NodeSort(SomasNodePtr, SomasNodePtr);
  std::vector<DynamicBitSet> reuse_matrix_;
  void ConvertToProfilingNode(uint32_t graph_id);

 private:
  // hash id
  std::string hash_id_;
  // Maps
  std::unordered_map<size_t, SomasTensorPtr> tensors_map_;
  std::map<void *, SomasNodePtr> nodes_map_;
  std::map<void *, vector<SomasParameterPtr>> parameters_map_;

  // Vectors
  std::vector<SomasNodePtr> nodes_list_;
  std::vector<SomasStreamPtr> streams_list_;
  std::vector<SomasTensorPtr> tensors_list_;
  std::vector<SomasParameterPtr> parameters_list_;

  // Stream groups
  std::vector<vector<uint32_t>> streams_groups_;

  // Solver
  TensorsDescMap solver_tensor_desc_map_;
  SomasSolverPrePtr somas_solver_;

  // Contiguous list
  std::vector<vector<size_t>> contiguous_tensors_list_;

  // Ref lists
  std::vector<vector<size_t>> ref_node_constraints_;
  std::vector<vector<size_t>> ref_overlap_constraints_;

  // total Offset
  size_t mem_offset_{0};

  // Memory base addr
  uint8_t *mem_base_addr_{nullptr};

  // Save debug info
  bool save_graphs_{false};
  std::string save_graphs_path_;

  // statistic info
  size_t upper_bound_{0};
  size_t lower_bound_{0};
  size_t workspace_total_size_{0};
  size_t comm_input_total_size_{0};
  size_t comm_output_total_size_{0};
  size_t lifelong_all_total_size_{0};
  size_t lifelong_start_total_size_{0};
  size_t lifelong_end_total_size_{0};

  bool InitSomasTensors(const session::KernelGraph *graph);
  void InitBasicInfo(const session::KernelGraph *graph);
  void InitSomasStreamAndNode(const session::KernelGraph *graph);
  void InitSomasOutputAndWorkspaceTensors(const session::KernelGraph *graph);
  void InitSomasInputTensors(const session::KernelGraph *graph);
  void GetNextOutputProcess(const session::KernelGraph *graph);
  void IndependentNodeOutputProcess(const session::KernelGraph *graph);
  void SummaryInputProcess(const session::KernelGraph *graph);
  void RefNodeProcess(const session::KernelGraph *graph);
  void NonTaskSplitProcess(const session::KernelGraph *graph);
  void UnReuseNodeProcess(const session::KernelGraph *graph);
  SomasTensorPtr CreateGapTensor(size_t gap_tensor_id);
  void GenContiguousList(const session::KernelGraph *graph);

  void ComputeConflictPairs();

  bool Assign(const session::KernelGraph *graph);

  std::string Offline();
  void DumpOfflineIR(const string filename);
  std::string GetSplitName(const string &scope_name) const;
  size_t CalcLowerBound() const;
  void GenGraphStatisticInfo();
  SomasParameterPtr GetSomasParameters(AnfNodePtr node, size_t index);
  SomasParameterPtr CreateSomasParameters(AnfNodePtr node, size_t index);
  void InitCommonNodeInputs(bool is_all_nop_node, const CNodePtr &kernel);
  void InitAtomicCleanInputs(bool is_all_nop_node, const CNodePtr &kernel);
  void ComputeOneTensorConflicts(const std::shared_ptr<SomasTensor> &calc_tensor,
                                 const std::vector<SomasTensorPtr> &all_tensors_list,
                                 const vector<DynamicBitSet> &nodes_dependency,
                                 std::vector<DynamicBitSet> *tensor_relation) const;
  void ComputeMultiTensorConflicts(const std::vector<SomasTensorPtr> &calc_tensors_list,
                                   const std::vector<SomasTensorPtr> &all_tensors_list,
                                   const vector<DynamicBitSet> &nodes_dependency,
                                   std::vector<DynamicBitSet> *tensor_relation) const;
  void UpdateTensorDestinations();
  void UpdateRefTensorsConflict();
  void UpdateRefOverlapTensorsConflicts();
  void UpdateRefTensorsOffset();
  void UpdateContiguousTensorsOffset(const std::map<size_t, size_t> &contiguous_ref_list_map);
  void DumpParameters(std::ostringstream &oss) const;
  void DumpTensors(std::ostringstream &oss) const;
  void DumpNodes(std::ostringstream &oss) const;
  std::map<size_t, size_t> GetContiguousListContainRefTensor();
  std::map<size_t, size_t> GetRefTensorsInContiguousList();
  bool SaveSomasResult(const session::KernelGraph *graph);
  bool VerifySomasResult(const session::KernelGraph *graph, const nlohmann::json &somas_json) const;
  bool LoadSomasResult(const session::KernelGraph *graph, const string filename);
  bool UpdateTensorsOffset(const std::vector<nlohmann::json> &tensors_json);
  bool CalcSomasModelHash(const session::KernelGraph *graph);
};

using SomasPtr = std::shared_ptr<Somas>;
}  // namespace somas
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_H_
