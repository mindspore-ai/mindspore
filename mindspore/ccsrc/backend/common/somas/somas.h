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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stack>
#include <set>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"
#include "backend/common/somas/somas_node.h"
#include "backend/common/somas/somas_solver_pre.h"
#include "backend/common/somas/somas_stream.h"
#include "backend/common/somas/somas_parameter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/device_type.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace somas {
struct EventPair {
  CNodePtr send_;
  CNodePtr recv_;
};

union DestinationUnion {
  size_t id;
  size_t index;
  DestinationUnion() : index(0) {}
};

struct TensorConflictInfo {
  size_t tensor_id;
  size_t src_node_id;
  size_t destination_num;
  DestinationUnion l;
  DestinationUnion r;
  TensorConflictInfo(size_t tensor_id, size_t src_node_id)
      : tensor_id(tensor_id), src_node_id(src_node_id), destination_num(0) {}
};

struct Block {
  size_t start_offset_;
  size_t size_;
  size_t end_offset_;

  Block(size_t start, size_t size) : start_offset_(start), size_(size), end_offset_(start + size) {}
};

void MergeBlocks(std::vector<Block> *block_list, std::stack<Block> *merged_blocks);

enum class UnReuseType { kUnReuseAll, kUnReuseInput, kUnReuseOutput, kUnReuseWorkspace };
class BACKEND_EXPORT Somas {
 public:
  // Constructors/Destructors
  Somas() = default;
  Somas(const Somas &) = delete;
  Somas &operator=(const Somas &) = delete;
  virtual ~Somas() = default;

  bool IsSupportSomas(const session::KernelGraph &graph);
  bool Assign(const session::KernelGraph &graph);
  bool Assign(const KernelGraphPtr &graph_ptr);
  std::string SomasInfo(bool calc_hash = false) const;
#ifndef ENABLE_SECURITY
  virtual void ConvertToProfilingNode(uint32_t /* graph_id */) const {}
#endif

 private:
  // device implementation interface
  virtual bool Initialize() = 0;
  virtual string GetDeviceName() const = 0;
  virtual size_t GetAlignSize(size_t original_size) const = 0;
  virtual size_t GetCommunicationReservedSize() const;

  virtual bool GetEnableCacheFlag(const session::KernelGraph &graph) const;
  virtual std::vector<vector<uint32_t>> GetStreamGroupInfo() const;
  virtual bool GetDependExecOrderFlag(const session::KernelGraph &graph) const = 0;
  virtual std::pair<bool, std::string> GetDebugConfig() const;

  virtual std::map<std::string, UnReuseType> GetUnReuseNodeType() const;
  virtual std::map<std::string, UnReuseType> GetUnReuseNodeName() const;

  virtual bool InitDevSpecControlTensors(const session::KernelGraph &graph) = 0;
  virtual bool DevSpecNodeProcess(const session::KernelGraph &graph) = 0;
  virtual void CommunicationTensorProcess(const std::vector<SomasTensorPtr> &tensors) const;
  virtual bool NeedContiguous(const std::vector<size_t> &inputs) const = 0;
  // end

  // SOMAS Configuration
  std::string device_name_{"SOMAS"};
  size_t communication_gap_size_{0};

  bool depend_exec_order_{false};
  bool enable_cache_{false};
  bool save_debug_info_{false};
  std::string debug_info_path_;

  std::map<std::string, UnReuseType> un_reuse_node_type_;
  std::map<std::string, UnReuseType> un_reuse_node_name_;
  // end

  std::vector<DynamicBitSet> reuse_matrix_;
  // hash id
  std::string hash_id_;

  // Stream groups
  std::vector<vector<uint32_t>> streams_groups_;

  // Solver
  TensorsDescMap solver_tensor_desc_map_;
  SomasSolverPrePtr somas_solver_;

  std::vector<vector<size_t>> ref_overlap_constraints_;

  // statistic info
  size_t upper_bound_{0};
  size_t lower_bound_{0};
  size_t workspace_total_size_{0};
  size_t comm_input_total_size_{0};
  size_t comm_output_total_size_{0};
  size_t lifelong_all_total_size_{0};
  size_t lifelong_start_total_size_{0};
  size_t lifelong_end_total_size_{0};

  std::vector<vector<size_t>> processed_contiguous_tensors_list_;
  // key: contiguous list index with first union tensor; value: contiguous list index with other union tensor
  std::map<size_t, size_t> contiguous_list_with_ref_index_map_;

  bool ConfigSomas(const session::KernelGraph &graph);

  // somas model
  void InitSomasModel(const session::KernelGraph &graph);
  bool InitBasicInfoFromGraph(const session::KernelGraph &graph);
  void InitSomasStreamAndNode(const session::KernelGraph &graph);
  void InitSomasOutputAndWorkspaceTensors(const session::KernelGraph &graph);
  void InitSomasInputTensors(const session::KernelGraph &graph);
  void InitCommonNodeInputs(const CNodePtr &kernel);
  void InitAtomicCleanInputs(bool enable_fusion_clear, const CNodePtr &kernel);
  SomasParameterPtr GetSomasParameter(const AnfNodePtr &node, size_t index, size_t param_size,
                                      const std::string &kernel_name);
  SomasParameterPtr CreateSomasParameter(const AnfNodePtr &node, size_t index, size_t param_size,
                                         const std::string &kernel_name);
  void InitControlTensors();
  bool CommonSpecNodeProcess(const session::KernelGraph &graph);
  SomasStreamPtr GetSomasStream(size_t stream_id) const;
#ifndef ENABLE_SECURITY
  void SummaryInputProcess(const session::KernelGraph &graph);
#endif
  void RefNodeProcess(const session::KernelGraph &graph);
  void UnReuseNodeProcess(const session::KernelGraph &graph);
  void CommunicationNodeProcess();
  std::map<size_t, std::map<size_t, std::set<size_t>>> GetContiguousRefListErrorCheckMap();
  void GetContiguousListContainUnionTensor();
  std::map<size_t, size_t> GetRefTensorsInContiguousList();
  common::KernelWithIndex GetVisitKernelWithReturnType(const AnfNodePtr &ori_node, size_t ori_index);

  // conflict matrix
  static bool NodeSort(const SomasNodePtr &node1, const SomasNodePtr &node2);
  void ComputeConflictMatrix();
  void ComputeBasicMatrix();
  static void ComputeOneTensorConflicts(const std::shared_ptr<SomasTensor> &target_tensor,
                                        const std::vector<TensorConflictInfo> &tensor_conflict_info,
                                        const std::vector<size_t> &destination_node_list,
                                        const vector<DynamicBitSet> &nodes_dependency,
                                        std::vector<DynamicBitSet> *tensor_relation);
  void ComputeMultiTensorConflicts(const std::vector<SomasTensorPtr> &target_tensors_list,
                                   const std::vector<TensorConflictInfo> &tensor_conflict_info,
                                   const std::vector<size_t> &destination_node_list,
                                   const vector<DynamicBitSet> &nodes_dependency,
                                   std::vector<DynamicBitSet> *tensor_relation) const;
  void UpdateTensorDestinations();
  void UpdateUnionTensorsConflict();
  static void BuildConflictInfo(const std::shared_ptr<SomasTensor> &tensor, TensorConflictInfo *tensor_conflict_info,
                                std::vector<size_t> *destination_node_list);
  static bool CheckIsDependency(const TensorConflictInfo &tensor_conflict_info, const size_t &src_node_id,
                                const vector<DynamicBitSet> &nodes_dependency,
                                const std::vector<size_t> &destination_node_list);
  void ProcessSemiLifeLongTensor();

  // solver
  void Solve(const session::KernelGraph &graph);
  void UpdateUnionTensorsOffset();
  void UpdateContiguousTensorsOffset(const std::map<size_t, size_t> &contiguous_ref_list_map);

  // cache
  void SaveSomasResult(const session::KernelGraph &graph);
  bool VerifySomasResult(const nlohmann::json &somas_json) const;
  bool LoadSomasResult(const string &filename);
  bool UpdateTensorsOffset(const std::vector<nlohmann::json> &tensors_json);
  bool CalcSomasModelHash(const session::KernelGraph &graph);
  bool LoadSomasCache(const session::KernelGraph &graph);

  // log
  std::string Offline() const;
  void DumpOfflineIR(const string &filename) const;
  size_t CalcLowerBound() const;
  void GenGraphStatisticInfo();
  void DumpParameters(std::ostringstream &oss) const;
  void DumpTensors(std::ostringstream &oss) const;
  void DumpNodes(std::ostringstream &oss) const;
  void DumpSomasModelInfo(const string &tag, uint32_t graph_id) const;

  // update graph
  std::vector<std::pair<size_t, size_t>> GetNodeOutputSomasResult(const AnfNodePtr &node) const;
  std::vector<std::pair<size_t, size_t>> GetNodeWorkSpaceSomasResult(const AnfNodePtr &node) const;
  void UpdateSomasResultToGraph(const session::KernelGraph &graph);

 protected:
  std::vector<SomasParameterPtr> parameters_list_;
  std::vector<SomasTensorPtr> control_tensors_list_;
  std::vector<SomasTensorPtr> tensors_list_;
  std::vector<SomasNodePtr> nodes_list_;

  mindspore::HashMap<size_t, SomasStreamPtr> streams_map_;
  mindspore::HashMap<void *, vector<SomasParameterPtr>> parameters_map_;
  mindspore::HashMap<void *, std::vector<SomasNodePtr>> nodes_map_;

  std::vector<vector<size_t>> union_tensors_list_;
  std::vector<vector<size_t>> contiguous_tensors_list_;

  void AddControlTensor(const SomasNodePtr &from, const SomasNodePtr &to);
  void AddControlTensorFromExecOrder();
  void GraphOutputProcess(const session::KernelGraph &graph);
  void UpdateContiguousTensorList();
  SomasNodePtr GetSomasNode(size_t node_id) const;
  static std::string GetSplitName(const string &scope_name);

  size_t reused_memory_size_{0};
  std::vector<std::pair<size_t, size_t>> dump_merged_blocks_;
};

using SomasPtr = std::shared_ptr<Somas>;
using SomasCreator = std::function<std::shared_ptr<Somas>()>;

// @todo will delete when old runtime remove
class BACKEND_EXPORT SomasManager {
 public:
  static SomasManager &Instance();
  void Register(device::DeviceType device_type, SomasCreator &&creator) {
    if (base_map_.find(device_type) == base_map_.end()) {
      (void)base_map_.emplace(device_type, creator);
    }
  }
  void Clear();

  SomasPtr GetSomas(device::DeviceType device_type) {
    auto iter = base_map_.find(device_type);
    if (base_map_.end() != iter) {
      MS_EXCEPTION_IF_NULL(iter->second);
      return (iter->second)();
    }
    return nullptr;
  }

 private:
  std::map<device::DeviceType, SomasCreator> base_map_;
};

class SomasRegister {
 public:
  SomasRegister(device::DeviceType device_type, SomasCreator &&creator) {
    SomasManager::Instance().Register(device_type, std::move(creator));
  }
  ~SomasRegister() = default;
};

#define REG_SOMAS(S, T, C) static const somas::SomasRegister g_##S##_reg(T, []() { return std::make_shared<C>(); });
}  // namespace somas
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_H_
