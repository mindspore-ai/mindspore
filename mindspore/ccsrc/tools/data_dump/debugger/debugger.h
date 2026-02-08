/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUGGER_DEBUGGER_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUGGER_DEBUGGER_H_

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "device_address/device_address.h"
#include "google/protobuf/repeated_field.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "tools/data_dump/debug_services.h"
#include "tools/tensor_data.h"
#include "tools/visible.h"

namespace debugger {
class Chunk;
class GraphProto;
class ModelProto;
class Statistics;
class TensorProto;
class TensorBase;
class TensorSummary;
enum DataType : int;
}  // namespace debugger

template <class T>
using ProtoVector = google::protobuf::RepeatedPtrField<T>;

namespace mindspore {
using mindspore::device::DeviceContext;

class DebugServices;
class TOOLS_EXPORT Debugger : public std::enable_shared_from_this<Debugger> {
 public:
  static std::shared_ptr<Debugger> GetInstance();

  // deconstructor
  ~Debugger() = default;

  // init
  // only save device_id
  void Init(const uint32_t device_id, const std::string device_target);

  // reset debugger
  void Reset();

  void PreExecuteGraphDebugger(const std::vector<KernelGraphPtr> &graphs,
                               const std::vector<AnfNodePtr> &origin_parameters_order);
  // enable debugger
  // send graph and wait for command
  // do nothing if graph is set already
  void PreExecute(const KernelGraphPtr &graph_ptr);

  void SetCurrentAndPrevRootGraph(uint32_t root_graph_id);

  void SetAscendKernelByKernelFlag(bool value) { ascend_kernel_by_kernel_ = value; }

  bool GetAscendKernelByKernelFlag() const { return ascend_kernel_by_kernel_; }

  void StoreRunGraphIdList(uint32_t graph_id);

  // analyze tensors and wait for command
  // don't need a graph_ptr because it is saved during pre_execute
  void PostExecute();

  void DumpSingleNode(const CNodePtr &node, uint32_t graph_id, const DeviceContext *device_context = nullptr) const;

  void DumpInGraphCompiler(const KernelGraphPtr &kernel_graph);

  void PostExecuteGraphDebugger();

  bool DumpTensorToFile(const std::string &filepath, const std::string &tensor_name, size_t slot) const;

  bool LoadNewTensor(const std::shared_ptr<TensorData> &tensor, bool keep_prev);

  std::shared_ptr<TensorData> GetTensor(const std::string &tensor_name) const;

  // check if any feature that uses the debugger backend is enabled
  bool DebuggerBackendEnabled() const;

  void LoadParametersAllGraphs();

  void LoadConstsForGraph(const KernelGraphPtr &graph);

  void DumpParamsAndConstAndHistory();

  void ClearCurrentData();

  void CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr);

  void LoadGraphs(const KernelGraphPtr &graph_ptr);

  uint32_t GetFirstRunGraphId() const;

  uint32_t GetCurrentRootGraphId() const { return cur_root_graph_id_; }

  uint32_t GetPrevRootGraphId() const { return prev_root_graph_id_; }

  std::vector<KernelGraphPtr> GetStepGraphPtrList() const { return graph_ptr_step_vec_; }

  void InsertExecutedGraph(const KernelGraphPtr &graph_ptr) { (void)executed_graph_ptr_set_.insert(graph_ptr); }

  void SetGraphPtr(const KernelGraphPtr &graph_ptr) { graph_ptr_ = graph_ptr; }

  const KernelGraphPtr GetGraphPtr() const { return graph_ptr_; }

  const std::list<KernelGraphPtr> GetGraphPtrList() const { return graph_ptr_list_; }

  bool TensorExistsInCurrent(const std::string &tensor_name);

  // check if dump using debugger backend is enabled
  bool CheckDebuggerDumpEnabled() const;
  std::map<uint32_t, int32_t> GetGraphIterMap() { return graph_iter_num_map_; }

  void UpdateGraphIterMap(uint32_t graph_id, int32_t iter_num);

  std::vector<AnfNodePtr> GetParametersMindRT() const { return parameters_mindRT_; }

 private:
  // private constructor for singleton
  Debugger();

  // check if the graph is a dataset graph
  void CheckDatasetGraph();

  // serialize graph and get proto
  std::unique_ptr<debugger::GraphProto> GetGraphProto(const KernelGraphPtr &graph_ptr) const;

  void LoadSingleAnfnode(const AnfNodePtr &anf_node, const size_t output_index, uint32_t root_graph_id);

  void LoadSingleParameterMindRT(const AnfNodePtr &anf_node);

  // class members
  std::unique_ptr<DebugServices> debug_services_;
  KernelGraphPtr graph_ptr_;
  uint32_t device_id_;
  std::string device_target_;
  bool is_dataset_graph_;
  std::mutex access_lock_;
  uint32_t cur_root_graph_id_ = UINT32_MAX;
  uint32_t prev_root_graph_id_ = UINT32_MAX;
  std::list<KernelGraphPtr> graph_ptr_list_;
  // The vector of all the kernel graph pointers for the root graph that will execute in the current step.
  std::vector<KernelGraphPtr> graph_ptr_step_vec_;
  // The set of graph pointers that have been run in the current step.
  std::set<KernelGraphPtr> executed_graph_ptr_set_;
  // The vector of all the parameters for the current step for mindRT.
  std::vector<AnfNodePtr> parameters_mindRT_;
  std::vector<uint32_t> visited_root_graph_ids_;
  // map to store iter num in each epoch when dataset_sink_mode is true
  std::map<uint32_t, int32_t> graph_iter_num_map_;

  // singleton
  inline static std::mutex instance_lock_ = {};
  inline static std::shared_ptr<Debugger> debugger_ = nullptr;
  uint32_t not_dataset_graph_sum_;
  std::list<uint32_t> rungraph_id_list_;
  bool ascend_kernel_by_kernel_;
  bool enable_debugger_called_;
  std::string version_;
};
using DebuggerPtr = std::shared_ptr<Debugger>;

TOOLS_EXPORT void DebuggerReset();
TOOLS_EXPORT void DebuggerInit(const uint32_t, const std::string &);
TOOLS_EXPORT void DumpInGraphCompiler(const KernelGraphPtr &);
TOOLS_EXPORT bool DebuggerBackendEnabled();
TOOLS_EXPORT void DebuggerLoadGraphs(const KernelGraphPtr &);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEBUGGER_DEBUGGER_H_
