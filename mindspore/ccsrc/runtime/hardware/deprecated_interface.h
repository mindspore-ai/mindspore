/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEPRECATED_INTERFACE_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEPRECATED_INTERFACE_H_

#include <vector>
#include <string>
#include <memory>
#include "ir/func_graph.h"
#include "pybind11/pytypes.h"
#include "utils/ms_context.h"
#include "include/api/types.h"
#include "nlohmann/json.hpp"

namespace mindspore {
namespace device {
class DeprecatedInterface {
 public:
  virtual ~DeprecatedInterface() = default;

  // ge
  virtual void DoExecNonInputGraph(const std::string &phase) {}
  virtual bool InitExecDataset(const std::string &queue_name, int64_t size, int64_t batch_size,
                               const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                               const std::vector<int64_t> &input_indexes, const std::string &phase) {
    return true;
  }
  virtual void ExportDFGraph(const std::string &file_name, const std::string &phase, const pybind11::object &encrypt,
                             char *key) {}

  virtual FuncGraphPtr BuildDFGraph(const FuncGraphPtr &anf_graph, const pybind11::dict &init_params) {
    return nullptr;
  }
  virtual void ClearGraphWrapper() {}
  virtual void ClearOpAdapterMap() {}
  virtual void EraseGeResource() {}

  // ascend
  virtual uint32_t InitCollective() { return 0; }  // return device id
  virtual void DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) {}
  virtual bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) { return true; }
  virtual bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) { return true; }
  virtual bool IsTsdOpened(const std::shared_ptr<MsContext> &inst_context) { return true; }
  virtual void AclOptimizer(const FuncGraphPtr &graph) {}
  virtual bool CheckIsAscend910Soc() { return true; }
  virtual void AclLoadModel(Buffer *om_data) {}
  // gpu
  virtual int GetGPUCapabilityMajor() { return -1; }
  virtual int GetGPUCapabilityMinor() { return -1; }
  virtual int GetGPUMultiProcessorCount() { return -1; }
  virtual void GPUInitCollective() {}
  virtual void GPUFinalizeCollective() {}
  virtual uint32_t GPUGetRankID(const std::string &group_name) { return 0; }
  virtual uint32_t GPUGetRankSize(const std::string &group_name) { return 0; }
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEPRECATED_INTERFACE_H_
