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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEPRECATED_INTERFACE_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEPRECATED_INTERFACE_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
class GeDeviceContext;

class AscendDeprecatedInterface : public DeprecatedInterface {
 public:
  explicit AscendDeprecatedInterface(GeDeviceContext *ge_device_context) : ge_device_context_(ge_device_context) {}
  ~AscendDeprecatedInterface() override = default;

  // for ge
  void DoExecNonInputGraph(const std::string &phase) override;
  bool InitExecDataset(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, const std::string &phase) override;
  void ExportDFGraph(const std::string &file_name, const std::string &phase, const pybind11::object &encrypt,
                     char *key) override;
  FuncGraphPtr BuildDFGraph(const FuncGraphPtr &anf_graph, const pybind11::dict &init_params) override;
  void ClearGraphWrapper() override;
  void ClearOpAdapterMap() override;
  void EraseGeResource() override;
  // for ascend
  uint32_t InitCollective() override;
  void DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) override;

  bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) override;
  bool CloseTsd(const std::shared_ptr<MsContext> &ms_context_ptr, bool force) override;
  bool IsTsdOpened(const std::shared_ptr<MsContext> &inst_context) override;
  void AclOptimizer(const FuncGraphPtr &graph) override;
  bool CheckIsAscend910Soc() override;
  void AclLoadModel(Buffer *om_data) override;

 private:
  GeDeviceContext *const ge_device_context_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_DEPRECATED_INTERFACE_H_
