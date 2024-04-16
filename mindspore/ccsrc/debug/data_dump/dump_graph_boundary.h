/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_GRAPH_BOUNDARY_H
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_GRAPH_BOUNDARY_H

#include <utility>
#include <vector>
#include <string>

#include "include/backend/device_address.h"
#include "include/backend/kernel_graph.h"
#include "runtime/hardware/device_context.h"

namespace mindspore::datadump {
class BACKEND_EXPORT DumpGraphBoundary {
 public:
  static DumpGraphBoundary &GetInstance();
  void HookDumpTask(const KernelGraphPtr &kernel_graph, const std::vector<device::DeviceAddress *> &device_addr,
                    const std::vector<std::pair<AnfNodeWeakPtr, size_t>> &nodes, void *stream, bool is_input = False);
  void DataDrop(device::DeviceContext *device_ctx);
  void InitEnableFlag();

  class DataContainer {
   public:
    DataContainer(std::vector<std::string> name, std::vector<size_t> size, std::vector<uint8_t *> data)
        : name_(std::move(name)), size_(std::move(size)), data_(std::move(data)) {}
    ~DataContainer() = default;
    void Clear() {
      name_.clear();
      size_.clear();
      for (auto &data : data_) {
        if (data != nullptr) {
          delete[] data;
          data = nullptr;
        }
      }
      data_.clear();
    }

    friend class DumpGraphBoundary;

   private:
    std::vector<std::string> name_{};
    std::vector<size_t> size_{};
    std::vector<uint8_t *> data_{};
  };

 private:
  DumpGraphBoundary() = default;
  ~DumpGraphBoundary() = default;
  bool enable_{false};
  std::string spec_kernel_graph_{""};
  std::vector<DataContainer> d_container_{};
};
}  // namespace mindspore::datadump

#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_DUMP_GRAPH_BOUNDARY_H
