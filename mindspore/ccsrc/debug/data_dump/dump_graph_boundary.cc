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
#include "debug/data_dump/dump_graph_boundary.h"

#include <iostream>
#include <vector>
#include <string>
#include "utils/ms_utils.h"
#include "utils/file_utils.h"
#include "utils/convert_utils_base.h"

namespace mindspore::datadump {
DumpGraphBoundary &DumpGraphBoundary::GetInstance() {
  static DumpGraphBoundary inst{};
  return inst;
}

void ReplaceSlashesWithUnderscores(std::string *str) {
  size_t pos = 0;
  while ((pos = str->find('/', pos)) != std::string::npos) {
    str->replace(pos, 1, "_");
    pos += 1;
  }
}

void DumpGraphBoundary::HookDumpTask(const KernelGraphPtr &kernel_graph,
                                     const std::vector<device::DeviceAddress *> &device_addr,
                                     const std::vector<std::pair<AnfNodeWeakPtr, size_t>> &nodes, void *stream,
                                     bool is_input) {
  if (!enable_) {
    return;
  }
  if (!spec_kernel_graph_.empty() && spec_kernel_graph_ != kernel_graph->ToString()) {
    return;
  }
  MS_LOG(INFO) << "entry hook =======";
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(stream);
  auto kernel_graph_name = kernel_graph->ToString();
  std::vector<std::string> names;
  std::vector<size_t> sizes;
  std::vector<uint8_t *> host_item;
  std::string mid_name = is_input ? "_input_" : "_output_";
  for (const auto &i : nodes) {
    auto node = i.first.lock();
    MS_EXCEPTION_IF_NULL(node);
    auto idx = i.second;
    auto file_name = kernel_graph_name;
    file_name.append("_" + node->fullname_with_scope() + mid_name + std::to_string(idx));
    ReplaceSlashesWithUnderscores(&file_name);
    (void)names.emplace_back(file_name);
    auto addr = device_addr[idx];
    MS_EXCEPTION_IF_NULL(addr);
    auto host_data = new (std::nothrow) uint8_t[addr->GetSize()];
    if (!addr->AsyncDeviceToHost(host_data, addr->GetSize(), stream)) {
      MS_LOG(ERROR) << "Call acl copy failed, name: " << names[idx] << ", size: " << addr->GetSize();
      delete[] host_data;
      return;
    }
    sizes.push_back(addr->GetSize());
    (void)host_item.emplace_back(host_data);
    MS_LOG(INFO) << "name: " << file_name << ", host addr: " << host_data << ", host size: " << addr->GetSize();
  }
  auto dc = DataContainer(names, sizes, host_item);
  (void)d_container_.emplace_back(dc);
}

void DumpGraphBoundary::DataDrop(device::DeviceContext *device_ctx) {
  if (!enable_) {
    return;
  }
  MS_LOG(INFO) << "Entry drop =======";
  device_ctx->device_res_manager_->SyncAllStreams();
  auto dir_path = FileUtils::CreateNotExistDirs("./dump_graph_boundary");
  if (!dir_path.has_value()) {
    MS_LOG(WARNING) << "Create dump graph boundary path failed.";
    d_container_.clear();
    return;
  }
  auto dir_path_pre = dir_path.value();
  for (auto &dc : d_container_) {
    for (size_t i = 0; i < dc.name_.size(); ++i) {
      auto name = dc.name_[i];
      auto size = dc.size_[i];
      auto data = dc.data_[i];
      std::string file_name = std::string(dir_path_pre) + "/" + name;
      MS_LOG(INFO) << "name: " << file_name << ", host addr: " << data << ", host size: " << size;
      std::ofstream outFile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
      if (!outFile.is_open()) {
        MS_LOG(ERROR) << "Failed to open file for writing." << file_name;
        d_container_.clear();
        return;
      }
      outFile.write(reinterpret_cast<char *>(data), SizeToLong(size));
      outFile.close();
    }
    dc.Clear();
  }
}

void DumpGraphBoundary::InitEnableFlag() {
  auto dgb_flag = common::GetEnv("MS_MEMORY_STATISTIC");
  if (dgb_flag.find("kernel") != std::string::npos) {
    spec_kernel_graph_ = dgb_flag;
    enable_ = true;
  } else {
    enable_ = dgb_flag == "3";
  }
}

}  // namespace mindspore::datadump
