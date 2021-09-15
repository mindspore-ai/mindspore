/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_DATADUMP_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_DATADUMP_H_
#include <tuple>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "backend/session/kernel_graph.h"

namespace aicpu {
namespace dump {
class OpMappingInfo;
class Task;
}  // namespace dump
}  // namespace aicpu
namespace mindspore {
namespace device {
namespace ascend {
// tuple(op_name, task_id, stream_id, args)
using RuntimeInfo = std::tuple<uint32_t, uint32_t, void *>;
class DataDumper {
 public:
  DataDumper(const session::KernelGraph *kernel_graph, NotNull<std::function<void *()>> model_handle)
      : model_handle_(model_handle),
        debug_task_id_(-1),
        debug_stream_id_(-1),
        op_debug_buffer_addr_(nullptr),
        op_debug_dump_args_(nullptr),
        load_flag_(false),
        dev_load_mem_(nullptr),
        dev_unload_mem_(nullptr),
        graph_id_(UINT32_MAX),
        kernel_graph_(kernel_graph) {}
  ~DataDumper();
  void set_runtime_info(const std::map<std::string, std::shared_ptr<RuntimeInfo>> &runtime_info) {
    runtime_info_map_ = runtime_info;
  }
#ifndef ENABLE_SECURITY
  void LoadDumpInfo();
  void OpDebugRegister();
  void OpDebugUnregister();
#endif
  void UnloadDumpInfo();

 private:
  void ReleaseDevMem(void **ptr) const noexcept;
#ifndef ENABLE_SECURITY
  bool KernelNeedDump(const CNodePtr &kernel) const;
  void SetOpMappingInfo(NotNull<aicpu::dump::OpMappingInfo *> dump_info) const;
#endif
  void SetOpDebugMappingInfo(const NotNull<aicpu::dump::OpMappingInfo *> dump_info) const;
  void ConstructDumpTask(NotNull<const CNodePtr &> kernel, NotNull<aicpu::dump::Task *> dump_task) const;
#ifndef ENABLE_SECURITY
  void GetNeedDumpKernelList(NotNull<std::map<std::string, CNodePtr> *> kernel_map) const;
  static void DumpKernelOutput(const CNodePtr &kernel, void *args, NotNull<aicpu::dump::Task *> task);
  static void DumpKernelInput(const CNodePtr &kernel, void *args, NotNull<aicpu::dump::Task *> task);
#endif
  static std::string StripUniqueId(const std::string node_name);
  static void RtLoadDumpData(const aicpu::dump::OpMappingInfo &dump_info, void **ptr);

  std::function<void *()> model_handle_;
  uint32_t debug_task_id_;
  uint32_t debug_stream_id_;
  void *op_debug_buffer_addr_;
  void *op_debug_dump_args_;
  bool load_flag_;
  void *dev_load_mem_;
  void *dev_unload_mem_;
  uint32_t graph_id_;
  std::vector<std::string> dump_kernel_names_;
  const session::KernelGraph *kernel_graph_;
  std::map<std::string, std::shared_ptr<RuntimeInfo>> runtime_info_map_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_DATADUMP_H_
