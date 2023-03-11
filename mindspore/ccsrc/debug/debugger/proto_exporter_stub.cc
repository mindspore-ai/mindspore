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
#include "include/backend/debug/debugger/proto_exporter.h"
#include "utils/log_adapter.h"
namespace mindspore {
void DumpIRProtoWithSrcInfo(const FuncGraphPtr &func_graph, const std::string &suffix, const std::string &target_dir,
                            LocDebugDumpMode dump_location) {
  MS_LOG(ERROR) << "Not support DumpIRProtoWithSrcInfo ";
  return;
}

void DumpConstantInfo(const KernelGraphPtr &graph, const std::string &target_dir) {
  MS_LOG(ERROR) << "Not support DumpConstantInfo ";
  return;
}
}  // namespace mindspore
