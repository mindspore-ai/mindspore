/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_DUMPER_BASE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_DUMPER_BASE_H_
#include <map>
#include <string>
#include "ir/anf.h"
#include "include/common/utils/contract.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "acl/acl_rt.h"
#include "runtime/rt_model.h"
#include "proto/op_mapping_info.pb.h"

namespace aicpu {
namespace dump {
class OpMappingInfo;
class Task;
}  // namespace dump
}  // namespace aicpu
namespace mindspore {
namespace device {
namespace ascend {
static constexpr uint32_t kAicpuLoadFlag = 1;
static constexpr uint32_t kAicpuUnloadFlag = 0;
static constexpr uint32_t kTupleTaskId = 0;
static constexpr uint32_t kTupleStreamId = 1;
static constexpr uint32_t kTupleArgs = 2;
static constexpr uint64_t kOpDebugShape = 2048;
static constexpr uint64_t kOpDebugHostMemSize = 2048;
static constexpr uint64_t kOpDebugDevMemSize = sizeof(void *);
static constexpr uint8_t kNoOverflow = 0;
static constexpr uint8_t kAiCoreOverflow = 0x1;
static constexpr uint8_t kAtomicOverflow = (0x1 << 1);
static constexpr uint8_t kAllOverflow = (kAiCoreOverflow | kAtomicOverflow);
static const std::map<uint32_t, std::string> kOverflowModeStr = {{kNoOverflow, "NoOverflow"},
                                                                 {kAiCoreOverflow, "AiCoreOverflow"},
                                                                 {kAtomicOverflow, "AtomicOverflow"},
                                                                 {kAllOverflow, "AllOverflow"}};
constexpr const char *kNodeNameOpDebug = "Node_OpDebug";
constexpr const char *kOpTypeOpDebug = "Opdebug";
constexpr const char *kNodeNameEndGraph = "Node_EndGraph";
constexpr const char *kOpTypeOpEndGraph = "EndGraph";
static constexpr auto kCurLoopCountName = "current_loop_count";
static constexpr auto kCurEpochCountName = "current_epoch_count";
static constexpr auto kConstLoopNumInEpochName = "const_loop_num_in_epoch";
#ifndef ENABLE_SECURITY
bool KernelNeedDump(const CNodePtr &kernel);
#endif
void SetOpDebugMappingInfo(const NotNull<aicpu::dump::OpMappingInfo *> dump_info, const uint32_t debug_task_id,
                           const uint32_t debug_stream_id, const void *op_debug_dump_args);
void SetDumpShape(const ShapeVector &ms_shape, NotNull<aicpu::dump::Shape *> dump_shape);
void RtLoadDumpData(const aicpu::dump::OpMappingInfo &dump_info, void **ptr);
void ReleaseDevMem(void **ptr);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DUMP_DUMPER_BASE_H_
