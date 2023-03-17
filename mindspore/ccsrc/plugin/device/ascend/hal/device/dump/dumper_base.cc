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
#include "plugin/device/ascend/hal/device/dump/dumper_base.h"

#include <memory>
#include <string>
#include "utility"
#include "acl/acl_rt.h"
#include "runtime/kernel.h"
#include "runtime/rt_model.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "debug/data_dump/dump_json_parser.h"
#include "proto/op_mapping_info.pb.h"
#include "plugin/device/ascend/hal/device/ge_types_convert.h"
#include "runtime/dev.h"
#include "runtime/mem.h"
#include "include/common/utils/comm_manager.h"
#include "google/protobuf/util/json_util.h"

namespace mindspore {
namespace device {
namespace ascend {
#ifndef ENABLE_SECURITY
bool KernelNeedDump(const CNodePtr &kernel) {
  if (AnfAlgo::GetKernelType(kernel) != TBE_KERNEL && AnfAlgo::GetKernelType(kernel) != AICPU_KERNEL &&
      AnfAlgo::GetKernelType(kernel) != AKG_KERNEL && AnfAlgo::GetKernelType(kernel) != HCCL_KERNEL &&
      AnfAlgo::GetKernelType(kernel) != ACL_KERNEL) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  // dump all kernel if mode is set 0 in data_dump.json
  return DumpJsonParser::GetInstance().NeedDump(kernel->fullname_with_scope());
}
#endif

void SetDumpShape(const ShapeVector &ms_shape, NotNull<aicpu::dump::Shape *> dump_shape) {
  for (auto &dim : ms_shape) {
    dump_shape->add_dim(dim);
  }
}

void SetOpDebugMappingInfo(const NotNull<aicpu::dump::OpMappingInfo *> dump_info, const uint32_t debug_task_id,
                           const uint32_t debug_stream_id, const void *op_debug_dump_args) {
  MS_LOG(INFO) << "[DumperBase] Add op debug info to OpMappingInfo, task id = " << debug_task_id
               << ", stream id = " << debug_stream_id;
  aicpu::dump::Task task;
  task.set_end_graph(false);
  task.set_task_id(debug_task_id);
  task.set_stream_id(debug_stream_id);
  MS_EXCEPTION_IF_NULL(task.mutable_op());
  task.mutable_op()->set_op_name(kNodeNameOpDebug);
  task.mutable_op()->set_op_type(kOpTypeOpDebug);

  aicpu::dump::Output output;
  output.set_data_type(static_cast<int>(ge::proto::DataType::DT_UINT8));
  output.set_format(static_cast<int>(ge::Format::FORMAT_ND));

  MS_EXCEPTION_IF_NULL(output.mutable_shape());
  output.mutable_shape()->add_dim(kOpDebugShape);

  output.set_original_name(kNodeNameOpDebug);
  output.set_original_output_index(0);
  output.set_original_output_format(static_cast<int>(ge::Format::FORMAT_ND));
  output.set_original_output_data_type(static_cast<int>(ge::proto::DataType::DT_UINT8));
  // due to lhisi virtual addr bug, cannot use args now
  output.set_address(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(op_debug_dump_args)));
  output.set_size(kOpDebugHostMemSize);

  MS_EXCEPTION_IF_NULL(task.mutable_output());
  task.mutable_output()->Add(std::move(output));
  MS_EXCEPTION_IF_NULL(dump_info->mutable_task());
  dump_info->mutable_task()->Add(std::move(task));
}

void RtLoadDumpData(const aicpu::dump::OpMappingInfo &dump_info, void **ptr) {
  std::string proto_str;
  size_t proto_size = dump_info.ByteSizeLong();
  bool ret = dump_info.SerializeToString(&proto_str);

  std::string proto_json;
  (void)google::protobuf::util::MessageToJsonString(dump_info, &proto_json);

  if (!ret || proto_size == 0) {
    MS_LOG(EXCEPTION) << "[DumperBase] Protobuf SerializeToString failed, proto size %zu.";
  }

  if (ptr == nullptr) {
    MS_LOG(ERROR) << "[DumperBase] rtMalloc failed, ptr is nullptr";
    return;
  }

  rtError_t rt_ret = rtMalloc(ptr, proto_size, RT_MEMORY_HBM, 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DumperBase] Call rtMalloc failed";
  }
  rt_ret = aclrtMemcpy(*ptr, proto_size, proto_str.c_str(), proto_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DumperBase] Call aclrtMemcpy failed";
  }

  MS_LOG(INFO) << "[DumperBase] rtDatadumpInfoLoad start";
  rt_ret = rtDatadumpInfoLoad(*ptr, SizeToUint(proto_size));
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "[DumperBase] Call rtDatadumpInfoLoad failed";
  }
}

void ReleaseDevMem(void **ptr) {
  if (ptr == nullptr) {
    return;
  }
  if (*ptr != nullptr) {
    rtError_t rt_error = rtFree(*ptr);
    if (rt_error != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "[DumperBase] Call rtFree failed, ret:" << rt_error;
    }
    *ptr = nullptr;
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
