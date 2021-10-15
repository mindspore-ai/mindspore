/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/ge_runtime/task/aicpu_task.h"
#include <vector>
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "runtime/device/ascend/ge_runtime/task/task_factory.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "mindspore/core/utils/convert_utils_base.h"

namespace mindspore::ge::model_runner {
AicpuTask::AicpuTask(const ModelContext &model_context, const std::shared_ptr<AicpuTaskInfo> &task_info)
    : TaskRepeater<AicpuTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      args_(nullptr),
      ext_info_(nullptr),
      input_output_addr_(nullptr) {
  MS_EXCEPTION_IF_NULL(task_info_);

  auto stream_list = model_context.stream_list();
  if (stream_list.size() == 1) {
    stream_ = stream_list[0];
  } else if (stream_list.size() > task_info_->stream_id()) {
    stream_ = stream_list[task_info_->stream_id()];
  } else {
    MS_LOG(EXCEPTION) << "Index: " << task_info_->stream_id() << " >= stream_list.size(): " << stream_list.size();
  }
}

AicpuTask::~AicpuTask() {
  ReleaseRtMem(&args_);
  ReleaseRtMem(&ext_info_);
}

void AicpuTask::Distribute() {
  MS_LOG(INFO) << "InitAicpuTask start.";
  std::vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), task_info_->input_data_addrs().begin(), task_info_->input_data_addrs().end());
  io_addrs.insert(io_addrs.end(), task_info_->output_data_addrs().begin(), task_info_->output_data_addrs().end());
  auto io_addrs_num = static_cast<uint32_t>(io_addrs.size());
  auto io_addrs_size = static_cast<uint32_t>(io_addrs_num * sizeof(void *));
  constexpr uint32_t io_addr_offset = sizeof(aicpu::AicpuParamHead);
  uint32_t node_def_len_offset = io_addr_offset + io_addrs_size;
  uint32_t node_def_addr_offset = node_def_len_offset + sizeof(uint32_t);
  uint32_t args_size = sizeof(aicpu::AicpuParamHead) + io_addrs_size +
                       static_cast<uint32_t>(task_info_->node_def().size()) + sizeof(uint32_t);

  // Malloc device memory for args
  rtError_t rt_ret = rtMalloc(&args_, args_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtMalloc failed, ret: " << rt_ret;
  }

  SetAicpuParamHead(args_size, io_addrs_num);
  SetInputOutputAddrs(io_addrs, io_addr_offset);
  SetNodeDef(node_def_len_offset, node_def_addr_offset);

  // for data dump
  input_output_addr_ = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + io_addr_offset);
  auto dump_flag = task_info_->dump_flag() ? RT_KERNEL_DUMPFLAG : RT_KERNEL_DEFAULT;

  MS_LOG(INFO) << "Distribute AicpuTask start, args_size = " << args_size << ", io_addrs_num =" << io_addrs_num
               << ", so_name = " << task_info_->so_name() << ", kernel_name = " << task_info_->kernel_name()
               << ", dump_flag = " << dump_flag;
  rt_ret = rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(task_info_->so_name().data()),
                                     reinterpret_cast<const void *>(task_info_->kernel_name().data()), 1, args_,
                                     args_size, nullptr, stream_, dump_flag);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtCpuKernelLaunchWithFlag failed, ret: " << rt_ret;
  }

  MS_LOG(INFO) << "Distribute AicpuTask end.";
}

void AicpuTask::ReleaseRtMem(void **ptr) noexcept {
  if (ptr == nullptr || *ptr == nullptr) {
    return;
  }

  rtError_t rt_ret = rtFree(*ptr);
  if (rt_ret != RT_ERROR_NONE) {
    return;
  }
  *ptr = nullptr;
}

void AicpuTask::SetAicpuParamHead(uint32_t args_size, uint32_t io_addrs_num) {
  aicpu::AicpuParamHead aicpu_param_head;
  aicpu_param_head.length = args_size;
  aicpu_param_head.ioAddrNum = io_addrs_num;

  const auto &ext_info = task_info_->ext_info();
  uint32_t ext_size = SizeToUint(ext_info.size());
  if (ext_info.empty()) {
    aicpu_param_head.extInfoLength = 0;
    aicpu_param_head.extInfoAddr = 0;
  } else {
    rtError_t flag = rtMalloc(&ext_info_, ext_size, RT_MEMORY_HBM);
    if (flag != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtMalloc failed, ret: " << flag;
    }

    flag = rtMemcpy(ext_info_, ext_size, const_cast<void *>(reinterpret_cast<const void *>(ext_info.data())), ext_size,
                    RT_MEMCPY_HOST_TO_DEVICE);
    if (flag != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtMemcpy failed, ret: " << flag;
    }

    MS_LOG(INFO) << "ext info size: " << ext_size;
    aicpu_param_head.extInfoLength = ext_size;
    aicpu_param_head.extInfoAddr = reinterpret_cast<uintptr_t>(ext_info_);
  }

  // Memcpy AicpuParamHead
  auto rt_ret = rtMemcpy(args_, sizeof(aicpu::AicpuParamHead), reinterpret_cast<void *>(&aicpu_param_head),
                         sizeof(aicpu::AicpuParamHead), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtMemcpy failed, ret: " << rt_ret;
  }
}

void AicpuTask::SetInputOutputAddrs(const std::vector<void *> &io_addrs, uint32_t io_addr_offset) {
  // Memcpy io addrs
  if (!io_addrs.empty()) {
    auto rt_ret = rtMemcpy(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + io_addr_offset),
                           static_cast<uint32_t>(io_addrs.size()) * sizeof(void *), io_addrs.data(),
                           static_cast<uint32_t>(io_addrs.size()) * sizeof(void *), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rt api rtMemcpy failed, ret: " << rt_ret;
    }
  }
}

void AicpuTask::SetNodeDef(uint32_t node_def_len_offset, uint32_t node_def_addr_offset) {
  // Memcpy node def
  auto size = task_info_->node_def().size();
  auto rt_ret =
    rtMemcpy(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + node_def_len_offset), sizeof(uint32_t),
             reinterpret_cast<const void *>(&size), sizeof(uint32_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtMemcpy failed, ret: " << rt_ret;
  }

  // Memcpy node def
  rt_ret = rtMemcpy(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + node_def_addr_offset),
                    task_info_->node_def().size(), reinterpret_cast<const void *>(task_info_->node_def().data()),
                    task_info_->node_def().size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rt api rtMemcpy failed, ret: " << rt_ret;
  }
}

REGISTER_TASK(TaskInfoType::AICPU, AicpuTask, AicpuTaskInfo);
}  // namespace mindspore::ge::model_runner
