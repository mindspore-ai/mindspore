/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/common/context_util.h"
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <string>
#include "src/common/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
namespace {
template <class T>
void PassBasicProperties(std::shared_ptr<T> device_info, const lite::DeviceContext &device_context) {
  MS_ASSERT(device_info != nullptr);
  device_info->SetProvider(device_context.provider_);
  device_info->SetProviderDevice(device_context.provider_device_);
  device_info->SetAllocator(device_context.allocator_);
}

std::shared_ptr<mindspore::CPUDeviceInfo> CPUDeviceInfoFromCPUDeviceContext(const lite::DeviceContext &cpu_context) {
  if (cpu_context.device_type_ != DT_CPU) {
    MS_LOG(ERROR) << "function input parameter is not cpu context.";
    return nullptr;
  }
  auto cpu_info = std::make_shared<mindspore::CPUDeviceInfo>();
  MS_CHECK_TRUE_RET(cpu_info != nullptr, nullptr);
  cpu_info->SetEnableFP16(cpu_context.device_info_.cpu_device_info_.enable_float16_);
  PassBasicProperties(cpu_info, cpu_context);
  return cpu_info;
}

std::shared_ptr<mindspore::GPUDeviceInfo> GPUDeviceInfoFromGPUDeviceContext(const lite::DeviceContext &gpu_context) {
  if (gpu_context.device_type_ != DT_GPU) {
    MS_LOG(ERROR) << "function input parameter is not gpu context.";
    return nullptr;
  }
  auto gpu_info = std::make_shared<mindspore::GPUDeviceInfo>();
  MS_CHECK_TRUE_RET(gpu_info != nullptr, nullptr);
  gpu_info->SetEnableFP16(gpu_context.device_info_.gpu_device_info_.enable_float16_);
  gpu_info->SetDeviceID(gpu_context.device_info_.gpu_device_info_.gpu_device_id_);
  PassBasicProperties(gpu_info, gpu_context);
  return gpu_info;
}

std::shared_ptr<mindspore::KirinNPUDeviceInfo> NPUDeviceInfoFromNPUDeviceContext(
  const lite::DeviceContext &npu_context) {
  if (npu_context.device_type_ != DT_NPU) {
    MS_LOG(ERROR) << "function input parameter is not npu context.";
    return nullptr;
  }
  auto npu_info = std::make_shared<mindspore::KirinNPUDeviceInfo>();
  MS_CHECK_TRUE_RET(npu_info != nullptr, nullptr);
  npu_info->SetEnableFP16(npu_context.device_info_.npu_device_info_.enable_float16_);
  npu_info->SetFrequency(npu_context.device_info_.npu_device_info_.frequency_);
  PassBasicProperties(npu_info, npu_context);
  return npu_info;
}

std::vector<size_t> GetBatchSize(const std::string &batch_size) {
  std::vector<size_t> res;
  std::vector<std::string> batch_size_vec = StrSplit(batch_size, ",");
  for (const auto &item : batch_size_vec) {
    int32_t val;
    if (ConvertStrToInt(item, &val)) {
      auto tmp_val = static_cast<size_t>(val);
      res.push_back(tmp_val);
    } else {
      MS_LOG(ERROR) << "Convert str to num failed, val = " << item;
      return res;
    }
  }
  MS_LOG(INFO) << "Batch size of context: " << batch_size;
  return res;
}

std::shared_ptr<mindspore::AscendDeviceInfo> AscendDeviceInfoFromAscendDeviceContext(
  const lite::DeviceContext &ascend_context) {
  if (ascend_context.device_type_ != DT_ASCEND) {
    MS_LOG(ERROR) << "Function input parameter is not ascend context.";
    return nullptr;
  }
  auto ascend_info = std::make_shared<mindspore::AscendDeviceInfo>();
  MS_CHECK_TRUE_RET(ascend_info != nullptr, nullptr);
  ascend_info->SetDeviceID(ascend_context.device_info_.ascend_device_info_.device_id_);
  std::string batch_size = ascend_context.device_info_.ascend_device_info_.batch_size_;
  if (!batch_size.empty()) {
    auto val = GetBatchSize(batch_size);
    ascend_info->SetDynamicBatchSize(val);
  }
  ascend_info->SetDynamicImageSize(ascend_context.device_info_.ascend_device_info_.image_size_);
  return ascend_info;
}

std::shared_ptr<mindspore::DeviceInfoContext> CustomDeviceInfoFromCustomDeviceContext(
  const lite::DeviceContext &inner_context) {
  if (inner_context.device_type_ != DT_CUSTOM) {
    MS_LOG(ERROR) << "Function input parameter is not extended context.";
    return nullptr;
  }
  auto device_info = inner_context.device_info_.custom_device_info_.user_defined_device_info_;
  MS_CHECK_TRUE_RET(device_info != nullptr, nullptr);
  return device_info;
}
}  // namespace

mindspore::Context *MSContextFromContext(const std::shared_ptr<InnerContext> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    return nullptr;
  }
  auto ms_context = new (std::nothrow) mindspore::Context();
  if (ms_context == nullptr) {
    MS_LOG(ERROR) << "New Context failed";
    return nullptr;
  }
  ms_context->SetThreadNum(context->thread_num_);
  ms_context->SetThreadAffinity(context->affinity_core_list_);
#ifndef ENABLE_CLOUD_FUSION_INFERENCE
  ms_context->SetEnableParallel(context->enable_parallel_);
#endif
  if (context->delegate) {
    ms_context->SetDelegate(context->delegate);
  }
  auto &device_infos = ms_context->MutableDeviceInfo();
  std::map<DeviceType, std::function<std::shared_ptr<mindspore::DeviceInfoContext>(const lite::DeviceContext &)>>
    transfer_funcs = {{DT_CPU, CPUDeviceInfoFromCPUDeviceContext},
                      {DT_GPU, GPUDeviceInfoFromGPUDeviceContext},
                      {DT_NPU, NPUDeviceInfoFromNPUDeviceContext},
                      {DT_ASCEND, AscendDeviceInfoFromAscendDeviceContext},
                      {DT_CUSTOM, CustomDeviceInfoFromCustomDeviceContext}};
  for (auto &device_context : context->device_list_) {
    auto device_type = device_context.device_type_;
    if (transfer_funcs.find(device_type) == transfer_funcs.end()) {
      MS_LOG(ERROR) << "device type is invalid.";
      delete ms_context;
      return nullptr;
    }
    auto device_info = transfer_funcs[device_type](device_context);
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "transfer device context to device info failed.";
      delete ms_context;
      return nullptr;
    }
    if (device_type == DT_CPU) {
      ms_context->SetThreadAffinity(static_cast<int>(device_context.device_info_.cpu_device_info_.cpu_bind_mode_));
    }
    device_infos.push_back(device_info);
  }
  return ms_context;
}

bool DeviceTypePriority(const InnerContext *context, int device_type1, int device_type2) {
  /* dt1 > dt2    true
   * dt1 < dt2    false    */

  if (context == nullptr) {
    return false;
  }
  std::vector<DeviceContext> device_infos = context->device_list_;
  for (DeviceContext device_info : device_infos) {
    if (device_info.device_type_ == device_type1) {
      return true;
    }
    if (device_info.device_type_ == device_type2) {
      return false;
    }
  }
  return false;
}

DeviceType KernelArchToDeviceType(kernel::KERNEL_ARCH kernel_arch) {
  switch (kernel_arch) {
    case kernel::KERNEL_ARCH::kCPU:
      return DT_CPU;
    case kernel::KERNEL_ARCH::kGPU:
      return DT_GPU;
    case kernel::KERNEL_ARCH::kNPU:
      return DT_NPU;
    default:
      return DT_CPU;
  }
}
}  // namespace lite
}  // namespace mindspore
