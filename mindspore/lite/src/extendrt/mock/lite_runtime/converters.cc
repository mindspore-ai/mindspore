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

#include "extendrt/mock/lite_runtime/converters.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
constexpr static int kMaxNumOfDevices = 3;
constexpr static int kDefaultThreadNumTwo = 2;
constexpr static int kDefaultThreadNumFour = 4;
constexpr static int kCoreNumThreshold = 32;
constexpr static int kDefaultInterOpParallelNum = 1;

void ContextUtils::SetContextAttr(int32_t thread_num, int32_t inter_op_parallel_num,
                                  const std::vector<int32_t> &affinity_core_list, lite::InnerContext *inner_context) {
  inner_context->thread_num_ = thread_num;
  inner_context->inter_op_parallel_num_ = inter_op_parallel_num;
  inner_context->affinity_core_list_ = affinity_core_list;
}

Status ContextUtils::AddCpuDevice(const std::shared_ptr<Allocator> &allocator, int affinity_mode, bool enable_fp16,
                                  const std::string &provider, const std::string &provider_device,
                                  lite::InnerContext *inner_context) {
  inner_context->allocator = allocator;
  if (!IsAffinityModeValid(affinity_mode)) {
    MS_LOG(ERROR) << "Invalid affinity mode, only supports 0:no affinities, 1:big cores first, 2:little cores first.";
    return kLiteInputParamInvalid;
  }
  lite::DeviceInfo device_info;
  device_info.cpu_device_info_ = {enable_fp16, static_cast<lite::CpuBindMode>(affinity_mode)};
  inner_context->device_list_.push_back({lite::DT_CPU, device_info, provider, provider_device, allocator});
  return kSuccess;
}

Status ContextUtils::AddGpuDevice(bool enable_fp16, uint32_t device_id, int rank_id, int group_size,
                                  bool enable_gl_texture, void *gl_context, void *gl_display,
                                  const std::string &provider, const std::string &provider_device,
                                  const std::shared_ptr<Allocator> &allocator, lite::InnerContext *inner_context) {
  lite::DeviceInfo device_info;
  device_info.gpu_device_info_ = {enable_fp16,       device_id,  rank_id,   group_size,
                                  enable_gl_texture, gl_context, gl_display};
  inner_context->device_list_.push_back({lite::DT_GPU, device_info, provider, provider_device, allocator});
  return kSuccess;
}

Status ContextUtils::AddNpuDevice(int frequency, lite::InnerContext *inner_context) {
  lite::DeviceInfo device_info;
  device_info.npu_device_info_.frequency_ = frequency;
  inner_context->device_list_.push_back({lite::DT_NPU, device_info});
  return kSuccess;
}

Status ContextUtils::AddAscendDevice(lite::InnerContext *inner_context, DeviceInfoContext *device) {
  lite::DeviceInfo device_info;
  auto ascend_context = device->Cast<AscendDeviceInfo>();
  device_info.ascend_device_info_ = {ascend_context->GetDeviceID(), ascend_context->GetDynamicBatchSize(),
                                     ascend_context->GetDynamicImageSize()};
  inner_context->device_list_.push_back({lite::DT_ASCEND, device_info});
  return kSuccess;
}

void ContextUtils::ResetContextDefaultParam(Context *context) {
  if (context->GetInterOpParallelNum() == 0) {
    context->SetInterOpParallelNum(kDefaultInterOpParallelNum);
  }
  if (context->GetThreadNum() != 0) {
    return;
  }
  MS_LOG(INFO) << "thread num is 0, will set the optimal number of threads";
#if defined(__ANDROID__) || defined(MS_COMPILE_IOS)
  context->SetThreadNum(kDefaultThreadNumTwo);
  MS_LOG(INFO) << "Set the number of threads to " << kDefaultThreadNumTwo;
  return;
#endif
  auto core_num = lite::GetCoreNum();
  if (core_num <= kCoreNumThreshold) {
    context->SetThreadNum(kDefaultThreadNumTwo);
    MS_LOG(INFO) << "Set the number of threads to " << kDefaultThreadNumTwo;
  } else {
    context->SetThreadNum(kDefaultThreadNumFour);
    MS_LOG(INFO) << "Set the number of threads to " << kDefaultThreadNumFour;
  }
  return;
}

std::shared_ptr<lite::InnerContext> ContextUtils::Convert(Context *context) {
  auto inner_context = std::make_shared<lite::InnerContext>();
  if ((context == nullptr) || (inner_context == nullptr)) {
    MS_LOG(ERROR) << "Invalid context pointers.";
    return nullptr;
  }
  ResetContextDefaultParam(context);
  auto device_list = context->MutableDeviceInfo();
  if (device_list.size() == 0 || device_list.size() > kMaxNumOfDevices) {
    MS_LOG(ERROR) << "Device num, support min: 1, max: " << kMaxNumOfDevices;
    return nullptr;
  }
  if (context->GetInterOpParallelNum() <= 0 || context->GetInterOpParallelNum() > context->GetThreadNum()) {
    MS_LOG(ERROR) << "Invalid inter op parallel num : " << context->GetInterOpParallelNum()
                  << " | thread num: " << context->GetThreadNum();
    return nullptr;
  }
  SetContextAttr(context->GetThreadNum(), context->GetInterOpParallelNum(), context->GetThreadAffinityCoreList(),
                 inner_context.get());
  inner_context->device_list_.clear();
  Status ret = kLiteError;
  for (auto &device : device_list) {
    MS_CHECK_TRUE_RET(device != nullptr, nullptr);
    if (device->GetDeviceType() == kCPU) {
      auto cpu_context = device->Cast<CPUDeviceInfo>();
      if (cpu_context->GetAllocator() == nullptr) {
        cpu_context->SetAllocator(Allocator::Create());
      }
      ret = AddCpuDevice(cpu_context->GetAllocator(), context->GetThreadAffinityMode(), cpu_context->GetEnableFP16(),
                         cpu_context->GetProvider(), cpu_context->GetProviderDevice(), inner_context.get());
    }
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Add device failed!";
      return nullptr;
    }
  }
  return inner_context;
}
}  // namespace mindspore
