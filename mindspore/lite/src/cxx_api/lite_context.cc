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
#include "include/api/lite_context.h"
#include <string>
#include <memory>
#include <any>
#include "include/api/types.h"
#include "src/common/log_adapter.h"

namespace mindspore {

constexpr char kVendorName[] = "vendor_name";
constexpr char kThreadNum[] = "thread_name";
constexpr char kAllocator[] = "allocator";
constexpr char kCPU[] = "cpu";
constexpr char kCPUEanbleFp16[] = "cpu_enable_fp16";
constexpr char kCPUBindMode[] = "cpu_bind_mode";
constexpr char kGPU[] = "gpu";
constexpr char kGPUEanbleFp16[] = "gpu_enable_fp16";
constexpr char kNPU[] = "npu";
constexpr char kNPUFrequency[] = "npu_frequency";

void Context::Clear(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  context->context_.clear();
}

void Context::SetAsDefault(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  context->context_.clear();
  context->context_.emplace(kCPU, true);
}

void Context::SetVendorName(const std::shared_ptr<Context> &context, const std::string &name) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kVendorName);
  if (iter != context->context_.end()) {
    iter->second = name;
  } else {
    context->context_.emplace(kVendorName, name);
  }
}

std::string Context::GetVendorName(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return std::string();
  }
  auto iter = context->context_.find(kVendorName);
  if (iter != context->context_.end()) {
    return std::any_cast<const std::string>(iter->second);
  }
  return std::string();
}

void Context::SetThreadNum(const std::shared_ptr<Context> &context, int num) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kThreadNum);
  if (iter != context->context_.end()) {
    iter->second = num;
  } else {
    context->context_.emplace(kThreadNum, num);
  }
}

int Context::GetThreadNum(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return 0;
  }
  auto iter = context->context_.find(kThreadNum);
  if (iter != context->context_.end()) {
    return std::any_cast<int>(iter->second);
  }
  return 2;
}

void Context::SetAllocator(const std::shared_ptr<Context> &context, std::shared_ptr<lite::Allocator> alloc) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kAllocator);
  if (iter != context->context_.end()) {
    iter->second = alloc;
  } else {
    context->context_.emplace(kAllocator, alloc);
  }
}

std::shared_ptr<lite::Allocator> Context::GetAllocator(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return nullptr;
  }
  auto iter = context->context_.find(kAllocator);
  if (iter != context->context_.end()) {
    return std::any_cast<std::shared_ptr<lite::Allocator>>(iter->second);
  }
  return nullptr;
}

void Context::ConfigCPU(const std::shared_ptr<Context> &context, bool conf) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kCPU);
  if (iter != context->context_.end()) {
    iter->second = conf;
  } else {
    context->context_.emplace(kCPU, conf);
  }
}

bool Context::IfCPUEnabled(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return false;
  }
  auto iter = context->context_.find(kCPU);
  if (iter != context->context_.end()) {
    return std::any_cast<bool>(iter->second);
  }
  return false;
}

void Context::ConfigCPUFp16(const std::shared_ptr<Context> &context, bool conf) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kCPUEanbleFp16);
  if (iter != context->context_.end()) {
    iter->second = conf;
  } else {
    context->context_.emplace(kCPUEanbleFp16, conf);
  }
}

bool Context::IfCPUFp16Enabled(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return false;
  }
  auto iter = context->context_.find(kCPUEanbleFp16);
  if (iter != context->context_.end()) {
    return std::any_cast<bool>(iter->second);
  }
  return false;
}

void Context::SetCPUBindMode(const std::shared_ptr<Context> &context, lite::CpuBindMode mode) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kCPUBindMode);
  if (iter != context->context_.end()) {
    iter->second = mode;
  } else {
    context->context_.emplace(kCPUBindMode, mode);
  }
}

lite::CpuBindMode Context::GetCPUBindMode(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return lite::NO_BIND;
  }
  auto iter = context->context_.find(kCPUBindMode);
  if (iter != context->context_.end()) {
    return std::any_cast<lite::CpuBindMode>(iter->second);
  }
  return lite::MID_CPU;
}

void Context::ConfigGPU(const std::shared_ptr<Context> &context, bool conf) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kGPU);
  if (iter != context->context_.end()) {
    iter->second = conf;
  } else {
    context->context_.emplace(kGPU, conf);
  }
}

bool Context::IfGPUEnabled(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return false;
  }
  auto iter = context->context_.find(kGPU);
  if (iter != context->context_.end()) {
    return std::any_cast<bool>(iter->second);
  }
  return false;
}

void Context::ConfigGPUFp16(const std::shared_ptr<Context> &context, bool conf) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kGPUEanbleFp16);
  if (iter != context->context_.end()) {
    iter->second = conf;
  } else {
    context->context_.emplace(kGPUEanbleFp16, conf);
  }
}

bool Context::IfGPUFp16Enabled(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return false;
  }
  auto iter = context->context_.find(kGPUEanbleFp16);
  if (iter != context->context_.end()) {
    return std::any_cast<bool>(iter->second);
  }
  return false;
}

void Context::ConfigNPU(const std::shared_ptr<Context> &context, bool conf) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kNPU);
  if (iter != context->context_.end()) {
    iter->second = conf;
  } else {
    context->context_.emplace(kNPU, conf);
  }
}

bool Context::IfNPUEnabled(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return false;
  }
  auto iter = context->context_.find(kNPU);
  if (iter != context->context_.end()) {
    return std::any_cast<bool>(iter->second);
  }
  return false;
}

void Context::SetNPUFrequency(const std::shared_ptr<Context> &context, int freq) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return;
  }
  auto iter = context->context_.find(kNPUFrequency);
  if (iter != context->context_.end()) {
    iter->second = freq;
  } else {
    context->context_.emplace(kNPUFrequency, freq);
  }
}

int Context::GetNPUFrequency(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Context is nullptr.";
    return 0;
  }
  auto iter = context->context_.find(kNPUFrequency);
  if (iter != context->context_.end()) {
    return std::any_cast<int>(iter->second);
  }
  return 3;
}

}  // namespace mindspore
