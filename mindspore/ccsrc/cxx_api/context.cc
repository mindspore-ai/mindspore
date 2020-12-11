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
#include "include/api/context.h"
#include "utils/log_adapter.h"

namespace mindspore::api {
class Context::ContextImpl {
 public:
  ContextImpl() : device_target_("NotSet"), device_id_(0) {}
  ~ContextImpl() = default;
  const std::string &GetDeviceTarget() const { return device_target_; }
  void SetDeviceTarget(std::string_view device_target) { device_target_ = device_target; }
  uint32_t GetDeviceID() const { return device_id_; }
  void SetDeviceID(uint32_t device_id) { device_id_ = device_id; }

 private:
  std::string device_target_;
  uint32_t device_id_;
};

Context &Context::Instance() {
  static Context context;
  return context;
}

const std::string &Context::GetDeviceTarget() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetDeviceTarget();
}

Context &Context::SetDeviceTarget(const std::string &device_target) {
  MS_EXCEPTION_IF_NULL(impl_);
  impl_->SetDeviceTarget(device_target);
  return *this;
}

uint32_t Context::GetDeviceID() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetDeviceID();
}

Context &Context::SetDeviceID(uint32_t device_id) {
  MS_EXCEPTION_IF_NULL(impl_);
  impl_->SetDeviceID(device_id);
  return *this;
}

Context::Context() : impl_(std::make_shared<Context::ContextImpl>()) { MS_EXCEPTION_IF_NULL(impl_); }

Context::~Context() {}
}  // namespace mindspore::api
