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

#ifndef MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_AUTO_GRAD_GUARD_H_
#define MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_AUTO_GRAD_GUARD_H_

#include <mutex>
#include <string>
#include <utility>
#include <memory>
#include "ir/tensor.h"
#include "device_address/device_type.h"
#include "utils/device_manager_conf.h"
#include "utils/ms_utils.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class OpRunner;

using OpPtr = std::shared_ptr<OpRunner>;

struct PYNATIVE_UTILS_EXPORT OpStatus {
  OpStatus();

  OpStatus(bool _disable_mix_precision, device::DeviceType device_target)
      : disable_mix_precision(_disable_mix_precision), device_target(device_target) {}

  bool disable_mix_precision{false};
  device::DeviceType device_target{};
};

class PYNATIVE_UTILS_EXPORT OpRunStatus {
 public:
  static OpRunStatus &Get();

  const OpStatus &op_status();

  void set_run_info(OpStatus &&run_info);

  bool RequireGrad() const;

  void SetRequireGrad(bool require_grad);

  device::DeviceType device_target() const;

  void ResetRequireGrad(bool require_grad);

  void HeterBarrier(device::DeviceType device);

  bool IsSafeView() const;

  void SetIsSafeView(bool is_safe);

  void ResetIsSaveView(bool is_safe);

 private:
  struct ThreadState {
    OpStatus status{};
    bool require_grad{false};
    bool is_safe_view{true};
  };

  static ThreadState &GetThreadState();

  OpRunStatus();

  ~OpRunStatus() = default;
  DISABLE_COPY_AND_ASSIGN(OpRunStatus);

  // Change device name to device type latter.
  std::mutex heter_mutex_;
  device::DeviceType cur_device_;
};

class PYNATIVE_UTILS_EXPORT RequireGradGuard {
 public:
  explicit RequireGradGuard(bool require_grad) {
    origin_require_grad_ = OpRunStatus::Get().RequireGrad();
    OpRunStatus::Get().SetRequireGrad(require_grad);
  }

  ~RequireGradGuard() { OpRunStatus::Get().ResetRequireGrad(origin_require_grad_); }

 private:
  bool origin_require_grad_{false};
};

class PYNATIVE_UTILS_EXPORT IsSafeViewGuard {
 public:
  explicit IsSafeViewGuard(bool is_safe) {
    origin_is_safe_view_ = OpRunStatus::Get().IsSafeView();
    OpRunStatus::Get().SetIsSafeView(is_safe);
  }

  ~IsSafeViewGuard() { OpRunStatus::Get().ResetIsSaveView(origin_is_safe_view_); }

 private:
  bool origin_is_safe_view_{true};
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_AUTO_GRAD_GUARD_H_
