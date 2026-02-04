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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_GUARD_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_GUARD_H_

#include <string>
#include <utility>
#include <memory>
#include "ir/tensor.h"
#include "device_address/device_type.h"
#include "utils/device_manager_conf.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class OpRunner;

using OpPtr = std::shared_ptr<OpRunner>;

struct PYBOOST_API OpStatus {
  OpStatus();

  OpStatus(bool _disable_mix_precision, device::DeviceType device_target)
      : disable_mix_precision(_disable_mix_precision), device_target(device_target) {}

  bool disable_mix_precision{false};
  device::DeviceType device_target{};
};

class PYBOOST_API OpRunStatus {
 public:
  static OpRunStatus &Get();

  const OpStatus &op_status() { return status_; }

  void set_run_info(OpStatus &&run_info) { status_ = run_info; }

  bool RequireGrad() const { return require_grad_; }

  void SetRequireGrad(bool require_grad) { require_grad_ = require_grad; }

  device::DeviceType device_target() const { return status_.device_target; }

  void ResetRequireGrad(bool require_grad) { require_grad_ = require_grad; }

  void SetLastOp(const OpPtr &op) { last_op_ = op; }

  OpPtr GetLastOp() { return std::move(last_op_); }

  void HeterBarrier(device::DeviceType device);

  bool IsSafeView() const { return is_safe_view_; }

  void SetIsSafeView(bool is_safe) { is_safe_view_ = is_safe; }

  void ResetIsSaveView(bool is_safe) { is_safe_view_ = is_safe; }

 private:
  OpRunStatus();

  ~OpRunStatus() = default;
  DISABLE_COPY_AND_ASSIGN(OpRunStatus);

  OpStatus status_{};
  bool require_grad_{false};
  bool is_safe_view_{true};
  OpPtr last_op_{nullptr};
  // Change device name to device type latter.
  device::DeviceType cur_device_;
};

class PYBOOST_API RequireGradGuard {
 public:
  explicit RequireGradGuard(bool require_grad) {
    origin_require_grad_ = OpRunStatus::Get().RequireGrad();
    OpRunStatus::Get().SetRequireGrad(require_grad);
  }

  ~RequireGradGuard() { OpRunStatus::Get().ResetRequireGrad(origin_require_grad_); }

 private:
  bool origin_require_grad_{false};
};

class PYBOOST_API IsSafeViewGuard {
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
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_GUARD_H_
