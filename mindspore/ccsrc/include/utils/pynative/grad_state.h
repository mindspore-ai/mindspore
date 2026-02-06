/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_GRAD_STATE_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_GRAD_STATE_H_

#include "include/utils/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace pynative {
class COMMON_EXPORT GradState {
 public:
  static GradState &Get();

  bool grad_flag() const { return grad_flag_; }
  void set_grad_flag(bool grad_flag) { grad_flag_ = grad_flag; }
  bool enable_grad() const { return enable_grad_; }
  void set_enable_grad(bool enable_grad) { enable_grad_ = enable_grad; }

  bool RequiresGrad() const { return enable_grad() && grad_flag(); }

 private:
  GradState() = default;
  ~GradState() = default;
  DISABLE_COPY_AND_ASSIGN(GradState);

  bool grad_flag_{false};
  bool enable_grad_{true};
};

class COMMON_EXPORT GradFlagGuard {
 public:
  explicit GradFlagGuard(bool grad_flag) {
    prev_grad_flag_ = GradState::Get().grad_flag();
    GradState::Get().set_grad_flag(grad_flag);
  }
  ~GradFlagGuard() { GradState::Get().set_grad_flag(prev_grad_flag_); }

 private:
  bool prev_grad_flag_{false};
};

class NoGradGuard {
 public:
  NoGradGuard() : enable_grad_(GradState::Get().enable_grad()) { GradState::Get().set_enable_grad(false); }
  ~NoGradGuard() { GradState::Get().set_enable_grad(enable_grad_); }

 private:
  bool enable_grad_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_GRAD_STATE_H_
