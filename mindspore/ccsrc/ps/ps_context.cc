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

#include "ps/ps_context.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace ps {
std::shared_ptr<PSContext> PSContext::instance() {
  static std::shared_ptr<PSContext> ps_instance = nullptr;
  if (ps_instance == nullptr) {
    ps_instance.reset(new (std::nothrow) PSContext());
  }
  return ps_instance;
}

void PSContext::SetPSEnable(bool enabled) {
  ps_enabled_ = enabled;
  if (ps_enabled_) {
    std::string ms_role = common::GetEnv(kEnvRole);
    MS_LOG(INFO) << "PS mode is enabled. MS_ROLE is " << ms_role;
    if (ms_role == kEnvRoleOfWorker) {
      is_worker_ = true;
    } else if (ms_role == kEnvRoleOfPServer) {
      is_pserver_ = true;
    } else if (ms_role == kEnvRoleOfScheduler) {
      is_sched_ = true;
    } else {
      MS_LOG(WARNING) << "MS_ROLE is " << ms_role << ", which is invalid.";
    }
  } else {
    MS_LOG(INFO) << "PS mode is disabled.";
    is_worker_ = false;
    is_pserver_ = false;
    is_sched_ = false;
  }
}

bool PSContext::is_ps_enabled() const { return ps_enabled_; }

void PSContext::Reset() {
  ps_enabled_ = false;
  is_worker_ = false;
  is_pserver_ = false;
  is_sched_ = false;
}

std::string PSContext::ms_role() const {
  if (is_worker_) {
    return kEnvRoleOfWorker;
  } else if (is_pserver_) {
    return kEnvRoleOfPServer;
  } else if (is_sched_) {
    return kEnvRoleOfScheduler;
  } else {
    return kEnvRoleOfNotPS;
  }
}

bool PSContext::is_role_worker() const { return is_worker_; }

bool PSContext::is_role_pserver() const { return is_pserver_; }

bool PSContext::is_role_sched() const { return is_sched_; }

void PSContext::SetPSRankId(int rank_id) { rank_id_ = rank_id; }

int PSContext::ps_rank_id() const { return rank_id_; }
}  // namespace ps
}  // namespace mindspore
