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
#include "backend/kernel_compiler/kernel.h"
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "ps/ps_cache/ps_cache_manager.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#endif

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

    worker_num_ = std::strtol(common::GetEnv(kEnvWorkerNum).c_str(), nullptr, 10);
    server_num_ = std::strtol(common::GetEnv(kEnvPServerNum).c_str(), nullptr, 10);
    scheduler_host_ = common::GetEnv(kEnvSchedulerHost);
    scheduler_port_ = std::strtol(common::GetEnv(kEnvSchedulerPort).c_str(), nullptr, 10);
    core::ClusterMetadata::instance()->Init(worker_num_, server_num_, scheduler_host_, scheduler_port_);
  } else {
    MS_LOG(INFO) << "PS mode is disabled.";
    is_worker_ = false;
    is_pserver_ = false;
    is_sched_ = false;
  }
}

bool PSContext::is_ps_mode() const { return ps_enabled_; }

void PSContext::Reset() {
  ps_enabled_ = false;
  is_worker_ = false;
  is_pserver_ = false;
  is_sched_ = false;
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    ps_cache_instance.Finalize();
    set_cache_enable(false);
  }
#endif
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

bool PSContext::is_worker() const { return is_worker_; }

bool PSContext::is_server() const { return is_pserver_; }

bool PSContext::is_scheduler() const { return is_sched_; }

uint32_t PSContext::initial_worker_num() { return worker_num_; }

uint32_t PSContext::initial_server_num() { return server_num_; }

std::string PSContext::scheduler_host() { return scheduler_host_; }

uint16_t PSContext::scheduler_port() { return scheduler_port_; }

void PSContext::SetPSRankId(int rank_id) { rank_id_ = rank_id; }

int PSContext::ps_rank_id() const { return rank_id_; }

void PSContext::InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                                    size_t vocab_size) const {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  ps_cache_instance.InsertHashTableSize(param_name, cache_vocab_size, embedding_size, vocab_size);
#endif
}

void PSContext::ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name,
                                      size_t cache_vocab_size, size_t embedding_size) const {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  ps_cache_instance.ReInsertHashTableSize(new_param_name, cur_param_name, cache_vocab_size, embedding_size);
#endif
}

void PSContext::InsertWeightInitInfo(const std::string &param_name, size_t global_seed, size_t op_seed) const {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  ps_cache_instance.InsertWeightInitInfo(param_name, global_seed, op_seed);
#endif
}

void PSContext::InsertAccumuInitInfo(const std::string &param_name, float init_val) const {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  ps_cache_instance.InsertAccumuInitInfo(param_name, init_val);
#endif
}

void PSContext::CloneHashTable(const std::string &dest_param_name, const std::string &src_param_name) const {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  ps_cache_instance.CloneHashTable(dest_param_name, src_param_name);
#endif
}

void PSContext::set_cache_enable(bool cache_enable) const {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  PsDataPrefetch::GetInstance().set_cache_enable(cache_enable);
#endif
}

void PSContext::set_rank_id(int rank_id) const {
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  ps_cache_instance.set_rank_id(rank_id);
#endif
}
}  // namespace ps
}  // namespace mindspore
