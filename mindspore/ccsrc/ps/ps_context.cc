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
#if ((defined ENABLE_CPU) && (!defined _WIN32))
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
      MS_LOG(INFO) << "MS_ROLE is " << ms_role;
    }

    worker_num_ = std::strtol(common::GetEnv(kEnvWorkerNum).c_str(), nullptr, kBase);
    server_num_ = std::strtol(common::GetEnv(kEnvPServerNum).c_str(), nullptr, kBase);
    scheduler_host_ = common::GetEnv(kEnvSchedulerHost);
    if (scheduler_host_.length() > kLength) {
      MS_LOG(EXCEPTION) << "The scheduler host's length can not exceed " << kLength;
    }
    scheduler_port_ = std::strtol(common::GetEnv(kEnvSchedulerPort).c_str(), nullptr, kBase);
    if (scheduler_port_ > kMaxPort) {
      MS_LOG(EXCEPTION) << "The port: " << scheduler_port_ << " is illegal.";
    }
    scheduler_manage_port_ =
      static_cast<uint16_t>((std::strtol(common::GetEnv(kEnvSchedulerManagePort).c_str(), nullptr, kBase)));
    if (scheduler_manage_port_ > kMaxPort) {
      MS_LOG(EXCEPTION) << "The port << " << scheduler_manage_port_ << " is illegal.";
    }
    cluster_config_ = std::make_unique<core::ClusterConfig>(worker_num_, server_num_, scheduler_host_, scheduler_port_);
    node_id_ = common::GetEnv(kEnvNodeId);
    if (node_id_.length() > kLength) {
      MS_LOG(EXCEPTION) << "The node id length can not exceed " << kLength;
    }
  } else {
    MS_LOG(INFO) << "PS mode is disabled.";
    is_worker_ = false;
    is_pserver_ = false;
    is_sched_ = false;
  }
}

bool PSContext::is_ps_mode() const {
  if ((server_mode_ == kServerModeFL || server_mode_ == kServerModeHybrid) && ps_enabled_) {
    return true;
  }
  return ps_enabled_;
}

void PSContext::Reset() {
  ps_enabled_ = false;
  is_worker_ = false;
  is_pserver_ = false;
  is_sched_ = false;
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    ps_cache_instance.Finalize();
    set_cache_enable(false);
  }
#endif
}

std::string PSContext::ms_role() const {
  if ((server_mode_ == kServerModeFL || server_mode_ == kServerModeHybrid) && ps_enabled_) {
    return role_;
  }
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

bool PSContext::is_worker() const {
  if ((server_mode_ == kServerModeFL || server_mode_ == kServerModeHybrid) && ps_enabled_) {
    return role_ == kEnvRoleOfWorker;
  }
  return is_worker_;
}

bool PSContext::is_server() const {
  if ((server_mode_ == kServerModeFL || server_mode_ == kServerModeHybrid) && ps_enabled_) {
    return role_ == kEnvRoleOfServer;
  }
  return is_pserver_;
}

bool PSContext::is_scheduler() const {
  if ((server_mode_ == kServerModeFL || server_mode_ == kServerModeHybrid) && ps_enabled_) {
    return role_ == kEnvRoleOfScheduler;
  }
  return is_sched_;
}

uint32_t PSContext::initial_worker_num() const { return worker_num_; }

uint32_t PSContext::initial_server_num() const { return server_num_; }

std::string PSContext::scheduler_host() const { return scheduler_host_; }

void PSContext::SetPSRankId(uint32_t rank_id) { rank_id_ = rank_id; }

uint32_t PSContext::ps_rank_id() const { return rank_id_; }

void PSContext::InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                                    size_t vocab_size) const {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  ps_cache_instance.InsertHashTableSize(param_name, cache_vocab_size, embedding_size, vocab_size);
#endif
}

void PSContext::ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name,
                                      size_t cache_vocab_size, size_t embedding_size) const {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  ps_cache_instance.ReInsertHashTableSize(new_param_name, cur_param_name, cache_vocab_size, embedding_size);
#endif
}

void PSContext::InsertWeightInitInfo(const std::string &param_name, size_t global_seed, size_t op_seed) const {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  ps_cache_instance.InsertWeightInitInfo(param_name, global_seed, op_seed);
#endif
}

void PSContext::InsertAccumuInitInfo(const std::string &param_name, float init_val) const {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  ps_cache_instance.InsertAccumuInitInfo(param_name, init_val);
#endif
}

void PSContext::CloneHashTable(const std::string &dest_param_name, const std::string &src_param_name) const {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  ps_cache_instance.CloneHashTable(dest_param_name, src_param_name);
#endif
}

void PSContext::set_cache_enable(bool cache_enable) const {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  PsDataPrefetch::GetInstance().set_cache_enable(cache_enable);
#endif
}

void PSContext::set_rank_id(uint32_t rank_id) const {
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  ps_cache_instance.set_rank_id(rank_id);
#endif
}

void PSContext::set_server_mode(const std::string &server_mode) {
  if (server_mode != kServerModePS && server_mode != kServerModeFL && server_mode != kServerModeHybrid) {
    MS_LOG(EXCEPTION) << server_mode << " is invalid. Server mode must be " << kServerModePS << " or " << kServerModeFL
                      << " or " << kServerModeHybrid;
    return;
  }
  MS_LOG(INFO) << "Server mode: " << server_mode << " is used for Server and Worker. Scheduler will ignore it.";
  server_mode_ = server_mode;
}

const std::string &PSContext::server_mode() const { return server_mode_; }

void PSContext::set_encrypt_type(const std::string &encrypt_type) {
  if (encrypt_type != kNotEncryptType && encrypt_type != kDPEncryptType && encrypt_type != kPWEncryptType) {
    MS_LOG(EXCEPTION) << encrypt_type << " is invalid. Encrypt type must be " << kNotEncryptType << " or "
                      << kDPEncryptType << " or " << kPWEncryptType;
    return;
  }
  encrypt_type_ = encrypt_type;
}
const std::string &PSContext::encrypt_type() const { return encrypt_type_; }

void PSContext::set_dp_eps(float dp_eps) {
  if (dp_eps > 0) {
    dp_eps_ = dp_eps;
  } else {
    MS_LOG(EXCEPTION) << dp_eps << " is invalid, dp_eps must be larger than 0.";
    return;
  }
}

float PSContext::dp_eps() const { return dp_eps_; }

void PSContext::set_dp_delta(float dp_delta) {
  if (dp_delta > 0 && dp_delta < 1) {
    dp_delta_ = dp_delta;
  } else {
    MS_LOG(EXCEPTION) << dp_delta << " is invalid, dp_delta must be in range of (0, 1).";
    return;
  }
}
float PSContext::dp_delta() const { return dp_delta_; }

void PSContext::set_dp_norm_clip(float dp_norm_clip) {
  if (dp_norm_clip > 0) {
    dp_norm_clip_ = dp_norm_clip;
  } else {
    MS_LOG(EXCEPTION) << dp_norm_clip << " is invalid, dp_norm_clip must be larger than 0.";
    return;
  }
}
float PSContext::dp_norm_clip() const { return dp_norm_clip_; }

void PSContext::set_ms_role(const std::string &role) {
  if (server_mode_ != kServerModeFL && server_mode_ != kServerModeHybrid) {
    MS_LOG(EXCEPTION) << "Only federated learning supports to set role by fl context.";
    return;
  }
  if (role != kEnvRoleOfWorker && role != kEnvRoleOfServer && role != kEnvRoleOfScheduler) {
    MS_LOG(EXCEPTION) << "ms_role " << role << " is invalid.";
    return;
  }
  role_ = role;
}

void PSContext::set_worker_num(uint32_t worker_num) {
  // Hybrid training mode only supports one worker for now.
  if (server_mode_ == kServerModeHybrid && worker_num != 1) {
    MS_LOG(EXCEPTION) << "The worker number should be set to 1 in hybrid training mode.";
    return;
  }
  worker_num_ = worker_num;
}
uint32_t PSContext::worker_num() const { return worker_num_; }

void PSContext::set_server_num(uint32_t server_num) {
  if (server_num == 0) {
    MS_LOG(EXCEPTION) << "Server number must be greater than 0.";
    return;
  }
  server_num_ = server_num;
}
uint32_t PSContext::server_num() const { return server_num_; }

void PSContext::set_scheduler_ip(const std::string &sched_ip) { scheduler_host_ = sched_ip; }

std::string PSContext::scheduler_ip() const { return scheduler_host_; }

void PSContext::set_scheduler_port(uint16_t sched_port) { scheduler_port_ = sched_port; }

uint16_t PSContext::scheduler_port() const { return scheduler_port_; }

void PSContext::GenerateResetterRound() {
  uint32_t binary_server_context = 0;
  bool is_parameter_server_mode = false;
  bool is_federated_learning_mode = false;
  bool is_mixed_training_mode = false;
  bool use_pairwise_encrypt = (encrypt_type_ == kPWEncryptType);

  if (server_mode_ == kServerModePS) {
    is_parameter_server_mode = true;
  } else if (server_mode_ == kServerModeFL) {
    is_federated_learning_mode = true;
  } else if (server_mode_ == kServerModeHybrid) {
    is_mixed_training_mode = true;
  } else {
    MS_LOG(EXCEPTION) << server_mode_ << " is invalid. Server mode must be " << kServerModePS << " or " << kServerModeFL
                      << " or " << kServerModeHybrid;
    return;
  }

  binary_server_context = ((unsigned int)is_parameter_server_mode) | ((unsigned int)is_federated_learning_mode << 1) |
                          ((unsigned int)is_mixed_training_mode << 2) | ((unsigned int)use_pairwise_encrypt << 3);
  if (kServerContextToResetRoundMap.count(binary_server_context) == 0) {
    resetter_round_ = ResetterRound::kNoNeedToReset;
  } else {
    resetter_round_ = kServerContextToResetRoundMap.at(binary_server_context);
  }
  MS_LOG(INFO) << "Server context is " << binary_server_context << ". Resetter round is " << resetter_round_;
  return;
}

ResetterRound PSContext::resetter_round() const { return resetter_round_; }

void PSContext::set_fl_server_port(uint16_t fl_server_port) { fl_server_port_ = fl_server_port; }

uint16_t PSContext::fl_server_port() const { return fl_server_port_; }

void PSContext::set_fl_client_enable(bool enabled) { fl_client_enable_ = enabled; }

bool PSContext::fl_client_enable() const { return fl_client_enable_; }

void PSContext::set_start_fl_job_threshold(uint64_t start_fl_job_threshold) {
  start_fl_job_threshold_ = start_fl_job_threshold;
}

uint64_t PSContext::start_fl_job_threshold() const { return start_fl_job_threshold_; }

void PSContext::set_start_fl_job_time_window(uint64_t start_fl_job_time_window) {
  start_fl_job_time_window_ = start_fl_job_time_window;
}

uint64_t PSContext::start_fl_job_time_window() const { return start_fl_job_time_window_; }

void PSContext::set_update_model_ratio(float update_model_ratio) {
  if (update_model_ratio > 1.0) {
    MS_LOG(EXCEPTION) << "update_model_ratio must be between 0 and 1.";
    return;
  }
  update_model_ratio_ = update_model_ratio;
}

float PSContext::update_model_ratio() const { return update_model_ratio_; }

void PSContext::set_update_model_time_window(uint64_t update_model_time_window) {
  update_model_time_window_ = update_model_time_window;
}

uint64_t PSContext::update_model_time_window() const { return update_model_time_window_; }

void PSContext::set_share_secrets_ratio(float share_secrets_ratio) {
  if (share_secrets_ratio > 0 && share_secrets_ratio <= 1) {
    share_secrets_ratio_ = share_secrets_ratio;
  } else {
    MS_LOG(EXCEPTION) << share_secrets_ratio << " is invalid, share_secrets_ratio must be in range of (0, 1].";
    return;
  }
}

float PSContext::share_secrets_ratio() const { return share_secrets_ratio_; }

void PSContext::set_cipher_time_window(uint64_t cipher_time_window) {
  if (cipher_time_window_ < 0) {
    MS_LOG(EXCEPTION) << "cipher_time_window should not be less than 0.";
    return;
  }
  cipher_time_window_ = cipher_time_window;
}

uint64_t PSContext::cipher_time_window() const { return cipher_time_window_; }

void PSContext::set_reconstruct_secrets_threshold(uint64_t reconstruct_secrets_threshold) {
  if (reconstruct_secrets_threshold == 0) {
    MS_LOG(EXCEPTION) << "reconstruct_secrets_threshold should be positive.";
    return;
  }
  reconstruct_secrets_threshold_ = reconstruct_secrets_threshold;
}

uint64_t PSContext::reconstruct_secrets_threshold() const { return reconstruct_secrets_threshold_; }

void PSContext::set_fl_name(const std::string &fl_name) { fl_name_ = fl_name; }

const std::string &PSContext::fl_name() const { return fl_name_; }

void PSContext::set_fl_iteration_num(uint64_t fl_iteration_num) { fl_iteration_num_ = fl_iteration_num; }

uint64_t PSContext::fl_iteration_num() const { return fl_iteration_num_; }

void PSContext::set_client_epoch_num(uint64_t client_epoch_num) { client_epoch_num_ = client_epoch_num; }

uint64_t PSContext::client_epoch_num() const { return client_epoch_num_; }

void PSContext::set_client_batch_size(uint64_t client_batch_size) { client_batch_size_ = client_batch_size; }

uint64_t PSContext::client_batch_size() const { return client_batch_size_; }

void PSContext::set_client_learning_rate(float client_learning_rate) { client_learning_rate_ = client_learning_rate; }

float PSContext::client_learning_rate() const { return client_learning_rate_; }

void PSContext::set_worker_step_num_per_iteration(uint64_t worker_step_num_per_iteration) {
  worker_step_num_per_iteration_ = worker_step_num_per_iteration;
}

uint64_t PSContext::worker_step_num_per_iteration() const { return worker_step_num_per_iteration_; }

bool PSContext::enable_ssl() const { return enable_ssl_; }

void PSContext::set_enable_ssl(bool enabled) { enable_ssl_ = enabled; }

core::ClusterConfig &PSContext::cluster_config() {
  if (cluster_config_ == nullptr) {
    MS_LOG(EXCEPTION) << "The cluster config is empty.";
  }
  return *cluster_config_;
}

void PSContext::set_scheduler_manage_port(uint16_t sched_port) { scheduler_manage_port_ = sched_port; }

uint16_t PSContext::scheduler_manage_port() const { return scheduler_manage_port_; }

void PSContext::set_config_file_path(const std::string &path) { config_file_path_ = path; }

std::string PSContext::config_file_path() const { return config_file_path_; }

void PSContext::set_node_id(const std::string &node_id) { node_id_ = node_id; }

const std::string &PSContext::node_id() const { return node_id_; }

std::string PSContext::client_password() const { return client_password_; }
void PSContext::set_client_password(const std::string &password) { client_password_ = password; }

std::string PSContext::server_password() const { return server_password_; }
void PSContext::set_server_password(const std::string &password) { server_password_ = password; }
}  // namespace ps
}  // namespace mindspore
