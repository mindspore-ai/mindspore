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

#ifndef MINDSPORE_CCSRC_PS_CONTEXT_H_
#define MINDSPORE_CCSRC_PS_CONTEXT_H_

#include <map>
#include <string>
#include <memory>
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace ps {
constexpr char kServerModePS[] = "PARAMETER_SERVER";
constexpr char kEnvRole[] = "MS_ROLE";
constexpr char kEnvRoleOfPServer[] = "MS_PSERVER";
constexpr char kEnvRoleOfServer[] = "MS_SERVER";
constexpr char kEnvRoleOfWorker[] = "MS_WORKER";
constexpr char kEnvRoleOfScheduler[] = "MS_SCHED";
constexpr char kEnvRoleOfNotPS[] = "MS_NOT_PS";
constexpr size_t kMaxPasswordLen = 1024;

namespace core {
class ClusterConfig;
}  // namespace core

class BACKEND_EXPORT PSContext {
 public:
  ~PSContext();
  PSContext(PSContext const &) = delete;
  PSContext &operator=(const PSContext &) = delete;
  static std::shared_ptr<PSContext> instance();

  void SetPSEnable(bool enabled);
  bool is_ps_mode() const;
  void Reset();
  std::string ms_role() const;
  bool is_worker() const;
  bool is_server() const;
  bool is_scheduler() const;
  uint32_t initial_worker_num() const;
  uint32_t initial_server_num() const;
  std::string scheduler_host() const;
  void SetPSRankId(uint32_t rank_id);
  uint32_t ps_rank_id() const;
  void InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                           size_t vocab_size, int32_t param_key) const;
  void ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name,
                             size_t cache_vocab_size, size_t embedding_size) const;
  void InsertAccumuInitInfo(const std::string &param_name, float init_val) const;
  void CloneHashTable(const std::string &dest_param_name, int32_t dest_param_key, const std::string &src_param_name,
                      int32_t src_param_key) const;
  void set_cache_enable(bool cache_enable) const;
  bool cache_enable() const;

  // Set embedding cache size for  ps cache mode.
  void set_cache_size(size_t cache_size) const;

  // Set if the storage format of embedding table is sparse or not.
  void set_sparse_format(bool is_sparse);

  void set_rank_id(uint32_t rank_id) const;

  // In new server framework, process role, worker number, server number, scheduler ip and scheduler port should be set
  // by ps_context.
  void set_server_mode(const std::string &server_mode);
  const std::string &server_mode() const;

  void set_ms_role(const std::string &role);

  void set_worker_num(uint32_t worker_num);
  uint32_t worker_num() const;

  void set_server_num(uint32_t server_num);
  uint32_t server_num() const;

  void set_scheduler_ip(const std::string &sched_ip);
  std::string scheduler_ip() const;

  void set_scheduler_port(uint16_t sched_port);
  uint16_t scheduler_port() const;

  core::ClusterConfig &cluster_config();

  void set_scheduler_manage_port(uint16_t sched_port);
  uint16_t scheduler_manage_port() const;

  void set_config_file_path(const std::string &path);
  std::string config_file_path() const;

  void set_node_id(const std::string &node_id);
  const std::string &node_id() const;

  bool enable_ssl() const;
  void set_enable_ssl(bool enabled);

  char *client_password();
  void set_client_password(const char *password);
  void ClearClientPassword();

  char *server_password();
  void set_server_password(const char *password);
  void ClearServerPassword();

  std::string http_url_prefix() const;

  void set_instance_name(const std::string &instance_name);
  const std::string &instance_name() const;

  // Whether distributed MindRT is enabled.
  bool enable_distributed_mindrt() const;

 private:
  PSContext();

  bool ps_enabled_;
  bool is_worker_;
  bool is_pserver_;
  bool is_sched_;
  uint32_t rank_id_;
  uint32_t worker_num_;
  uint32_t server_num_;
  std::string scheduler_host_;
  uint16_t scheduler_port_;

  // The server process's role.
  std::string role_;

  // Server mode which could be Parameter Server.
  std::string server_mode_;

  // The cluster config read through environment variables, the value does not change.
  std::unique_ptr<core::ClusterConfig> cluster_config_;

  // The port used by scheduler to receive http requests for scale out or scale in.
  uint16_t scheduler_manage_port_;

  // The path of the configuration file, used to configure the certification path and persistent storage type, etc.
  std::string config_file_path_;

  // Unique id of the node
  std::string node_id_;

  // Whether to enable ssl for network communication.
  bool enable_ssl_;
  // Password used to decode p12 file.
  char client_password_[kMaxPasswordLen];
  // Password used to decode p12 file.
  char server_password_[kMaxPasswordLen];
  // http url prefix for http communication
  std::string http_url_prefix_;
  // The name of instance
  std::string instance_name_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CONTEXT_H_
