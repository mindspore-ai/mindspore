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

#include <string>
#include <memory>
#include "ps/constants.h"
#include "ps/core/cluster_metadata.h"

namespace mindspore {
namespace ps {
constexpr char kEnvRole[] = "MS_ROLE";
constexpr char kEnvRoleOfPServer[] = "MS_PSERVER";
constexpr char kEnvRoleOfWorker[] = "MS_WORKER";
constexpr char kEnvRoleOfScheduler[] = "MS_SCHED";
constexpr char kEnvRoleOfNotPS[] = "MS_NOT_PS";

class PSContext {
 public:
  ~PSContext() = default;
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
  uint32_t initial_worker_num();
  uint32_t initial_server_num();
  std::string scheduler_host();
  uint16_t scheduler_port();
  void SetPSRankId(int rank_id);
  int ps_rank_id() const;
  void InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                           size_t vocab_size) const;
  void ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name,
                             size_t cache_vocab_size, size_t embedding_size) const;
  void InsertWeightInitInfo(const std::string &param_name, size_t global_seed, size_t op_seed) const;
  void InsertAccumuInitInfo(const std::string &param_name, float init_val) const;
  void CloneHashTable(const std::string &dest_param_name, const std::string &src_param_name) const;
  void set_cache_enable(bool cache_enable) const;
  void set_rank_id(int rank_id) const;

 private:
  PSContext()
      : ps_enabled_(false),
        is_worker_(false),
        is_pserver_(false),
        is_sched_(false),
        rank_id_(-1),
        worker_num_(0),
        server_num_(0),
        scheduler_host_(""),
        scheduler_port_(0) {}
  bool ps_enabled_;
  bool is_worker_;
  bool is_pserver_;
  bool is_sched_;
  int rank_id_;
  uint32_t worker_num_;
  uint32_t server_num_;
  std::string scheduler_host_;
  uint16_t scheduler_port_;
};
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CONTEXT_H_
