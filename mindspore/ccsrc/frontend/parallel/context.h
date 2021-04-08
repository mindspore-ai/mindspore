/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_CONTEXT_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_CONTEXT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/status.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/convert_utils.h"
#include "utils/info.h"
#include "pipeline/jit/pipeline.h"

namespace mindspore {
namespace parallel {
constexpr char STAND_ALONE[] = "stand_alone";
constexpr char DATA_PARALLEL[] = "data_parallel";
constexpr char HYBRID_PARALLEL[] = "hybrid_parallel";
constexpr char AUTO_PARALLEL[] = "auto_parallel";
constexpr char SEMI_AUTO_PARALLEL[] = "semi_auto_parallel";

constexpr char DYNAMIC_PROGRAMMING[] = "dynamic_programming";
constexpr char RECURSIVE_PROGRAMMING[] = "recursive_programming";

constexpr char TRAINING[] = "training";
constexpr char ACCUMULATION[] = "accumulation";

constexpr char ALL_GROUP_PARALLEL[] = "all_group_parallel";
constexpr char SAME_SERVER_GROUP_PARALLEL[] = "same_server_group_parallel";
constexpr char NO_GROUP_PARALLEL[] = "no_group_parallel";

class ParallelContext {
 public:
  ~ParallelContext() = default;
  ParallelContext(const ParallelContext &) = delete;
  ParallelContext &operator=(const ParallelContext &) = delete;

  static std::shared_ptr<ParallelContext> GetInstance();

  void set_gradients_mean(bool gradients_mean);
  bool gradients_mean() const { return gradients_mean_; }

  void set_full_batch(bool full_batch);
  bool full_batch() const { return full_batch_; }

  void set_gradient_fp32_sync(bool gradient_fp32_sync);
  bool gradient_fp32_sync() const { return gradient_fp32_sync_; }

  void set_loss_repeated_mean(bool loss_repeated_mean);
  bool loss_repeated_mean() const { return loss_repeated_mean_; }

  void set_device_num(int64_t device_num);
  int64_t device_num() const { return device_num_; }

  void set_pipeline_stage_split_num(const int64_t stages);
  int64_t pipeline_stage_split_num() const { return pipeline_stage_split_num_; }

  void set_global_rank(int64_t global_rank);
  int64_t global_rank() const { return global_rank_; }

  void set_grad_accumulation_step(int64_t grad_accumulation_step);
  int64_t grad_accumulation_step() const { return grad_accumulation_step_; }

  bool set_parallel_mode(const std::string &parallel_mode);
  std::string parallel_mode() const { return parallel_mode_; }

  bool set_strategy_search_mode(const std::string &strategy_search_mode);
  std::string strategy_search_mode() const { return strategy_search_mode_; }

  void set_parameter_broadcast(bool parameter_broadcast);
  bool parameter_broadcast() const { return parameter_broadcast_; }

  bool device_num_is_set() const { return device_num_is_set_; }
  bool global_rank_is_set() const { return global_rank_is_set_; }
  bool parameter_broadcast_is_set() const { return parameter_broadcast_is_set_; }

  void SetAllReduceFusionSplitIndices(const std::vector<uint32_t> indices, const std::string &group);
  const std::vector<uint32_t> GetAllReduceFusionSplitIndices(const std::string &group) const;
  void SetAllReduceFusionSplitSizes(const std::vector<uint32_t> sizes, const std::string &group);
  const std::vector<uint32_t> GetAllReduceFusionSplitSizes(const std::string &group) const;
  void set_enable_all_reduce_fusion(bool enable_all_reduce_fusion) {
    enable_all_reduce_fusion_ = enable_all_reduce_fusion;
  }
  bool enable_all_reduce_fusion() const { return enable_all_reduce_fusion_; }

  void set_strategy_ckpt_load_file(const std::string &strategy_ckpt_load_file);
  std::string strategy_ckpt_load_file() const { return strategy_ckpt_load_file_; }
  void set_strategy_ckpt_save_file(const std::string &strategy_ckpt_save_file);
  std::string strategy_ckpt_save_file() const { return strategy_ckpt_save_file_; }
  void set_group_ckpt_save_file(const std::string &group_ckpt_save_file);
  std::string group_ckpt_save_file() const { return group_ckpt_save_file_; }

  void set_enable_parallel_optimizer(bool enable_parallel_optimizer) {
    enable_parallel_optimizer_ = enable_parallel_optimizer;
  }
  bool enable_parallel_optimizer() const { return enable_parallel_optimizer_; }

  bool set_communi_parallel_mode(const std::string &communi_parallel_mode);
  std::string communi_parallel_mode() const { return communi_parallel_mode_; }

  void Reset();
  void ParallelParameterContextInitShape(const FuncGraphPtr &func_graph);
  void ParallelParameterContextRestoreShape(const FuncGraphPtr &func_graph, const ParameterPtr &param_node,
                                            AbstractBasePtr ptr);
  void ParallelParameterContextCkptShape(const FuncGraphPtr &func_graph, const ParameterPtr &param_node,
                                         const AbstractBasePtr &ptr);

 private:
  ParallelContext();
  static std::shared_ptr<ParallelContext> inst_context_;
  bool gradients_mean_;
  bool full_batch_;
  bool gradient_fp32_sync_;
  bool loss_repeated_mean_;
  int64_t device_num_;
  int64_t global_rank_;
  int64_t grad_accumulation_step_;
  std::string parallel_mode_;
  std::string strategy_search_mode_;
  int64_t pipeline_stage_split_num_;
  bool parameter_broadcast_;
  bool device_num_is_set_;
  bool global_rank_is_set_;
  bool parameter_broadcast_is_set_;
  bool enable_all_reduce_fusion_;
  std::map<std::string, std::vector<uint32_t>> all_reduce_fusion_split_indices_;
  std::map<std::string, std::vector<uint32_t>> all_reduce_fusion_split_sizes_;
  std::string strategy_ckpt_load_file_;
  std::string strategy_ckpt_save_file_;
  std::string group_ckpt_save_file_;
  bool enable_parallel_optimizer_;
  bool init_param_shape_;
  std::string communi_parallel_mode_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_CONTEXT_H_
