/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PARALLEL_CONTEXT_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PARALLEL_CONTEXT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/common/utils/convert_utils.h"
#include "utils/info.h"
#include "include/common/visible.h"

namespace mindspore::parallel {
constexpr char kStandalone[] = "stand_alone";
constexpr char kDataParallel[] = "data_parallel";
constexpr char kHybridParallel[] = "hybrid_parallel";
constexpr char kAutoParallel[] = "auto_parallel";
constexpr char kSemiAutoParallel[] = "semi_auto_parallel";

constexpr char kDynamicProgramming[] = "dynamic_programming";
constexpr char kRecursiveProgramming[] = "recursive_programming";
constexpr char kShardingPropagation[] = "sharding_propagation";

constexpr char kAccumulation[] = "accumulation";

constexpr char kAllGroupParallel[] = "all_group_parallel";
constexpr char kSameServerGroupParallel[] = "same_server_group_parallel";
constexpr char kNoGroupParallel[] = "no_group_parallel";

constexpr char kPynativeShard[] = "pynative_shard";
constexpr char kSkipAutoParallelCompile[] = "skip_auto_parallel_compile";
constexpr char kKeepInputUnchanged[] = "keep_input_unchanged";

constexpr char kFusionAuto[] = "auto";
constexpr char kFusionSize[] = "size";
constexpr char kFusionIndex[] = "index";
constexpr int64_t kFusionThreshold = 64;
constexpr int64_t kDataParallelFusionThreshold = -1;
constexpr char kRelatedFusionKey[] = "related_fusion_key";
constexpr char kRelatedNodeId[] = "related_node_id";
constexpr char FIRST_RECEIVE[] = "first_receive";
constexpr char kRelatedCommNodeId[] = "related_comm_node_id";

class COMMON_EXPORT ParallelContext {
 public:
  static std::shared_ptr<ParallelContext> GetInstance();
  ~ParallelContext() = default;
  ParallelContext(const ParallelContext &) = delete;
  ParallelContext &operator=(const ParallelContext &) = delete;

  void set_gradients_mean(bool gradients_mean);
  bool gradients_mean() const { return gradients_mean_; }

  void set_full_batch(bool full_batch);
  bool full_batch() const { return full_batch_; }

  void set_dataset_strategy(const std::vector<std::vector<int64_t>> &dataset_strategy);
  std::vector<std::vector<int64_t>> dataset_strategy() const { return dataset_strategy_; }

  void set_gradient_fp32_sync(bool gradient_fp32_sync);
  bool gradient_fp32_sync() const { return gradient_fp32_sync_; }

  void set_loss_repeated_mean(bool loss_repeated_mean);
  bool loss_repeated_mean() const { return loss_repeated_mean_; }

  void set_device_num(int64_t device_num);
  int64_t device_num() const { return device_num_; }

  void set_fusion_threshold_mb(int64_t fusion_threshold);
  int64_t fusion_threshold_mb() const { return fusion_threshold_mb_; }

  int64_t dp_fusion_threshold_mb() const { return dp_fusion_threshold_mb_; }

  void set_allgather_fusion_threshold_mb(int64_t fusion_threshold);
  int64_t allgather_fusion_threshold_mb() const { return allgather_fusion_threshold_mb_; }

  void set_reducescatter_fusion_threshold_mb(int64_t fusion_threshold);
  int64_t reducescatter_fusion_threshold_mb() const { return reducescatter_fusion_threshold_mb_; }

  bool set_fusion_mode(const std::string &fusion_mode);
  std::string get_fusion_mode() const { return fusion_mode_; }

  void set_pipeline_stage_split_num(const int64_t stage_num);
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
  bool full_batch_is_set() const { return full_batch_is_set_; }

  void set_optimizer_weight_shard_size(int64_t optimizer_weight_shard_size);
  int64_t optimizer_weight_shard_size() const { return optimizer_weight_shard_size_; }
  void set_optimizer_weight_shard_aggregated_save(bool optimizer_weight_shard_aggregated_save);
  bool optimizer_weight_shard_aggregated_save() const { return optimizer_weight_shard_aggregated_save_; }

  void SetAllReduceFusionSplitIndices(const std::vector<uint32_t> &indices, const std::string &group);
  std::vector<uint32_t> GetAllReduceFusionSplitIndices(const std::string &group) const;
  void SetAllReduceFusionSplitSizes(const std::vector<uint32_t> &sizes, const std::string &group);
  std::vector<uint32_t> GetAllReduceFusionSplitSizes(const std::string &group) const;
  void set_enable_all_reduce_fusion(bool enable_all_reduce_fusion) {
    enable_all_reduce_fusion_ = enable_all_reduce_fusion;
  }
  bool enable_all_reduce_fusion() const { return enable_all_reduce_fusion_; }
  void set_enable_all_gather_fusion(bool enable_all_gather_fusion) {
    enable_all_gather_fusion_ = enable_all_gather_fusion;
  }
  bool enable_all_gather_fusion() const { return enable_all_gather_fusion_; }
  void set_enable_reduce_scatter_fusion(bool enable_reduce_scatter_fusion) {
    enable_reduce_scatter_fusion_ = enable_reduce_scatter_fusion;
  }
  bool enable_reduce_scatter_fusion() const { return enable_reduce_scatter_fusion_; }

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

  void set_hccl_test_available(bool hccl_test_available) { hccl_test_available_ = hccl_test_available; }
  bool hccl_test_available() const { return hccl_test_available_; }
  void set_grad_accumulation_shard(const bool grad_accumulation_shard) {
    grad_accumulation_shard_ = grad_accumulation_shard;
  }
  bool grad_accumulation_shard() const { return grad_accumulation_shard_; }
  void set_parallel_optimizer_threshold(const int64_t parallel_optimizer_threshold) {
    parallel_optimizer_threshold_ = parallel_optimizer_threshold;
  }
  int64_t get_parallel_optimizer_threshold() const { return parallel_optimizer_threshold_; }

  bool set_communi_parallel_mode(const std::string &communi_parallel_mode);
  std::string communi_parallel_mode() const { return communi_parallel_mode_; }
  void set_enable_all2all(const bool enable);
  bool enable_all2all() const { return enable_all2all_; }
  void set_dataset_repeat_dim_right(const bool dataset_repeat_dim_right) {
    dataset_repeat_dim_right_ = dataset_repeat_dim_right;
  }
  bool dataset_repeat_dim_right() const { return dataset_repeat_dim_right_; }

  void Reset();
  void ParallelParameterContextRestoreShape(const FuncGraphPtr &func_graph, const ParameterPtr &param_node,
                                            const AbstractBasePtr &ptr) const;
  void set_sharding_propagation(const bool stra_pto);
  bool sharding_propagation() const { return sharding_propagation_; }

  void set_enable_micro_interleaved(const bool);
  bool enable_micro_interleaved() const { return enable_micro_interleaved_; }

  void set_pipeline_micro_size(const size_t);
  size_t pipeline_micro_size() const { return pipeline_micro_size_; }

  void set_do_transform(const bool);
  bool do_transform() const { return do_transform_; }

  void set_stra_file_only_trainable_params(const bool);
  bool stra_file_only_trainable_params() const { return stra_file_only_trainable_params_; }

 private:
  ParallelContext();
  bool ParallelContextCareGraph(const FuncGraphPtr &func_graph) const;

  bool gradients_mean_;
  bool full_batch_;
  bool full_batch_is_set_;
  bool gradient_fp32_sync_;
  bool loss_repeated_mean_;
  int64_t device_num_;
  int64_t dp_fusion_threshold_mb_;
  int64_t fusion_threshold_mb_;
  int64_t allgather_fusion_threshold_mb_;
  int64_t reducescatter_fusion_threshold_mb_;  // reducescatter
  int64_t global_rank_;
  int64_t grad_accumulation_step_;
  std::string parallel_mode_;
  std::string strategy_search_mode_;
  int64_t pipeline_stage_split_num_;
  size_t pipeline_micro_size_;
  bool parameter_broadcast_;
  bool device_num_is_set_;
  bool fusion_threshold_is_set_;
  bool global_rank_is_set_;
  bool parameter_broadcast_is_set_;
  bool enable_all_reduce_fusion_;
  bool enable_all_gather_fusion_;
  bool enable_reduce_scatter_fusion_;
  std::map<std::string, std::vector<uint32_t>> all_reduce_fusion_split_indices_;
  std::map<std::string, std::vector<uint32_t>> all_reduce_fusion_split_sizes_;
  std::string strategy_ckpt_load_file_;
  std::string strategy_ckpt_save_file_;
  std::string group_ckpt_save_file_;
  bool enable_parallel_optimizer_;
  std::string communi_parallel_mode_;
  int64_t optimizer_weight_shard_size_;
  bool optimizer_weight_shard_aggregated_save_;
  bool grad_accumulation_shard_;
  int64_t parallel_optimizer_threshold_;
  // Enable AllToAll or not. If false, use AllGather and Split.
  bool enable_all2all_;
  std::vector<std::vector<int64_t>> dataset_strategy_;
  bool dataset_repeat_dim_right_ = false;
  bool hccl_test_available_ = false;
  bool sharding_propagation_;
  bool enable_micro_interleaved_ = false;
  bool do_transform_ = false;
  bool stra_file_only_trainable_params_ = true;
  std::string fusion_mode_;
};
}  // namespace mindspore::parallel
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PARALLEL_CONTEXT_H_
