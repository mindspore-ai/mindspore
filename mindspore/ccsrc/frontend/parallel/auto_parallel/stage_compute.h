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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_STAGE_COMPUTE_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_STAGE_COMPUTE_H_

#include <tuple>
#include <memory>

#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/parallel_context.h"
#include "mindspore/core/ops/other_ops.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_strategy.h"

namespace mindspore {
namespace parallel {

// Get hyperparams
std::tuple<size_t, size_t> GetSeqLengthAndAttentionHeads(const FuncGraphPtr &root);
std::tuple<size_t, size_t> GetDPAndMP(const std::shared_ptr<Graph> &graph, const size_t stage);
std::tuple<size_t, size_t> GetVocabAndHiddenSize(const FuncGraphPtr &root);
size_t GetNumLayers(const FuncGraphPtr &root);
size_t GetNumMicro(const FuncGraphPtr &root);
size_t GetPerBatch(const FuncGraphPtr &root, size_t seq_l);
size_t GetNumDevices();

class StageComputing {
 private:
  const FuncGraphPtr &root_;
  const std::shared_ptr<Graph> &graph_;
  // Hyperparameters
  const size_t num_devices_ = 0;
  const size_t device_capacity_ = 0;
  const size_t vocab_size_ = 0;
  const size_t seq_length_ = 0;
  const size_t hidden_size_ = 0;
  const size_t attention_heads_ = 0;
  const size_t num_layers_ = 0;
  const size_t expansion_ratio_ = 0;

  const bool parallel_opt_ = 0;
  const bool recompute_ = 0;

  // Parallelism parameters
  size_t dp_dim_ = 0;
  size_t mp_dim_ = 0;
  size_t pp_dim_ = 0;
  size_t per_batch_ = 0;
  size_t num_micros_ = 0;

  std::tuple<size_t, size_t, size_t, size_t, size_t> saved_config_;
  void SaveConfig();
  void LoadConfig();

  size_t NumParametersParsing(size_t l);
  size_t GetStaticMemoryParsing(size_t d, size_t t, size_t p, size_t P);
  size_t GetDynamicMemoryParsing(size_t l, size_t b, size_t m, size_t p, size_t t);

  size_t GetLayerPerStage();
  size_t GetMemory();
  bool fits(size_t memory);

 public:
  StageComputing(const FuncGraphPtr &r, const std::shared_ptr<Graph> &g, size_t device_num, size_t device_capacity,
                 size_t hidden_size, size_t vocab_size, size_t seq_length, size_t head_num, size_t layer_num,
                 size_t expansion_ratio, size_t dp, size_t mp, size_t pp, size_t per_batch, size_t micro,
                 bool parallel_opt, bool recompute);

  size_t GlobalBatchSize();
  size_t CurrentEstimation();
  Status FindSmallerStage();
  size_t LaunchStageCompute();
  void PrintHyperparams();
  void PrintResults(size_t StaticMEM, size_t DynamicMEM, size_t num_param);
  void ParsingException();
  void OOMSuggestion();
  void FittingSuggestion();
};

size_t ParallelSuggestion(const FuncGraphPtr &root, const std::shared_ptr<Graph> &graph);
void ChangeStageNumber(const FuncGraphPtr &root, size_t new_stage_num);

}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_AUTO_PARALLEL_STAGE_COMPUTE_H_
