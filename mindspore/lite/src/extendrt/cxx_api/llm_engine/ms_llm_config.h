/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MS_LLM_CONFIG_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MS_LLM_CONFIG_H_
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/types.h"
#include "include/api/context.h"

namespace mindspore {
namespace llm {
struct KVCacheConfig {
  size_t max_size;        // max key/value cache size
  size_t block_size = 0;  // Size of a cache block in number of tokens;for page attention
  size_t swap_space = 0;  // Size of the CPU swap space per NPU (in GiB).
};

struct GenerateParameters {
  bool do_sample = false;
  int max_new_tokens;  // max number of generated tokens
  int seed;
  // temperature: Float that controls the randomness of the sampling. Lower
  // values make the model more deterministic, while higher values make
  // the model more random. Zero means greedy sampling.
  float temperature = 0;
  // Integer that controls the number of top tokens to consider. Set
  // to -1 to consider all tokens.
  int top_k = -1;
  // Float that controls the cumulative probability of the top tokens
  // to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
  float top_p = 1;
  bool use_beam_search = false;
  // Generate best_of sequences and return the one if the highest token logprobs
  int best_of = -1;
  // 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
  float repetition_penalty = 1.0;
};

struct LLMModelConfig {
  std::vector<std::string> prefill_model_paths;
  std::shared_ptr<mindspore::Context> prefill_context;
  std::string prefill_config_file;
  std::map<std::string, std::string> prefill_model_options;
  std::vector<std::string> decoder_model_paths;
  std::shared_ptr<mindspore::Context> decoder_context;
  std::string decoder_config_file;
  std::map<std::string, std::string> decoder_model_options;
  std::string tokenizer_path;
};

struct MSLLMRequest {
  uint64_t requst_id = 0;
  uint64_t prompt_length = 0;
  uint64_t prompt_cluster_id = 0;
  uint64_t decoder_cluster_id = 0;
};

struct MSLLMEngineStatus {
  uint64_t empty_max_prompt_kv = 0;
};

enum MSLLMRole {
  kLLMRolePrefill = 0,
  kLLMRoleDecoder = 1,
};

}  // namespace llm
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MS_LLM_CONFIG_H_
