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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MS_LLM_ENGINE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MS_LLM_ENGINE_H_
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/types.h"
#include "include/api/status.h"
#include "extendrt/cxx_api/llm_engine/ms_llm_config.h"
#include "extendrt/cxx_api/model/model_impl.h"
#include "extendrt/cxx_api/model/model_group_impl.h"

namespace mindspore {
namespace llm {
class MS_API MSLLMEngine {
 public:
  MSLLMEngine() {}
  ~MSLLMEngine() = default;
  Status Init(const LLMModelConfig *model_config, KVCacheConfig *kvcache_config, uint64_t cluster_id,
              const std::map<std::string, std::string> &options, GenerateParameters *default_param);

  Status ResizeKVCache(const KVCacheConfig &config);
  Status SetMaxSeqLen(int64_t max_seq_len);
  Status CopyKVCacheToHost(int64_t batch_index, const std::vector<MSTensor> &key_tensor,
                           const std::vector<MSTensor> &value_tensor);
  Status CopyKVCacheToDevice(int64_t batch_index, const std::vector<MSTensor> &key_tensor,
                             const std::vector<MSTensor> &value_tensor);

  Status Prefill(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 GenerateParameters *param = nullptr);

  std::vector<MSTensor> Prefill(const std::vector<MSTensor> &inputs, GenerateParameters *param = nullptr);

  Status Decode(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                GenerateParameters *param = nullptr);

  std::vector<MSTensor> Decode(const std::vector<MSTensor> &inputs, GenerateParameters *param = nullptr);

  std::vector<MSTensor> DecodeAsync(const MSLLMRequest &req, const std::vector<MSTensor> &inputs,
                                    GenerateParameters *param = nullptr);

  std::vector<MSTensor> PrefillAsync(const MSLLMRequest &req, const std::vector<MSTensor> &inputs,
                                     GenerateParameters *param = nullptr);

 private:
  int64_t num_heads_;
  int64_t num_n_kv_heads_;
  int64_t hidden_size_;
  int64_t head_size_;
  int64_t num_attention_heads_;
  int64_t num_layers_;
  int64_t max_seq_len_;

  std::shared_ptr<mindspore::ModelImpl> prefill_model_;
  std::shared_ptr<mindspore::ModelImpl> decoder_model_;
  std::shared_ptr<mindspore::ModelImpl> post_process_model_;
  std::shared_ptr<mindspore::ModelGroupImpl> model_group_;
  GenerateParameters *default_param_;
};
}  // namespace llm
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MS_LLM_ENGINE_H_
