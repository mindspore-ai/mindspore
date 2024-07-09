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

#include "utils/llm_manager.h"

#include "utils/ms_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
LLMManager &LLMManager::GetInstance() noexcept {
  static LLMManager instance;
  return instance;
}

LLMManager::LLMManager() { init(); }

void LLMManager::init() {
  if (inited_) {
    return;
  }

  auto llm_seq_length_idx_env = common::GetEnv("MS_LLM_SEQ_LENGTH_INDEX");
  if (!llm_seq_length_idx_env.empty()) {
    seq_length_graph_input_index_ = stoi(llm_seq_length_idx_env);
    enable_multi_level_seq_length_ = true;
    MS_LOG(INFO) << "LLM Manager init: enable multi_level_seq_length with graph input index: "
                 << seq_length_graph_input_index_;
  }

  auto llm_seq_length_level_size = common::GetEnv("MS_LLM_SEQ_LENGTH_LEVEL_SIZE");
  if (!llm_seq_length_level_size.empty()) {
    seq_length_level_size_ = stoi(llm_seq_length_level_size);
  }
  MS_LOG(INFO) << "LLM Manager init: use seq_length_level_size: " << seq_length_level_size_;

  inited_ = true;
}

bool LLMManager::update_round_up_max_seq_length(int32_t max_seq_lenght) {
  if (!enable_multi_level_seq_length_) {
    return false;
  }

  auto round_up_max_seq_length = ((max_seq_lenght / seq_length_level_size_) + 1) * seq_length_level_size_;
  if (current_round_up_max_seq_length_ != round_up_max_seq_length) {
    current_round_up_max_seq_length_ = round_up_max_seq_length;
    return true;
  }
  return false;
}

int32_t LLMManager::get_current_round_up_max_seq_length() { return current_round_up_max_seq_length_; }

int32_t LLMManager::LLMManager::get_seq_length_graph_input_index() { return seq_length_graph_input_index_; }
}  // namespace mindspore
