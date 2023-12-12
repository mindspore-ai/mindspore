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
#include "extendrt/cxx_api/llm_engine/ms_llm_engine.h"
#include <set>
#include "mindspore/lite/src/common/common.h"
#include "src/extendrt/cxx_api/llm_engine/post_process_model_manager.h"

namespace mindspore {
namespace llm {
Status MSLLMEngine::Init(const LLMModelConfig *model_config, KVCacheConfig *kvcache_config, uint64_t cluster_id,
                         const std::map<std::string, std::string> &options, GenerateParameters *default_param) {
  if (model_config == nullptr) {
    MS_LOG(ERROR) << "Model config is nullptr";
    return kLiteNullptr;
  }

  if (model_config->prefill_model_paths.size() != 1 || model_config->prefill_model_paths[0].empty() ||
      model_config->decoder_model_paths.size() != 1 || model_config->decoder_model_paths[0].empty()) {
    MS_LOG(ERROR) << "Model path cannot be empty, and only support one model now";
    return kLiteNullptr;
  }

  model_group_ = std::make_shared<mindspore::ModelGroupImpl>(mindspore::ModelGroupFlag::kShareWeight);
  if (model_group_ == nullptr) {
    MS_LOG(ERROR) << "New model group impl_ failed.";
    return kLiteNullptr;
  }

  prefill_model_ = std::make_shared<mindspore::ModelImpl>();
  if (prefill_model_ == nullptr) {
    MS_LOG(ERROR) << "New prefill model impl_ failed.";
    return kLiteNullptr;
  }

  decoder_model_ = std::make_shared<mindspore::ModelImpl>();
  if (decoder_model_ == nullptr) {
    MS_LOG(ERROR) << "New prefill model impl_ failed.";
    return kLiteNullptr;
  }

  model_group_->AddModel({prefill_model_, decoder_model_});
  auto status = prefill_model_->Build(model_config->prefill_model_paths[0], mindspore::ModelType::kMindIR,
                                      model_config->prefill_context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to load prefill model " << model_config->prefill_model_paths[0];
    return status;
  }

  status = decoder_model_->Build(model_config->decoder_model_paths[0], mindspore::ModelType::kMindIR,
                                 model_config->decoder_context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to load decoder model " << model_config->decoder_model_paths[0];
    return status;
  }

  default_param_ = default_param;
  return status;
}

Status MSLLMEngine::SetMaxSeqLen(int64_t max_seq_len) {
  max_seq_len_ = max_seq_len;
  // reshape kvcache shape
  return kSuccess;
}

Status MSLLMEngine::ResizeKVCache(const KVCacheConfig &config) {
  MS_LOG(ERROR) << "The lib is not support resize KVCache.";
  return kLiteNotSupport;
}

Status MSLLMEngine::CopyKVCacheToHost(int64_t batch_index, const std::vector<MSTensor> &key_tensor,
                                      const std::vector<MSTensor> &value_tensor) {
  MS_LOG(ERROR) << "The lib is not support copy KVCache to host.";
  return kLiteNotSupport;
}
Status MSLLMEngine::CopyKVCacheToDevice(int64_t batch_index, const std::vector<MSTensor> &key_tensor,
                                        const std::vector<MSTensor> &value_tensor) {
  MS_LOG(ERROR) << "The lib is not support copy KVCache to device.";
  return kLiteNotSupport;
}

Status MSLLMEngine::Prefill(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                            GenerateParameters *param) {
  if (prefill_model_ == nullptr) {
    MS_LOG(ERROR) << "prefill_model_ is nullptr";
    return kLiteNullptr;
  }
  auto gen_param = param;
  if (gen_param == nullptr) {
    gen_param = default_param_;
  }
  std::vector<MSTensor> tmp_outputs;

  std::vector<MSTensor> *prefill_outputs = &tmp_outputs;
  if (gen_param == nullptr) {
    prefill_outputs = outputs;
  }

  auto status = prefill_model_->Predict(inputs, prefill_outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to prefill";
    return status;
  }

  auto post_model = PostProcessModelManager::GetInstance().GetModel(gen_param);
  if (post_model == nullptr) {
    return kSuccess;
  }
  status = post_model->Predict(*prefill_outputs, outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to post process";
    return status;
  }

  return status;
}

std::vector<MSTensor> MSLLMEngine::Prefill(const std::vector<MSTensor> &inputs, GenerateParameters *param) {
  std::vector<MSTensor> outputs;
  auto status = Prefill(inputs, &outputs, param);
  if (status != kSuccess) {
    return {};
  }
  return outputs;
}

Status MSLLMEngine::Decode(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                           GenerateParameters *param) {
  if (decoder_model_ == nullptr) {
    MS_LOG(ERROR) << "decoder_model_ is nullptr";
    return kLiteNullptr;
  }
  auto gen_param = param;
  if (gen_param == nullptr) {
    gen_param = default_param_;
  }
  std::vector<MSTensor> tmp_outputs;

  std::vector<MSTensor> *prefill_outputs = &tmp_outputs;
  if (gen_param == nullptr) {
    prefill_outputs = outputs;
  }

  auto status = decoder_model_->Predict(inputs, prefill_outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to prefill";
    return status;
  }

  auto post_model = PostProcessModelManager::GetInstance().GetModel(gen_param);
  if (post_model == nullptr) {
    return kSuccess;
  }
  status = post_model->Predict(*prefill_outputs, outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to post process";
    return status;
  }

  return status;
}

std::vector<MSTensor> MSLLMEngine::Decode(const std::vector<MSTensor> &inputs, GenerateParameters *param) {
  std::vector<MSTensor> outputs;
  auto status = Decode(inputs, &outputs, param);
  if (status != kSuccess) {
    return {};
  }
  return outputs;
}

std::vector<MSTensor> MSLLMEngine::DecodeAsync(const MSLLMRequest &req, const std::vector<MSTensor> &inputs,
                                               GenerateParameters *param) {
  MS_LOG(ERROR) << "The lib is not support decode async.";
  return {};
}

std::vector<MSTensor> MSLLMEngine::PrefillAsync(const MSLLMRequest &req, const std::vector<MSTensor> &inputs,
                                                GenerateParameters *param) {
  MS_LOG(ERROR) << "The lib is not support prefill async.";
  return {};
}

}  // namespace llm
}  // namespace mindspore
