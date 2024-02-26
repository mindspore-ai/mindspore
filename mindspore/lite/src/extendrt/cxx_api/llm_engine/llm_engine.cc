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
#include "extendrt/cxx_api/llm_engine/llm_engine.h"
#include "mindspore/lite/src/extendrt/cxx_api/dlutils.h"
#include "mindspore/lite/src/extendrt/cxx_api/file_utils.h"
#include "mindspore/lite/src/common/common.h"
#include "mindspore/lite/src/extendrt/cxx_api/llm_engine/llm_engine_impl.h"

namespace mindspore {
LLMEngine::LLMEngine(LLMRole role, uint64_t cluster_id, const VecChar &batch_mode) {
  impl_ = std::make_shared<LLMEngineImpl>(role, cluster_id, CharToString(batch_mode));
}

Status LLMEngine::AddModelInner(mindspore::LLMModel *llm_model, const std::vector<VecChar> &model_paths_c,
                                const std::map<VecChar, VecChar> &options_c, const VecChar &postprocess_model_path_c) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  if (llm_model == nullptr || llm_model->impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed to add model, input argument llm_model is nullptr";
    return kLiteError;
  }
  auto model_paths = VectorCharToString(model_paths_c);
  auto options = MapVectorCharToString(options_c);
  auto postprocess_model_path = CharToString(postprocess_model_path_c);
  uint64_t model_id = -1;
  auto status = impl_->AddModel(model_paths, options, postprocess_model_path, &model_id);
  if (status != kSuccess) {
    return status;
  }
  llm_model->impl_->SetModelId(model_id);
  llm_model->impl_->SetLLMEngine(impl_);
  return kSuccess;
}

Status LLMEngine::InitInner(const std::map<VecChar, VecChar> &options) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->Init(MapVectorCharToString(options));
}

void LLMEngine::Finalize() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return;
  }
  impl_->Finalize();
}

Status LLMEngine::LinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets,
                               int32_t timeout) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->LinkClusters(clusters, rets, timeout);
}

Status LLMEngine::UnlinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets,
                                 int32_t timeout) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->UnlinkClusters(clusters, rets, timeout);
}

LLMEngineStatus LLMEngine::FetchStatus() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return LLMEngineStatus();
  }
  return impl_->FetchStatus();
}

Status LLMEngine::CompleteRequest(const LLMReq &req) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->CompleteRequest(req);
}

LLMModel::LLMModel() { impl_ = std::make_shared<LLMModelImpl>(); }

Status LLMModel::Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (impl_ == nullptr || impl_->GetLLMEngine() == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->GetLLMEngine()->Predict(req, inputs, outputs, impl_->GetModelId());
}

Status LLMModel::Predict(const std::vector<LLMReq> &req, const std::vector<MSTensor> &inputs,
                         std::vector<MSTensor> *outputs) {
  if (impl_ == nullptr || impl_->GetLLMEngine() == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->GetLLMEngine()->Predict(req, inputs, outputs, impl_->GetModelId());
}
Status LLMModel::PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs) {
  if (impl_ == nullptr || impl_->GetLLMEngine() == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->GetLLMEngine()->PreloadPromptPrefix(req, inputs, impl_->GetModelId());
}

Status LLMModel::ReleasePromptPrefix(const LLMReq &req) {
  if (impl_ == nullptr || impl_->GetLLMEngine() == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->GetLLMEngine()->ReleasePromptPrefix(req, impl_->GetModelId());
}

Status LLMModel::PullKV(const LLMReq &req) {
  if (impl_ == nullptr || impl_->GetLLMEngine() == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->GetLLMEngine()->PullKV(req, impl_->GetModelId());
}

Status LLMModel::MergeKV(const LLMReq &req, uint32_t batch_index, uint32_t batch_id) {
  if (impl_ == nullptr || impl_->GetLLMEngine() == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return kLiteError;
  }
  return impl_->GetLLMEngine()->MergeKV(req, batch_index, batch_id, impl_->GetModelId());
}

std::vector<LLMTensorInfo> LLMModel::GetInputInfos() {
  if (impl_ == nullptr || impl_->GetLLMEngine() == nullptr) {
    MS_LOG(ERROR) << "LLMEngine impl is nullptr";
    return {};
  }
  return impl_->GetLLMEngine()->GetInputInfos(impl_->GetModelId());
}
}  // namespace mindspore
