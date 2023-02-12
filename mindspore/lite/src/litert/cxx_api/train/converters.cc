/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "src/litert/cxx_api/converters.h"
#include "include/train/train_cfg.h"
#include "include/api/cfg.h"
#include "src/common/log_adapter.h"

namespace mindspore {
Status A2L_ConvertConfig(const TrainCfg *a_train_cfg, lite::TrainCfg *l_train_cfg) {
  if ((a_train_cfg == nullptr) || (l_train_cfg == nullptr)) {
    MS_LOG(ERROR) << "Invalid train_cfg pointers";
    return kLiteNullptr;
  }

  std::vector<std::string> a_loss_name = a_train_cfg->GetLossName();
  l_train_cfg->loss_name_.assign(a_loss_name.begin(), a_loss_name.end());
  l_train_cfg->mix_precision_cfg_.dynamic_loss_scale_ = a_train_cfg->mix_precision_cfg_.loss_scale_;
  l_train_cfg->mix_precision_cfg_.loss_scale_ = a_train_cfg->mix_precision_cfg_.loss_scale_;
  l_train_cfg->mix_precision_cfg_.keep_batchnorm_fp32_ = (a_train_cfg->optimization_level_ != kO3);
  l_train_cfg->mix_precision_cfg_.num_of_not_nan_iter_th_ = a_train_cfg->mix_precision_cfg_.num_of_not_nan_iter_th_;
  l_train_cfg->accumulate_gradients_ = a_train_cfg->accumulate_gradients_;
  return kSuccess;
}
}  // namespace mindspore
