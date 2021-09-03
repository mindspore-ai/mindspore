/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "include/api/model.h"
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/callback/callback.h"
#include "include/api/dual_abi_helper.h"
#include "src/cxx_api/model/model_impl.h"
#include "src/cxx_api/callback/callback_impl.h"
#include "src/cxx_api/callback/callback_adapter.h"
#include "src/common/log_adapter.h"
#include "include/train/train_loop.h"
#include "include/train/train_loop_callback.h"

namespace mindspore {
Status Model::Train(int epochs, std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> i_cbs) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr) || ds == nullptr) {
    MS_LOG(ERROR) << "Model implement or dataset is null.";
    return kLiteUninitializedObj;
  }
  auto loop = std::unique_ptr<session::TrainLoop>(session::TrainLoop::CreateTrainLoop((impl_->session_).get()));
  if (loop == nullptr) {
    MS_LOG(ERROR) << "Error during allocation of train loop";
    return kLiteNullptr;
  }

  // Convert Metrics to MS Lite and init loop
  std::vector<session::Metrics *> metrics;
  std::vector<session::Metrics *> adapter_metrics;
  auto status = impl_->PrepareMetrics(this, &metrics, &adapter_metrics);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Error during preparation of metrics";
    return status;
  }
  (void)loop->Init(metrics);

  // Convert Callbacks to be used by loop
  std::vector<session::TrainLoopCallBack *> cbs;
  std::vector<session::TrainLoopCallBack *> adapter_cbs;
  status = impl_->ConvertCallbacks(this, &i_cbs, &cbs, &adapter_cbs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Error during preparation of callbacks";
    clearVectorOfPointers(&adapter_metrics);
    return status;
  }

  auto ret = loop->Train(epochs, ds.get(), cbs);

  clearVectorOfPointers(&adapter_metrics);
  clearVectorOfPointers(&adapter_cbs);

  return (ret == mindspore::lite::RET_OK) ? kSuccess : kLiteError;
}

Status Model::Evaluate(std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> i_cbs) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr) || ds == nullptr) {
    MS_LOG(ERROR) << "Model implement or dataset is null.";
    return kLiteUninitializedObj;
  }

  auto loop = std::unique_ptr<session::TrainLoop>(session::TrainLoop::CreateTrainLoop((impl_->session_).get()));
  if (loop == nullptr) {
    MS_LOG(ERROR) << "Error during allocation of train loop";
    return kLiteNullptr;
  }

  // Convert Metrics to MS Lite and init loop
  std::vector<session::Metrics *> metrics;
  std::vector<session::Metrics *> adapter_metrics;
  auto status = impl_->PrepareMetrics(this, &metrics, &adapter_metrics);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Error during preparation of metrics";
    return status;
  }
  (void)loop->Init(metrics);

  // Convert Callbacks to be used by loop
  std::vector<session::TrainLoopCallBack *> cbs;
  std::vector<session::TrainLoopCallBack *> adapter_cbs;
  status = impl_->ConvertCallbacks(this, &i_cbs, &cbs, &adapter_cbs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Error during preparation of callbacks";
    clearVectorOfPointers(&adapter_metrics);
    return status;
  }

  auto ret = loop->Eval(ds.get(), cbs);

  clearVectorOfPointers(&adapter_metrics);
  clearVectorOfPointers(&adapter_cbs);

  return (ret == mindspore::lite::RET_OK) ? kSuccess : kLiteError;
}

}  // namespace mindspore
