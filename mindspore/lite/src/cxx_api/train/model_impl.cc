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

#include "src/cxx_api/model/model_impl.h"
#include <memory>
#include <unordered_map>
#include <algorithm>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/dual_abi_helper.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "include/api/callback/callback.h"
#include "include/api/metrics/metrics.h"
#include "src/lite_model.h"
#include "src/runtime/inner_allocator.h"
#include "src/cxx_api/converters.h"
#include "src/cxx_api/graph/graph_data.h"
#include "src/cxx_api/tensor/tensor_impl.h"
#include "src/cxx_api/tensor_utils.h"
#include "src/cxx_api/metrics/metrics_adapter.h"
#include "src/cxx_api/metrics/metrics_impl.h"
#include "src/cxx_api/callback/callback_adapter.h"
#include "src/cxx_api/callback/callback_impl.h"
#include "src/common/log_adapter.h"
#include "src/train/train_session.h"

namespace mindspore {
Status ModelImpl::PrepareMetrics(Model *model, std::vector<session::Metrics *> *out_ms,
                                 std::vector<session::Metrics *> *adapter_ms) {
  if (out_ms == nullptr || adapter_ms == nullptr) {
    MS_LOG(ERROR) << "Null input callbacks";
    return kLiteUninitializedObj;
  }
  auto model_metrics = GetMetrics();
  for (auto m : model_metrics) {
    if (m == nullptr) {
      MS_LOG(ERROR) << "Null input metrics";
      return kLiteUninitializedObj;
    }
    if (m->metrics_impl_) {
      // For off-the-shelf metrics it is guaranteed that we have also an MSLite implementation
      auto internal_m = m->metrics_impl_->GetInternalMetrics();
      if (internal_m == nullptr) {
        MS_LOG(ERROR) << "Internal metric is null.";
        clearVectorOfPointers(adapter_ms);
        return kLiteUninitializedObj;
      }
      out_ms->push_back(internal_m);
    } else {
      // For custom metric we use the metric adapter to mediate between MSLite level to API level
      auto adapter_m = new (std::nothrow) MetricsAdapter(m);
      if (adapter_m == nullptr) {  // Error during allocation
        MS_LOG(ERROR) << "Error during allocation";
        clearVectorOfPointers(adapter_ms);
        return kLiteNullptr;
      }
      out_ms->push_back(adapter_m);
      adapter_ms->push_back(adapter_m);
    }
  }
  return kSuccess;
}

Status ModelImpl::ConvertCallbacks(Model *model, std::vector<TrainCallBack *> *i_cbs,
                                   std::vector<session::TrainLoopCallBack *> *o_cbs,
                                   std::vector<session::TrainLoopCallBack *> *adapter_cbs) {
  if (i_cbs == nullptr || o_cbs == nullptr || adapter_cbs == nullptr) {
    MS_LOG(ERROR) << "Null input callbacks";
    return kLiteUninitializedObj;
  }
  for (auto cb : *i_cbs) {
    if (cb == nullptr) {
      return kLiteUninitializedObj;
    }
    if (cb->callback_impl_) {
      // For off-the-shelf callback it is guaranteed that we have also an MSLite implementation
      auto internal_cb = cb->callback_impl_->GetInternalCallback();
      if (internal_cb == nullptr) {
        MS_LOG(ERROR) << "Internal callback is null";
        clearVectorOfPointers(adapter_cbs);
        return kLiteUninitializedObj;
      }
      o_cbs->push_back(internal_cb);
    } else {
      // For custom callbacks we use the callback adapter to mediate between MSLite level to API level
      auto adapter_cb = new (std::nothrow) TrainLoopCallBackAdapter(model, cb);
      if (adapter_cb == nullptr) {  // Error during allocation
        MS_LOG(ERROR) << "Error during allocation";
        clearVectorOfPointers(adapter_cbs);
        return kLiteNullptr;
      }
      o_cbs->push_back(adapter_cb);
      adapter_cbs->push_back(adapter_cb);
    }
  }
  return kSuccess;
}
}  // namespace mindspore
