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

#include <memory>
#include <unordered_map>
#include <algorithm>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/dual_abi_helper.h"
#include "include/api/callback/callback.h"
#include "include/api/metrics/metrics.h"
#include "src/litert/lite_model.h"
#include "src/litert/inner_context.h"
#include "src/litert/inner_allocator.h"
#include "src/litert/cxx_api/model/model_impl.h"
#include "src/litert/cxx_api/converters.h"
#include "src/litert/cxx_api/graph/graph_data.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/litert/cxx_api/tensor_utils.h"
#include "src/litert/cxx_api/metrics/metrics_adapter.h"
#include "src/litert/cxx_api/metrics/metrics_impl.h"
#include "src/litert/cxx_api/callback/callback_adapter.h"
#include "src/litert/cxx_api/callback/callback_impl.h"
#include "src/common/log_adapter.h"
#include "src/train/train_session.h"
#include "src/train/static_allocator.h"

namespace mindspore {
std::shared_ptr<lite::LiteSession> CreateTrainSession(std::shared_ptr<Graph::GraphData> graph_data,
                                                      std::shared_ptr<TrainCfg> cfg,
                                                      const std::shared_ptr<lite::InnerContext> &context) {
  MS_CHECK_TRUE_MSG(graph_data != nullptr, nullptr, "graph data cannot be nullptr");
  bool is_train_session = graph_data->IsTrainModel();
  if (is_train_session) {
    auto model = graph_data->lite_model();
    if (model == nullptr || model->buf == nullptr) {
      MS_LOG(ERROR) << "Lite model has been freed.";
      return nullptr;
    }
    std::shared_ptr<lite::LiteSession> shared_session;
    auto session = new (std::nothrow) lite::TrainSession();
    if (session == nullptr) {
      MS_LOG(ERROR) << "create session failed";
      return nullptr;
    }
    shared_session.reset(session);

    context->allocator = std::make_shared<StaticAllocator>();
    if (context->allocator == nullptr) {
      MS_LOG(ERROR) << " cannot convert to static allocation";
      return nullptr;
    }

    lite::TrainCfg train_cfg;
    if (cfg != nullptr) {
      auto status = A2L_ConvertConfig(cfg.get(), &train_cfg);
      if (status != kSuccess) {
        MS_LOG(ERROR) << "Failed to convert Config to Lite Config";
        return nullptr;
      }
    }

    auto ret = session->TrainInit(context, &train_cfg);
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(ERROR) << "init session failed";
      return nullptr;
    }

    ret = session->CompileTrainGraph(model);
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(ERROR) << "Compiling Train Graph session failed";
      return nullptr;
    }
    return shared_session;
  }
  MS_LOG(DEBUG) << "Session is not a train session.";
  return nullptr;
}

class TrainSupport {
 public:
  TrainSupport() { CreateTrainSessionCallbackHolder(CreateTrainSession); }
  ~TrainSupport() {}
};

TrainSupport support_train_api;
}  // namespace mindspore
