/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "cxx_api/model/model_impl.h"

namespace mindspore {
Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid data, graph_ is null.";
    return kMCFailed;
  }

  if (graph_cell_ == nullptr) {
    MS_LOG(WARNING) << "Model has not been built, it will be built with default options";
    Status ret = Build();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed.";
      return ret;
    }
  }

  MS_EXCEPTION_IF_NULL(graph_cell_);
  Status ret = graph_cell_->Run(inputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed.";
    return ret;
  }

  return kSuccess;
}
}  // namespace mindspore
