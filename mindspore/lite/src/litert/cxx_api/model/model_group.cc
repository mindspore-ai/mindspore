/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/api/model_group.h"
#include <mutex>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/dual_abi_helper.h"
#include "src/litert/cxx_api/model/model_group_impl.h"
#include "src/common/log_adapter.h"

namespace mindspore {
ModelGroup::ModelGroup(ModelGroupFlag flags) {
  impl_ = std::make_shared<ModelGroupImpl>(flags);
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "New model group impl_ failed.";
  }
}

Status ModelGroup::AddModel(const std::vector<std::vector<char>> &model_path_list) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model group implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->AddModel(model_path_list);
}

Status ModelGroup::AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model group implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->AddModel(model_buff_list);
}

Status ModelGroup::AddModel(const std::vector<Model> &) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status ModelGroup::CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model group implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->CalMaxSizeOfWorkspace(model_type, ms_context);
}
}  // namespace mindspore
