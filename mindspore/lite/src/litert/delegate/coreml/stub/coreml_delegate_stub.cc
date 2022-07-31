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
#include "include/api/delegate.h"
#include "src/common/log_adapter.h"

namespace mindspore {
Status CoreMLDelegate::Init() {
  MS_LOG(ERROR) << "Only supported by IOS system and the MSLITE_ENABLE_COREML is turned on";
  return kLiteError;
}

Status CoreMLDelegate::Build(DelegateModel<schema::Primitive> *model) {
  MS_LOG(ERROR) << "Only supported by IOS system and the MSLITE_ENABLE_COREML is turned on";
  return kLiteError;
}
}  // namespace mindspore
