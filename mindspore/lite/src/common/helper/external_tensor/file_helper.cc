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
#include "src/common/helper/external_tensor/file_helper.h"
#include "src/common/file_utils.h"

namespace mindspore::infer::helper {
void *FileExternalTensorHelper::GetExternalTensorData(const mindspore::schema::ExternalData *external_info) {
  if (external_info == nullptr) {
    MS_LOG_ERROR << "external_info is nullptr.";
    return nullptr;
  }
  return mindspore::lite::ReadFileSegment(base_path_ + external_info->location()->str(), external_info->offset(),
                                          external_info->length());
}

void FileExternalTensorHelper::SetExternalTensorData(const mindspore::schema::ExternalData *external_info, void *data) {
  MS_LOG_WARNING << "FileExternalTensorManager not support SetExternalTensorData";
}
}  // namespace mindspore::infer::helper
