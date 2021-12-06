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

#ifdef RUNTIME_CONVERT
#include "src/runtime/runtime_convert.h"
#include "include/version.h"
#include "tools/converter/converter.h"
#include "tools/converter/converter_flags.h"

namespace mindspore::lite {
char *RuntimeConvert(const char *model_buf, const size_t &buf_size, size_t *size) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "Invalid input model buffer.";
    return nullptr;
  }

  auto flag = std::make_unique<converter::Flags>();
  flag->fmk = converter::kFmkTypeMs;
  flag->inputDataType = kTypeUnknown;
  flag->outputDataType = kTypeUnknown;
  flag->saveFP16 = false;
  flag->trainModel = false;
#ifdef ENABLE_LITE_ACL
  flag->device = "Ascend310";
#endif

  Converter cvt;
  auto meta_graph = cvt.Convert(flag, model_buf, buf_size);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Convert failed.";
    return nullptr;
  }

  void *lite_buf = nullptr;
  meta_graph->version = Version();
  auto status = TransferMetaGraph(*meta_graph, &lite_buf, size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transfer model failed.";
    delete meta_graph;
    return nullptr;
  }

  delete meta_graph;
  return reinterpret_cast<char *>(lite_buf);
}

char *RuntimeConvert(const std::string &file_path, size_t *size) {
  auto flag = std::make_unique<converter::Flags>();
  flag->fmk = converter::kFmkTypeMs;
  flag->modelFile = file_path;
  flag->inputDataType = kTypeUnknown;
  flag->outputDataType = kTypeUnknown;
  flag->saveFP16 = false;
  flag->trainModel = false;

  Converter cvt;
  auto meta_graph = cvt.Convert(flag);
  MS_LOG(ERROR) << "Convert failed.";
  if (meta_graph == nullptr) {
    return nullptr;
  }

  void *model_buf = nullptr;
  meta_graph->version = Version();
  auto status = TransferMetaGraph(*meta_graph, &model_buf, size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transfer model failed.";
    delete meta_graph;
    return nullptr;
  }

  delete meta_graph;
  return reinterpret_cast<char *>(model_buf);
}
}  // namespace mindspore::lite
#endif
