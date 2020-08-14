/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include<memory>
#include "src/common/anf_importer/import_from_meta_graph.h"
namespace mindspore::lite::train {
std::shared_ptr<ModelImpl> Import(const char *model_buf, size_t size) {
  MS_EXCEPTION_IF_NULL(model_buf);
  flatbuffers::Verifier verify((const uint8_t *) model_buf, size);
  if (!schema::VerifyMetaGraphBuffer(verify)) {
    MS_LOG(ERROR) << "The buffer is invalid and fail to create graph.";
    return nullptr;
  }
  // todo hangangqiang remove when copy primitive done
  if (size <= 0) {
    MS_LOG(ERROR) << "size is zero";
    return nullptr;
  }
  auto *inner_buf = new char[size];
  memcpy(inner_buf, model_buf, size);
  auto meta_graph = schema::GetMetaGraph(inner_buf);
  auto model = std::make_shared<ModelImpl>(meta_graph);
  auto ret = model->BuildOps();
  if (0 != ret) {
    MS_LOG(ERROR) << "BuildOps failed";
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(meta_graph);
  auto importer = new AnfImporterFromMetaGraph(model);
  auto ret2 = importer->Import();
  if (0 != ret2) {
    MS_LOG(ERROR) << "Import anf_graph from meta_graph failed, ret2: " << ret2;
    return nullptr;
  }
  return model;
}
}  // namespace mindspore::lite::train
