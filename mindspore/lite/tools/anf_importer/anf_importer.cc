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

#include "tools/anf_importer/anf_importer.h"
#include <utility>
#include "schema/model_generated.h"
#include "ir/dtype.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
namespace mindspore {
namespace lite {
int AnfImporter::Import(const converter::Flags *flag) {
  auto ret = ConverterConstTensor();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "ConverterConstTensor failed " << ret;
    return ret;
  }
  ret = ConverterCNode();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "ConverterCNode failed " << ret;
    return ret;
  }
  ret = AddReturnCNode();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "AddReturnCNode failed " << ret;
    return ret;
  }
  return RET_OK;
}

AnfNodePtr AnfImporter::GetNode(int tensor_id) {
  auto n = nodes_.find(tensor_id);
  if (n == nodes_.end()) {
    return nullptr;
  }
  return n->second;
}

void AnfImporter::AddNode(int tensor_id, AnfNodePtr node) { nodes_[tensor_id] = std::move(node); }
}  // namespace lite
}  // namespace mindspore
