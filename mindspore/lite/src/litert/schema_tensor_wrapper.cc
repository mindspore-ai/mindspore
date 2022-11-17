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

#include "src/litert/schema_tensor_wrapper.h"
#include "src/common/log_adapter.h"
#include "src/common/file_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
// don't check data_size and shape_size: bit_pack or huffman_code
// don't check tensor category: variable-tensor-list may have data
#ifdef ENABLE_LITE_HELPER
bool SchemaTensorWrapper::Init(const schema::Tensor &tensor, const SCHEMA_VERSION schema_version,
                               const std::string &base_path, mindspore::infer::helper::InferHelpers *infer_helpers) {
#else
bool SchemaTensorWrapper::Init(const schema::Tensor &tensor, const SCHEMA_VERSION schema_version,
                               const std::string &base_path) {
#endif
  // add magic-num-check and checksum-check here
  this->handler_ = &tensor;
  if (tensor.data() != nullptr && tensor.data()->data() != nullptr) {
    auto data = tensor.data()->data();
    auto data_size = tensor.data()->size();
    this->length_ = data_size;
    this->data_ = const_cast<unsigned char *>(data);
    this->if_own_data_ = false;
    return true;
  }
  if (schema_version == SCHEMA_V0) {
    return true;
  }
  if (tensor.externalData() == nullptr) {
    return true;
  }
  if (tensor.externalData()->size() != 1) {
    MS_LOG(ERROR) << "Only support tensor saved in one file now";
    return false;
  }
  auto external_data = tensor.externalData()->Get(0);
  this->length_ = static_cast<size_t>(external_data->length());
#ifdef ENABLE_LITE_HELPER
  if (infer_helpers != nullptr && infer_helpers->GetExternalTensorHelper() != nullptr) {
    this->data_ = infer_helpers->GetExternalTensorHelper()->GetExternalTensorData(external_data);
    this->if_own_data_ = false;
  } else {
    this->data_ =
      ReadFileSegment(base_path + external_data->location()->str(), external_data->offset(), external_data->length());
    this->if_own_data_ = true;
  }
#else
  this->data_ =
    ReadFileSegment(base_path + external_data->location()->str(), external_data->offset(), external_data->length());
  this->if_own_data_ = true;
#endif
  return true;
}
}  // namespace lite
}  // namespace mindspore
