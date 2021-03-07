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

#include "schema/model_v0_generated.h"
#include "src/ops/compat/attr_transfer_common.h"

namespace mindspore {
namespace lite {
int TransferPadAttr(Model::Node *node, std::vector<schema::Tensor *> *dst_tensors, std::vector<char *> *tensor_bufs) {
  if (node == nullptr || node->primitive_ == nullptr || dst_tensors == nullptr || tensor_bufs == nullptr) {
    MS_LOG(ERROR) << "the parameter of this function is nullptr.";
    return RET_ERROR;
  }
  dst_tensors->clear();
  if (node->input_indices_.size() > 1) {
    MS_LOG(DEBUG) << "pad don't need to convert attr to tensor.";
    return RET_OK;
  }
  auto prim = reinterpret_cast<const schema::v0::Primitive *>(node->primitive_);
  auto paddings_attr = prim->value_as_Pad()->paddings();
  std::vector<int> paddings = std::vector<int>(paddings_attr->begin(), paddings_attr->end());
  auto paddings_tensor = AttrToTensor(paddings.data(), paddings.size(), true, kNumberTypeInt32, tensor_bufs);
  if (paddings_tensor == nullptr) {
    MS_LOG(ERROR) << "attr tensor is nullptr, transform is failed.";
    return RET_NULL_PTR;
  }
  dst_tensors->push_back(paddings_tensor);
  return RET_OK;
}

Register PadTransferRegistry(SCHEMA_VERSION::SCHEMA_V0, schema::v0::PrimitiveType_Pad, TransferPadAttr);
}  // namespace lite
}  // namespace mindspore
