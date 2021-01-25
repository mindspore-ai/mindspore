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
int TransferSliceAttr(Model::Node *node, std::vector<schema::Tensor *> *dst_tensors, std::vector<char *> *tensor_bufs) {
  if (node == nullptr || dst_tensors == nullptr || tensor_bufs == nullptr) {
    MS_LOG(ERROR) << "the parameter of this function is nullptr.";
    return RET_ERROR;
  }
  if (node->input_indices_.size() != 1) {
    MS_LOG(DEBUG) << "Slice don't need to convert attr to tensor.";
    return RET_OK;
  }
  MS_ASSERT(dst_tensors->size() == 0);
  auto prim = reinterpret_cast<const schema::v0::Primitive *>(node->primitive_);

  /* transfer begin tensor */
  auto begin_attr = prim->value_as_Slice()->begin();
  std::vector<int32_t> begin_shape = std::vector<int>(begin_attr->begin(), begin_attr->end());
  auto begin_tensor = AttrToTensor(begin_shape.data(), begin_shape.size(), true, kNumberTypeInt32, tensor_bufs);
  if (begin_tensor == nullptr) {
    MS_LOG(ERROR) << "slice transfer begin failed";
    return RET_NULL_PTR;
  }
  dst_tensors->push_back(begin_tensor);

  /* transfer size tensor */
  auto size_attr = prim->value_as_Slice()->size();
  std::vector<int32_t> size_shape = std::vector<int>(size_attr->begin(), size_attr->end());
  auto size_tensor = AttrToTensor(size_shape.data(), size_shape.size(), true, kNumberTypeInt32, tensor_bufs);
  if (size_tensor == nullptr) {
    MS_LOG(ERROR) << "slice transfer size failed";
    return RET_NULL_PTR;
  }
  dst_tensors->push_back(size_tensor);

  return RET_OK;
}

Register SliceTransferRegistry(SCHEMA_VERSION::SCHEMA_V0, schema::v0::PrimitiveType_Slice, TransferSliceAttr);
}  // namespace lite
}  // namespace mindspore
