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
int TransferStridedSliceAttr(Model::Node *node, std::vector<schema::Tensor *> *dst_tensors,
                             std::vector<char *> *tensor_bufs) {
  if (node == nullptr || node->primitive_ == nullptr || dst_tensors == nullptr || tensor_bufs == nullptr) {
    MS_LOG(ERROR) << "the parameter of this function is nullptr.";
    return RET_ERROR;
  }
  dst_tensors->clear();
  auto prim = reinterpret_cast<const schema::v0::Primitive *>(node->primitive_);
  int inputs_size = node->input_indices_.size();
  switch (inputs_size) {
    case 1: {
      auto begins_attr = prim->value_as_StridedSlice()->begin();
      std::vector<int> dst_begins = std::vector<int>(begins_attr->begin(), begins_attr->end());
      auto dst_begins_tensor = AttrToTensor(dst_begins.data(), dst_begins.size(), true, kNumberTypeInt32, tensor_bufs);
      dst_tensors->push_back(dst_begins_tensor);
    }
    case 2: {
      auto ends_attr = prim->value_as_StridedSlice()->end();
      std::vector<int> dst_ends = std::vector<int>(ends_attr->begin(), ends_attr->end());
      auto dst_ends_tensor = AttrToTensor(dst_ends.data(), dst_ends.size(), true, kNumberTypeInt32, tensor_bufs);
      dst_tensors->push_back(dst_ends_tensor);
    }
    case 3: {
      auto strides_attr = prim->value_as_StridedSlice()->stride();
      std::vector<int> dst_strides = std::vector<int>(strides_attr->begin(), strides_attr->end());
      auto dst_strides_tensor =
        AttrToTensor(dst_strides.data(), dst_strides.size(), true, kNumberTypeInt32, tensor_bufs);
      dst_tensors->push_back(dst_strides_tensor);
      break;
    }
    default: {
      MS_LOG(DEBUG) << "stride_slice don't need to convert attr to tensor.";
      return RET_OK;
    }
  }
  if (std::any_of(dst_tensors->begin(), dst_tensors->end(), [](schema::Tensor *tensor) { return tensor == nullptr; })) {
    MS_LOG(ERROR) << "convert attr to tensor failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

Register StridedSliceTransferRegistry(SCHEMA_VERSION::SCHEMA_V0, schema::v0::PrimitiveType_StridedSlice,
                                      TransferStridedSliceAttr);
}  // namespace lite
}  // namespace mindspore
