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

#include "src/ops/compat/attr_transfer_common.h"
#include <vector>
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
schema::Tensor *AttrToTensor(void *data, int data_size, bool is_array, TypeId type_id,
                             std::vector<char *> *tensor_bufs) {
  if (data == nullptr || tensor_bufs == nullptr) {
    MS_LOG(ERROR) << "the parameter of this function is nullptr.";
    return nullptr;
  }
  auto dst_tensor =
    (is_array ? new (std::nothrow) Tensor(type_id, {data_size}, schema::Format_NHWC, Tensor::Category::CONST_TENSOR)
              : new (std::nothrow) Tensor(type_id, {}, schema::Format_NHWC, Tensor::Category::CONST_SCALAR));
  auto dst_data = dst_tensor->MutableData();
  if (dst_data == nullptr) {
    MS_LOG(ERROR) << "Data from tensor is nullptr";
    return nullptr;
  }
  std::vector<uint8_t> uint8_data;
  uint8_data.resize(dst_tensor->Size());
  memcpy(uint8_data.data(), data, dst_tensor->Size());
  auto shape = dst_tensor->shape();
  flatbuffers::FlatBufferBuilder fbb(1024);
  auto tensor_offset =
    schema::CreateTensorDirect(fbb, NodeType_ValueNode, type_id, &shape, schema::Format_NHWC, 0, 0, &uint8_data);
  fbb.Finish(tensor_offset);
  delete dst_tensor;
  auto buf = fbb.GetBufferPointer();
  if (buf == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer return nullptr";
    fbb.Clear();
    return nullptr;
  }
  auto tensor_buf = reinterpret_cast<char *>(malloc(fbb.GetSize()));
  if (tensor_buf == nullptr) {
    MS_LOG(ERROR) << "malloc primitive_buf_ failed";
    fbb.Clear();
    return nullptr;
  }
  memcpy(tensor_buf, buf, fbb.GetSize());
  auto tensor = flatbuffers::GetRoot<schema::Tensor>(tensor_buf);
  tensor_bufs->push_back(tensor_buf);
  fbb.Clear();
  return const_cast<schema::Tensor *>(tensor);
}
}  // namespace lite
}  // namespace mindspore
