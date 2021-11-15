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
#include "src/tensor_category.h"

namespace mindspore {
namespace lite {
schema::Tensor *AttrToTensor(const void *data, size_t data_size, bool is_array, TypeId type_id,
                             std::vector<char *> *const tensor_bufs) {
  if (data == nullptr || tensor_bufs == nullptr) {
    MS_LOG(ERROR) << "the parameter of this function is nullptr.";
    return nullptr;
  }
  if (data_size > static_cast<size_t>(INT32_MAX)) {
    MS_LOG(ERROR) << "the amount of data exceeds the INT32_MAX.";
    return nullptr;
  }
  auto shape = is_array ? std::vector<int>{static_cast<int>(data_size)} : std::vector<int>{};
  auto dst_tensor = (is_array ? new (std::nothrow) Tensor(type_id, shape, mindspore::NHWC, Category::CONST_TENSOR)
                              : new (std::nothrow) Tensor(type_id, shape, mindspore::NHWC, Category::CONST_SCALAR));
  if (dst_tensor == nullptr) {
    MS_LOG(ERROR) << "w a tensor failed.";
    return nullptr;
  }
  std::vector<uint8_t> uint8_data;
  uint8_data.resize(dst_tensor->Size());
  memcpy(uint8_data.data(), data, dst_tensor->Size());
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
  size_t byte_num = fbb.GetSize();
  auto tensor_buf = reinterpret_cast<char *>(malloc(byte_num));
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
