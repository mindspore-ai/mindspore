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
#include "minddata/dataset/engine/gnn/tensor_proto.h"

#include <algorithm>
#include <utility>
#include <unordered_map>

namespace mindspore {
namespace dataset {

const std::unordered_map<DataTypePb, DataType::Type> g_pb2datatype_map{
  {DataTypePb::DE_PB_UNKNOWN, DataType::DE_UNKNOWN}, {DataTypePb::DE_PB_BOOL, DataType::DE_BOOL},
  {DataTypePb::DE_PB_INT8, DataType::DE_INT8},       {DataTypePb::DE_PB_UINT8, DataType::DE_UINT8},
  {DataTypePb::DE_PB_INT16, DataType::DE_INT16},     {DataTypePb::DE_PB_UINT16, DataType::DE_UINT16},
  {DataTypePb::DE_PB_INT32, DataType::DE_INT32},     {DataTypePb::DE_PB_UINT32, DataType::DE_UINT32},
  {DataTypePb::DE_PB_INT64, DataType::DE_INT64},     {DataTypePb::DE_PB_UINT64, DataType::DE_UINT64},
  {DataTypePb::DE_PB_FLOAT16, DataType::DE_FLOAT16}, {DataTypePb::DE_PB_FLOAT32, DataType::DE_FLOAT32},
  {DataTypePb::DE_PB_FLOAT64, DataType::DE_FLOAT64}, {DataTypePb::DE_PB_STRING, DataType::DE_STRING},
};

const std::unordered_map<DataType::Type, DataTypePb> g_datatype2pb_map{
  {DataType::DE_UNKNOWN, DataTypePb::DE_PB_UNKNOWN}, {DataType::DE_BOOL, DataTypePb::DE_PB_BOOL},
  {DataType::DE_INT8, DataTypePb::DE_PB_INT8},       {DataType::DE_UINT8, DataTypePb::DE_PB_UINT8},
  {DataType::DE_INT16, DataTypePb::DE_PB_INT16},     {DataType::DE_UINT16, DataTypePb::DE_PB_UINT16},
  {DataType::DE_INT32, DataTypePb::DE_PB_INT32},     {DataType::DE_UINT32, DataTypePb::DE_PB_UINT32},
  {DataType::DE_INT64, DataTypePb::DE_PB_INT64},     {DataType::DE_UINT64, DataTypePb::DE_PB_UINT64},
  {DataType::DE_FLOAT16, DataTypePb::DE_PB_FLOAT16}, {DataType::DE_FLOAT32, DataTypePb::DE_PB_FLOAT32},
  {DataType::DE_FLOAT64, DataTypePb::DE_PB_FLOAT64}, {DataType::DE_STRING, DataTypePb::DE_PB_STRING},
};

Status TensorToPb(const std::shared_ptr<Tensor> tensor, TensorPb *tensor_pb) {
  CHECK_FAIL_RETURN_UNEXPECTED(tensor, "Parameter tensor is a null pointer");
  CHECK_FAIL_RETURN_UNEXPECTED(tensor_pb, "Parameter tensor_pb is a null pointer");

  std::vector<dsize_t> shape = tensor->shape().AsVector();
  for (auto dim : shape) {
    tensor_pb->add_dims(static_cast<google::protobuf::int64>(dim));
  }
  auto iter = g_datatype2pb_map.find(tensor->type().value());
  if (iter == g_datatype2pb_map.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid tensor type: " + tensor->type().ToString());
  }
  tensor_pb->set_tensor_type(iter->second);
  tensor_pb->set_data(tensor->GetBuffer(), tensor->SizeInBytes());
  return Status::OK();
}

Status PbToTensor(const TensorPb *tensor_pb, std::shared_ptr<Tensor> *tensor) {
  CHECK_FAIL_RETURN_UNEXPECTED(tensor_pb, "Parameter tensor_pb is a null pointer");
  CHECK_FAIL_RETURN_UNEXPECTED(tensor, "Parameter tensor is a null pointer");

  std::vector<dsize_t> shape;
  shape.resize(tensor_pb->dims().size());
  std::transform(tensor_pb->dims().begin(), tensor_pb->dims().end(), shape.begin(),
                 [](const google::protobuf::int64 dim) { return static_cast<dsize_t>(dim); });
  auto iter = g_pb2datatype_map.find(tensor_pb->tensor_type());
  if (iter == g_pb2datatype_map.end()) {
    RETURN_STATUS_UNEXPECTED("Invalid Tensor_pb type: " + std::to_string(tensor_pb->tensor_type()));
  }
  DataType::Type type = iter->second;
  std::shared_ptr<Tensor> tensor_out;
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(TensorShape(shape), DataType(type),
                                            reinterpret_cast<const unsigned char *>(tensor_pb->data().data()),
                                            tensor_pb->data().size(), &tensor_out));
  *tensor = std::move(tensor_out);
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
