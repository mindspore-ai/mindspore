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

#include "tools/converter/parser/caffe/caffe_node_parser.h"
#include <memory>
#include "securec/include/securec.h"
#include "ir/dtype/type_id.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
schema::TensorT *ConvertWeight(const caffe::BlobProto &proto) {
  std::unique_ptr<schema::TensorT> weight = std::make_unique<schema::TensorT>();
  if (weight == nullptr) {
    MS_LOG(ERROR) << "new weight failed";
    return nullptr;
  }

  weight->format = schema::Format::Format_NCHW;
  std::vector<int32_t> shapeVec;
  ConvertShape(proto, &shapeVec);
  weight->dims = shapeVec;
  weight->dataType = kNumberTypeFloat32;
  weight->nodeType = NodeType_ValueNode;

  // cal Weight num
  int count = 1;
  for (int dim : shapeVec) {
    if (dim <= 0) {
      MS_LOG(ERROR) << "Convert weight fail, Blob size invalid";
      return nullptr;
    }
    if (dim >= INT_MAX / count) {
      MS_LOG(ERROR) << "Convert weight fail, Blob size exceeds INT_MAX, dim:" << dim << "count:" << count;
      return nullptr;
    }
    count *= dim;
  }

  // get weight
  std::unique_ptr<float[]> buf = std::make_unique<float[]>(count);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "new weight buf failed";
    return nullptr;
  }
  if (proto.double_data_size() > 0) {
    // datatype double
    if (count != proto.double_data_size()) {
      MS_LOG(ERROR) << "Convert weight fail, Blob size does not match shape size, shape size: " << count
                    << "blob size:" << proto.double_data_size();
      return nullptr;
    }

    for (int i = 0; i < count; ++i) {
      buf[i] = proto.double_data(i);
    }
    weight->data.resize(count * sizeof(float));
    if (::memcpy_s(weight->data.data(), count * sizeof(float), reinterpret_cast<uint8_t *>(buf.get()),
                   count * sizeof(float)) != EOK) {
      MS_LOG(ERROR) << "memcpy failed";
      return nullptr;
    }
  } else {
    // datatype float
    if (count != proto.data_size()) {
      MS_LOG(ERROR) << "Convert weight fail, Blob size does not match shape size, shape size" << count
                    << "blob.data_size:%d" << proto.data_size();
      return nullptr;
    }

    weight->data.resize(count * sizeof(float));
    const float *data_ptr = proto.data().data();
    if (data_ptr == nullptr) {
      MS_LOG(ERROR) << "data_ptr is nullptr";
      return nullptr;
    }
    if (::memcpy_s(weight->data.data(), count * sizeof(float), (uint8_t *)data_ptr, count * sizeof(float)) != EOK) {
      MS_LOG(ERROR) << "memcpy failed";
      return nullptr;
    }
  }
  weight->refCount = 1;

  return weight.release();
}

STATUS ConvertShape(const caffe::BlobProto &proto, std::vector<int32_t> *shape) {
  if (shape == nullptr) {
    MS_LOG(ERROR) << "shape is null";
    return RET_ERROR;
  }

  shape->clear();
  if (proto.has_num() || proto.has_channels() || proto.has_height() || proto.has_width()) {
    shape->push_back(proto.num());
    shape->push_back(proto.channels());
    shape->push_back(proto.height());
    shape->push_back(proto.width());
  } else {
    for (int i = 0; i < proto.shape().dim_size(); ++i) {
      shape->push_back(proto.shape().dim(i));
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
