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

#include "parser/caffe/caffe_node_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
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
