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

#ifndef MINDSPORE_CCSRC_TOOLS_LITE_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_H_
#define MINDSPORE_CCSRC_TOOLS_LITE_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_H_

#include <string>
#include <vector>
#include "google/protobuf/message.h"
#include "mindspore/lite/schema/inner/model_generated.h"
#include "tools/converter/parser/caffe/caffe.pb.h"
#include "mindspore/lite/tools/converter/parser/caffe/caffe_node_parser.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {

class CaffeNodeParser {
 public:
  explicit CaffeNodeParser(const std::string &nodeName) : name(nodeName) {}

  virtual ~CaffeNodeParser() {}

  virtual int Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight, schema::CNodeT *op,
                       std::vector<schema::TensorT *> *weightVec) = 0;

 protected:
  const std::string &name;
};

schema::TensorT *ConvertWeight(const caffe::BlobProto &proto);

STATUS ConvertShape(const caffe::BlobProto &proto, std::vector<int32_t> *shape);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_LITE_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_H_

