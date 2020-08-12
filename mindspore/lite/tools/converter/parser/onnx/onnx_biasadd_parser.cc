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

#include <memory>
#include "tools/converter/parser/onnx/onnx_biasadd_parser.h"

// using namespace mindspore::predict;
// using namespace onnx;
// using namespace std;
namespace mindspore {
namespace lite {
STATUS OnnxBiasAddParser::Parse(const onnx::GraphProto &onnx_graph,
                                const onnx::NodeProto &onnx_node,
                                schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx BiasAddParser";
  std::unique_ptr<schema::BiasAddT> attr(new schema::BiasAddT());
  // use channel dim as axis
  attr->axis = {1};
  if (op != nullptr) {
    op->primitive = std::make_unique<schema::PrimitiveT>();
    op->primitive->value.type = schema::PrimitiveType_BiasAdd;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxBiasAddParser("BiasAdd", new OnnxBiasAddParser());
}  // namespace lite
}  // namespace mindspore

