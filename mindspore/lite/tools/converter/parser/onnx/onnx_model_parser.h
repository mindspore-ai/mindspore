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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <map>
#include "securec/include/securec.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "tools/converter/parser/onnx/onnx_tensor_parser.h"
#include "proto/onnx.pb.h"

namespace mindspore {
namespace lite {
class OnnxModelParser : public ModelParser {
 public:
  OnnxModelParser();

  virtual ~OnnxModelParser();

  //  schema::MetaGraphT *ParseGraph(const onnx::GraphProto &graph, const QuantType &quantType = QuantType_QUANT_NONE);
  int ParseGraph(schema::MetaGraphT *dst_graph, schema::SubGraphT *dst_sub_graph, const onnx::GraphProto &onnx_graph,
                 const QuantType &quantType);

  schema::MetaGraphT *ParseToFb(const std::string &modelFile, const std::string &weightFile,
                                const QuantType &quantType = QuantType_QUANT_NONE) override;

  static TypeId GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type);

 private:
  std::vector<int32_t> GetDimsFromOnnxValue(const onnx::ValueInfoProto &onnx_value);

  STATUS SetGraphConstTensor(const onnx::GraphProto &onnx_graph);

  STATUS SetGraphInputTensor(const onnx::GraphProto &onnx_graph, schema::SubGraphT *graph);

  STATUS SetGraphOutputTensor(const onnx::GraphProto &onnx_graph, schema::SubGraphT *graph);

  STATUS AddValueInfo(const onnx::ValueInfoProto &proto, const std::string &name, const Category &type, int *index);

  STATUS AddTensorProto(const onnx::TensorProto &proto, const std::string &name, const Category &type, int *index);

  STATUS ParseOnnxNodeToDstOp(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                              schema::CNodeT *dst_op, const QuantType &quantType, schema::MetaGraphT *dst_graph);

  void ParseOnnxGemmNode(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                         schema::SubGraphT *sub_graph, schema::MetaGraphT *graph, const QuantType &quant_type);

  STATUS ParseOnnxGivenFillNode(const onnx::NodeProto &onnx_node);

  STATUS ParseOnnxNodeAttr(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                           const string &onnx_op_type, schema::CNodeT *dst_op);

  void SetOpQuantParams(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *dst_op,
                        schema::TensorT *dst_tensor);

  STATUS SetOpInputIndex(const std::vector<string> &node_inputs, schema::CNodeT *dst_op,
                         const onnx::NodeProto &onnx_node);

  STATUS SetOpOutputIndex(const std::vector<string> &node_outputs, schema::CNodeT *dst_op);

  STATUS CopyOnnxTensorData(const onnx::TensorProto &onnx_init_value, schema::TensorT *tensor);

  STATUS SetAllTensors(schema::MetaGraphT *graphDef);

  void FindGraphInputAndConst(const onnx::GraphProto &onnx_graph);

  STATUS ParseSubgraph(schema::CNodeT *dst_op, const onnx::NodeProto &onnx_node, const QuantType &quantType,
                       schema::MetaGraphT *dst_graph);

 private:
  std::vector<std::string> graphInputNames;
  std::vector<std::string> graphConstNames;
  int subGraphNum = 0;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_MODEL_PARSER_H
