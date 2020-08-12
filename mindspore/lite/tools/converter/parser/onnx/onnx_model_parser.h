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

#ifndef MS_ONNX_MODEL_PARSER_H
#define MS_ONNX_MODEL_PARSER_H

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include "securec/include/securec.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/onnx/onnx.pb.h"

namespace mindspore {
namespace lite {
class OnnxModelParser : public ModelParser {
 public:
  OnnxModelParser();
  virtual ~OnnxModelParser();
  MetaGraphT *Parse(const std::string &modelFile, const std::string &weightFile,
                    const QuantType &quantType = QuantType_QUANT_NONE) override;

 private:
  TypeId GetDataTypeFromOnnx(onnx::TensorProto_DataType onnx_type);
  std::vector<int32_t> GetDimsFromOnnxValue(const onnx::ValueInfoProto &onnx_value);
  STATUS ReadOnnxModelFromBinary(const std::string &modelFile, google::protobuf::Message *model_proto);
  STATUS SetGraphConstTensor(const onnx::GraphProto &onnx_graph, TensorCache *tensor_cache);
  STATUS SetGraphInputTensor(const onnx::GraphProto &onnx_graph, schema::MetaGraphT *graph, TensorCache *tensor_cache);
  STATUS SetGraphOutputTensor(const onnx::GraphProto &onnx_graph, schema::MetaGraphT *graph, TensorCache *tensor_cache);
  STATUS AddTensorCache(const onnx::ValueInfoProto &proto, schema::TensorT *tensor);
  STATUS ParseOnnxNodeToDstOp(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                              schema::CNodeT *dst_op, schema::TensorT *dst_tensor, TensorCache *tensor_cache);
  void ParseOnnxGemmNode(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                         schema::MetaGraphT *graph, TensorCache *tensor_cache);
  STATUS ParseOnnxGivenFillNode(const onnx::NodeProto &onnx_node, TensorCache *tensor_cache);
  STATUS ParseOnnxNodeAttr(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                           const string &onnx_op_type, schema::CNodeT *dst_op);
  void SetOpQuantParams(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *dst_op,
                        schema::TensorT *dst_tensor, TensorCache *tensor_cache);
  STATUS SetOpInputIndex(const std::vector<string> &node_inputs, schema::CNodeT *dst_op,
                         const onnx::NodeProto &onnx_node, TensorCache *tensor_cache);
  STATUS SetOpOutputIndex(const std::vector<string> &node_outputs, schema::CNodeT *dst_op, TensorCache *tensor_cache);
  STATUS CopyOnnxTensorData(const onnx::TensorProto &onnx_init_value, schema::TensorT *tensor);
  STATUS SetAllTensors(const TensorCache &tensor_cache, schema::MetaGraphT *graphDef);
  void FindGraphInputAndConst(const onnx::GraphProto &onnx_graph);

 private:
  std::vector<string> graphInputNames;
  std::vector<string> graphConstNames;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MS_ONNX_MODEL_PARSER_H
