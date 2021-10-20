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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_NODE_PARSER_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_NODE_PARSER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "include/registry/converter_context.h"

namespace onnx {
class GraphProto;
class NodeProto;
}  // namespace onnx

namespace caffe {
class LayerParameter;
}  // namespace caffe

namespace tensorflow {
class NodeDef;
}  // namespace tensorflow

namespace tflite {
struct OperatorT;
struct SubGraphT;
struct ModelT;
}  // namespace tflite

namespace mindspore {
namespace ops {
/// \brief PrimitiveC defined a base class for storing properties
class PrimitiveC;
}  // namespace ops
namespace converter {
/// \brief NodeParser defined a base class for parsing node's attributes.
class MS_API NodeParser {
 public:
  /// \brief Constructor.
  NodeParser() = default;

  /// \brief Destructor.
  virtual ~NodeParser() = default;

  /// \brief Method to parse node of ONNX.
  ///
  /// \param[in] onnx_graph Define the onnx graph, which contains all information about the graph.
  /// \param[in] onnx_node Define the node to be resolved.
  ///
  /// \return PrimitiveC Attribute storage.
  virtual ops::PrimitiveC *Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
    return nullptr;
  }

  /// \brief Method to parse node of CAFFE.
  ///
  /// \param[in] proto Define the node which contains attributes.
  /// \param[in] weight Define the node which contains weight information.
  ///
  /// \return PrimitiveC Attribute storage.
  virtual ops::PrimitiveC *Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
    return nullptr;
  }

  /// \brief Method to parse node of TF.
  ///
  /// \param[in] tf_op Define the node to be resolved.
  /// \param[in] tf_node_map Define the all nodes of the graph.
  /// \param[in] inputs Define the input name, that determines which inputs will be parsed including their order.
  ///            Determined by user.
  /// \param[in] output_size Define the output num of current node, which need to be determined by user.
  ///
  /// \return PrimitiveC Attribute storage.
  virtual ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
    return nullptr;
  }

  /// \brief Method to parse node of TFLITE
  ///
  /// \param[in] tflite_op Define the node to be resolved.
  /// \param[in] tflite_model Define the model, which contains all information abort the graph.
  ///
  /// \return PrimitiveC Attribute storage.
  virtual ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                 const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                 const std::unique_ptr<tflite::ModelT> &tflite_model) {
    return nullptr;
  }
};
/// \brief NodeParserPtr defined a shared_ptr type.
using NodeParserPtr = std::shared_ptr<NodeParser>;
}  // namespace converter
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_NODE_PARSER_H_
