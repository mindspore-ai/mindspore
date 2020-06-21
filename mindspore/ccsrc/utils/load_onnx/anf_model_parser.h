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

#ifndef MINDSPORE_CCSRC_UTILS_LOAD_ONNX_ANF_MODEL_PARSER_H
#define MINDSPORE_CCSRC_UTILS_LOAD_ONNX_ANF_MODEL_PARSER_H

#include <string>
#include <map>
#include <unordered_map>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "ir/func_graph.h"
#include "proto/onnx.pb.h"

namespace mindspore {
namespace lite {
using int32 = int32_t;
using int64 = int64_t;
using uint64 = uint64_t;
using float16 = Eigen::half;
class MSANFModelParser {
 public:
  MSANFModelParser() = default;
  ~MSANFModelParser() = default;

  FuncGraphPtr Parse(const onnx::ModelProto &model_proto);
  bool MSANFParseModelConfigureInfo(const onnx::ModelProto &model_proto);

  std::string GetProducerName() { return producer_name_; }
  int GetProducerVersion() { return model_version_; }
  int GetIrVersion() { return ir_version_; }

 private:
  bool BuildFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto);
  bool ImportParametersForGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto);
  bool ImportNodesForGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto);
  bool BuildParameterForFuncGraph(const ParameterPtr &node, const onnx::ValueInfoProto &value_proto);
  CNodePtr BuildCNodeForFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::NodeProto &node_proto);
  bool BuildReturnForFuncGraph(const FuncGraphPtr &outputFuncGraph, const onnx::GraphProto &importProto,
                               const CNodePtr &cnode_ptr);
  bool GetAttrValueForCNode(const PrimitivePtr &prim, const onnx::AttributeProto &attr_proto);
  bool ObtainCNodeAttrInTypeForm(const PrimitivePtr &prim, const std::string &attr_name,
                                 const onnx::TensorProto &attr_tensor);
  bool ObtainCNodeAttrInScalarForm(const PrimitivePtr &prim, const std::string &attr_name,
                                   const onnx::TensorProto &attr_tensor);
  bool ObtainCNodeAttrInTensorForm(const PrimitivePtr &prim, const std::string &attr_name,
                                   const onnx::TensorProto &attr_tensor);
  bool BuildValueNodeForFuncGraph(const onnx::NodeProto &node_proto);
  bool ObtainValueNodeInTensorForm(const string &value_node_name, const onnx::TensorProto &attr_tensor);

  bool ObtainValueNodeInScalarForm(const string &value_node_name, const onnx::TensorProto &attr_tensor);
  bool GetAttrValueForValueNode(const string &ref_attr_name, const std::string &value_node_name,
                                const onnx::TensorProto &attr_tensor);
  bool ObtainValueNodeInTypeForm(const string &value_node_name, const onnx::TensorProto &attr_tensor);
  AbstractBasePtr GetAbstractForCNode(const onnx::AttributeProto &attr_proto);

  std::string producer_name_;
  int model_version_;
  int ir_version_;
  std::unordered_map<std::string, AnfNodePtr> anfnode_build_map_;
  std::map<std::string, onnx::TensorProto> default_para_map_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_LOAD_ONNX_ANF_MODEL_PARSER_H
