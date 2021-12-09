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
#include "parser/parser_utils.h"
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <string>
#include "common/file_util.h"
#include "ops/op_utils.h"
#include "common/format_utils.h"
#include "common/anf_util.h"
#include "parser/caffe/inputs_adjust.h"
#include "common/data_transpose_utils.h"

namespace mindspore::lite {
namespace {
const int WARNING_THRESHOLD = 536870912 * 2;
bool IsWeightNodeSensitive(const AnfNodePtr &node) {
  return dpico::CheckPrimitiveType(node, prim::kPrimConv2DFusion) ||
         dpico::CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion) ||
         dpico::CheckPrimitiveType(node, prim::kPrimApplyMomentum) || dpico::CheckPrimitiveType(node, prim::kPrimSGD) ||
         dpico::CheckPrimitiveType(node, prim::kPrimAdam);
}

int GetTransposePerm(mindspore::Format src_format, mindspore::Format dst_format, std::vector<int> *perm) {
  MS_ASSERT(perm != nullptr);
  auto src_format_str = dpico::FormatEnumToString(src_format);
  auto dst_format_str = dpico::FormatEnumToString(dst_format);
  if (src_format_str.empty() || dst_format_str.empty() || src_format_str.size() != dst_format_str.size()) {
    MS_LOG(ERROR) << "src_format or dst_format is error.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < src_format_str.size(); ++i) {
    auto pos = src_format_str.find(dst_format_str[i]);
    if (pos == std::string::npos) {
      MS_LOG(ERROR) << "src_format and dst_format don't match.";
      return lite::RET_ERROR;
    }
    perm->push_back(static_cast<int>(pos));
  }
  return lite::RET_OK;
}

int GetTransposePermSharing(mindspore::Format src_format, mindspore::Format dst_format, std::vector<int> *perm) {
  MS_ASSERT(perm != nullptr);
  auto src_format_str = dpico::FormatEnumToString(src_format);
  auto dst_format_str = dpico::FormatEnumToString(dst_format);
  if (src_format_str.empty() || dst_format_str.empty() || src_format_str.size() != dst_format_str.size()) {
    MS_LOG(ERROR) << "src_format or dst_format is error.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < src_format_str.size(); ++i) {
    auto pos = dst_format_str.find(src_format_str[i]);
    if (pos == std::string::npos) {
      MS_LOG(ERROR) << "src_format and dst_format don't match.";
      return lite::RET_ERROR;
    }
    perm->push_back(static_cast<int>(pos));
  }
  return lite::RET_OK;
}

int UnifyVariableConvWeight(const api::FuncGraphPtr &graph, const AnfNodePtr &weight_node, mindspore::Format src_format,
                            mindspore::Format dst_format, std::set<AnfNodePtr> *has_visited) {
  MS_ASSERT(graph != nullptr && weight_node != nullptr && has_visited != nullptr);
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  std::vector<int> perm;
  auto status = GetTransposePerm(src_format, dst_format, &perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get perm failed.";
    return status;
  }
  auto manager = api::FuncGraphManager::Manage(graph);
  MS_ASSERT(manager != nullptr);
  CNodePtr trans_cnode = nullptr;
  auto node_map = manager->node_users();
  auto &weight_node_users = node_map[weight_node];
  for (auto &weight_node_user : weight_node_users) {
    auto post_node = weight_node_user.first;
    if (!utils::isa<CNodePtr>(post_node)) {
      MS_LOG(ERROR) << "post node is invalid.";
      return RET_ERROR;
    }
    if (!IsWeightNodeSensitive(post_node)) {
      continue;
    }
    has_visited->insert(post_node);
    if (trans_cnode == nullptr) {
      trans_cnode =
        dpico::GenTransposeNode(graph, weight_node, perm, weight_node->fullname_with_scope() + "_post_perm");
      MS_ASSERT(trans_cnode != nullptr);
      auto abstract = weight_node->abstract();
      ShapeVector shape;
      if (abstract != nullptr) {
        ShapeVector weight_shape;
        if (dpico::FetchShapeFromAbstract(abstract, &weight_shape) != RET_OK) {
          MS_LOG(ERROR) << "fetch shape from abstract failed.";
          return RET_ERROR;
        }
        if (!weight_shape.empty()) {
          if (weight_shape.size() != dpico::kDims4) {
            MS_LOG(ERROR) << "conv weight shape is invalid, which is not 4D, now is " << weight_shape.size();
            return RET_ERROR;
          }
          (void)std::transform(perm.begin(), perm.end(), std::back_inserter(shape),
                               [&weight_shape](const int index) { return weight_shape[index]; });
        }
        abstract = abstract->Clone();
      } else {
        abstract = dpico::CreateTensorAbstract(shape, TypeId::kNumberTypeFloat32);
        MS_ASSERT(abstract != nullptr);
      }
      auto shape_ptr = std::make_shared<abstract::Shape>(shape);
      if (shape_ptr == nullptr) {
        MS_LOG(ERROR) << "shape ptr is nullptr.";
        return RET_ERROR;
      }
      abstract->set_shape(shape_ptr);
      trans_cnode->set_abstract(abstract);
    }
    auto post_cnode = post_node->cast<CNodePtr>();
    manager->SetEdge(post_cnode, weight_node_user.second, trans_cnode);
  }
  return RET_OK;
}

int HandleConstConvWeightShared(const api::FuncGraphPtr &graph, const AnfNodePtr &weight_node,
                                mindspore::Format src_format, mindspore::Format dst_format,
                                std::set<AnfNodePtr> *has_visited) {
  MS_ASSERT(graph != nullptr && weight_node != nullptr && has_visited != nullptr);
  if (src_format == dst_format) {
    return RET_OK;
  }
  std::vector<int> perm;
  auto status = GetTransposePermSharing(src_format, dst_format, &perm);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get perm failed.";
    return status;
  }
  auto manager = api::FuncGraphManager::Manage(graph);
  MS_ASSERT(manager != nullptr);
  CNodePtr trans_cnode = nullptr;
  auto node_map = manager->node_users();
  auto &weight_node_users = node_map[weight_node];
  for (auto &weight_node_user : weight_node_users) {
    auto post_node = weight_node_user.first;
    if (!utils::isa<CNodePtr>(post_node)) {
      MS_LOG(ERROR) << "post node is invalid.";
      return RET_ERROR;
    }
    if (IsWeightNodeSensitive(post_node)) {
      has_visited->insert(post_node);
      continue;
    }
    if (trans_cnode == nullptr) {
      trans_cnode =
        dpico::GenTransposeNode(graph, weight_node, perm, weight_node->fullname_with_scope() + "_post_perm");
      MS_ASSERT(trans_cnode != nullptr);
      auto prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
      MS_ASSERT(prim != nullptr);
      prim->AddAttr(ops::kFormat, MakeValue<int64_t>(dst_format));
      auto weight_value = dpico::GetTensorInfo(weight_node);
      MS_ASSERT(weight_value != nullptr);
      auto weight_shape = weight_value->shape();
      ShapeVector shape;
      if (!weight_shape.empty()) {
        if (weight_shape.size() != dpico::kDims4) {
          MS_LOG(ERROR) << "conv weight shape is invalid, which is not 4D, now is " << weight_shape.size();
          return RET_ERROR;
        }
        (void)std::transform(perm.begin(), perm.end(), std::back_inserter(shape),
                             [&weight_shape](const int index) { return weight_shape[index]; });
      }
      auto abstract = weight_node->abstract();
      MS_ASSERT(abstract != nullptr);
      abstract = abstract->Clone();
      auto shape_ptr = std::make_shared<abstract::Shape>(shape);
      if (shape_ptr == nullptr) {
        MS_LOG(ERROR) << "shape ptr is nullptr.";
        return RET_ERROR;
      }
      abstract->set_shape(shape_ptr);
      trans_cnode->set_abstract(abstract);
    }
    auto post_cnode = post_node->cast<CNodePtr>();
    manager->SetEdge(post_cnode, weight_node_user.second, trans_cnode);
  }
  return RET_OK;
}

int UnifyConstConvWeight(const api::FuncGraphPtr &graph, const AnfNodePtr &weight_node, mindspore::Format src_format,
                         mindspore::Format dst_format, std::set<AnfNodePtr> *has_visited) {
  MS_ASSERT(graph != nullptr && weight_node != nullptr && has_visited != nullptr);
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  auto weight_value = dpico::GetTensorInfo(weight_node);
  if (weight_value == nullptr) {
    MS_LOG(ERROR) << "conv weight is non-const.";
    return RET_ERROR;
  }
  auto status = dpico::TransFilterFormat(weight_value, src_format, dst_format);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "TransFilter " << dpico::FormatEnumToString(src_format) << "To"
                  << dpico::FormatEnumToString(dst_format) << " failed, node : " << weight_node->fullname_with_scope();
    return RET_ERROR;
  }
  auto type_id = static_cast<TypeId>(weight_value->data_type());
  auto shape = weight_value->shape();
  auto abstract = dpico::CreateTensorAbstract(shape, type_id);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return RET_ERROR;
  }
  weight_node->set_abstract(abstract);
  if (HandleConstConvWeightShared(graph, weight_node, src_format, dst_format, has_visited) != RET_OK) {
    MS_LOG(ERROR) << "handle const conv weight-shared failed, node name is " << weight_node->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

void GetAllFuncGraph(const api::FuncGraphPtr &func_graph, std::set<api::FuncGraphPtr> *all_func_graphs) {
  MS_ASSERT(func_graph != nullptr && all_func_graphs != nullptr);
  if (all_func_graphs->find(func_graph) == all_func_graphs->end()) {
    all_func_graphs->insert(func_graph);
  } else {
    return;
  }

  auto nodes = func_graph->nodes();
  for (auto &node : nodes) {
    auto new_fg = api::FuncGraph::GetFuncGraphFromAnfNode(node);
    if (new_fg != nullptr) {
      GetAllFuncGraph(new_fg, all_func_graphs);
    }
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      for (auto &input : cnode->inputs()) {
        if (input->isa<ValueNode>()) {
          new_fg = api::FuncGraph::GetFuncGraphFromAnfNode(node);
          if (new_fg != nullptr) {
            GetAllFuncGraph(new_fg, all_func_graphs);
          }
        }
      }
    }
  }
}

int PostAdjust(const std::set<api::FuncGraphPtr> &all_func_graphs) {
  for (const auto &func_graph : all_func_graphs) {
    auto adjust_input = std::make_shared<InputAdjust>();
    if (adjust_input == nullptr) {
      MS_LOG(ERROR) << "adjust input is nullptr.";
      return RET_ERROR;
    }
    if (!adjust_input->Run(func_graph)) {
      MS_LOG(ERROR) << "adjust input failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int UnifyConvWeightFormat(const api::FuncGraphPtr &graph, const CNodePtr &cnode, mindspore::Format src_format,
                          mindspore::Format dst_format, std::set<AnfNodePtr> *has_visited) {
  MS_ASSERT(graph != nullptr && cnode != nullptr && has_visited != nullptr);
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  if (!dpico::CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) &&
      !dpico::CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
    MS_LOG(ERROR) << "cnode is not a member of convolution's family.";
    return RET_ERROR;
  }
  bool is_const_weight = true;
  auto weight_node = cnode->input(dpico::kInputIndex2);
  if (utils::isa<CNode>(weight_node)) {
    is_const_weight = false;
  } else if (utils::isa<Parameter>(weight_node)) {
    auto weight_param_node = weight_node->cast<ParameterPtr>();
    if (!weight_param_node->has_default()) {
      is_const_weight = false;
    }
  }
  int status;
  if (is_const_weight) {
    status = UnifyConstConvWeight(graph, weight_node, src_format, dst_format, has_visited);
  } else {
    status = UnifyVariableConvWeight(graph, weight_node, src_format, dst_format, has_visited);
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "unfiy conv weight failed, cnode name is " << cnode->fullname_with_scope();
  }
  return status;
}
bool ReadProtoFromCodedInputStream(google::protobuf::io::CodedInputStream *coded_stream,
                                   google::protobuf::Message *proto) {
  if (proto == nullptr) {
    MS_LOG(ERROR) << "incorrect parameter. nullptr == proto";
    return false;
  }
  coded_stream->SetTotalBytesLimit(INT_MAX, WARNING_THRESHOLD);
  return proto->ParseFromCodedStream(coded_stream);
}

STATUS ReadProtoFromText(const char *file, google::protobuf::Message *message) {
  if (file == nullptr || message == nullptr) {
    return RET_ERROR;
  }

  std::string realPath = dpico::RealPath(file);
  if (realPath.empty()) {
    MS_LOG(ERROR) << "Proto file path " << file << " is  not valid";
    return RET_ERROR;
  }

  std::ifstream fs(realPath.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    MS_LOG(ERROR) << "Open proto file " << file << " failed.";
    return RET_ERROR;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  bool status = google::protobuf::TextFormat::Parse(&input, message);
  if (!status) {
    MS_LOG(ERROR) << "call [google::protobuf::TextFormat::Parse] func status fail, please check your text file.";
    fs.close();
    return RET_ERROR;
  }

  fs.close();
  return RET_OK;
}

STATUS ReadProtoFromBinaryFile(const char *file, google::protobuf::Message *message) {
  if (file == nullptr || message == nullptr) {
    return RET_ERROR;
  }

  std::string realPath = dpico::RealPath(file);
  if (realPath.empty()) {
    MS_LOG(ERROR) << "Binary proto file path " << file << " is not valid";
    return RET_ERROR;
  }

  std::ifstream fs(realPath, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    MS_LOG(ERROR) << "Open binary proto file " << file << " failed.";
    return RET_ERROR;
  }

  google::protobuf::io::IstreamInputStream istream(&fs);
  google::protobuf::io::CodedInputStream coded_stream(&istream);

  bool success = ReadProtoFromCodedInputStream(&coded_stream, message);
  fs.close();

  if (!success) {
    MS_LOG(DEBUG) << "Parse " << file << " failed.";
    return RET_ERROR;
  }

  return RET_OK;
}
STATUS ValidateFileStr(const std::string &modelFile, const std::string &fileType) {
  if (modelFile.size() > fileType.size() && modelFile.substr(modelFile.size() - fileType.size()) == fileType) {
    return RET_OK;
  } else {
    return RET_ERROR;
  }
}
}  // namespace mindspore::lite
