/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/weight_format_transform_pass.h"
#include <memory>
#include <algorithm>
#include <vector>
#include "ops/fusion/conv2d_backprop_input_fusion.h"
#include "ops/transpose.h"
#include "tools/optimizer/common/gllo_utils.h"

using mindspore::lite::converter::FmkType_CAFFE;
using mindspore::lite::converter::FmkType_MS;
using mindspore::lite::converter::FmkType_ONNX;
using mindspore::lite::converter::FmkType_TFLITE;
using mindspore::schema::QuantType_AwareTraining;
using mindspore::schema::QuantType_PostTraining;
using mindspore::schema::QuantType_QUANT_NONE;
using mindspore::schema::QuantType_WeightQuant;

namespace mindspore::opt {
namespace {
constexpr size_t kFirstInputIndex = 1;
constexpr size_t kConvWeightIndex = 2;
const PrimitivePtr kPrimConv2DBackpropInputFusion = std::make_shared<Primitive>(ops::kNameConv2DBackpropInputFusion);
lite::STATUS GetTransposePerm(schema::Format src_format, schema::Format dst_format, std::vector<int> *perm) {
  MS_ASSERT(perm != nullptr);
  auto src_format_str = std::string(schema::EnumNameFormat(src_format));
  auto dst_format_str = std::string(schema::EnumNameFormat(dst_format));
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
}  // namespace

void WeightFormatTransformPass::SetQuantType(QuantType type) { this->quant_type = type; }
void WeightFormatTransformPass::SetFmkType(FmkType type) { this->fmk_type = type; }
void WeightFormatTransformPass::SetDstFormat(schema::Format format) { this->dst_format = format; }
lite::STATUS WeightFormatTransformPass::TransposeInsertForWeightSharing(const FuncGraphPtr &graph,
                                                                        const ParameterPtr &weight_node,
                                                                        std::vector<int> perm) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(weight_node != nullptr);
  auto node_list = TopoSort(graph->get_return());
  std::vector<CNodePtr> adjust_nodes;
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimConv2DFusion) || CheckPrimitiveType(node, kPrimConv2DBackpropInputFusion) ||
        CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion) ||
        CheckPrimitiveType(node, prim::kPrimApplyMomentum) || CheckPrimitiveType(node, prim::kPrimSGD) ||
        CheckPrimitiveType(node, prim::kPrimAdam)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    if (std::any_of(inputs.begin(), inputs.end(),
                    [&weight_node](const AnfNodePtr &anf_node) { return weight_node == anf_node; })) {
      adjust_nodes.push_back(cnode);
    }
  }
  if (adjust_nodes.empty()) {
    MS_LOG(DEBUG) << "do not need to adjust nodes.";
    return lite::RET_OK;
  }
  auto perm_node = BuildIntVecParameterNode(graph, perm, weight_node->fullname_with_scope() + "_perm");
  auto prim = std::make_shared<ops::Transpose>();
  auto transpose_node = graph->NewCNode(prim, {weight_node, perm_node});
  auto type_ptr = TypeIdToType(kTypeUnknown);
  std::vector<int64_t> shape_vector;
  auto abstract = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  transpose_node->set_abstract(abstract);
  transpose_node->set_fullname_with_scope(weight_node->fullname_with_scope() + "_post");
  for (auto &adjust_node : adjust_nodes) {
    auto inputs = adjust_node->inputs();
    std::replace_if(
      inputs.begin(), inputs.end(), [&weight_node](const AnfNodePtr &anf_node) { return weight_node == anf_node; },
      transpose_node);
    adjust_node->set_inputs(inputs);
  }
  return lite::RET_OK;
}

lite::STATUS WeightFormatTransformPass::HandleWeightSharing(const FuncGraphPtr &graph, const ParameterPtr &weight_node,
                                                            schema::Format src_format, schema::Format dst_format) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(weight_node != nullptr);
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  std::vector<int> perm;
  auto status = GetTransposePerm(src_format, dst_format, &perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get perm failed.";
    return status;
  }
  status = TransposeInsertForWeightSharing(graph, weight_node, perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "transpose insert failed.";
  }
  return status;
}

lite::STATUS WeightFormatTransformPass::ConvWeightFormatTrans(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimConv2DFusion) &&
        !CheckPrimitiveType(node, kPrimConv2DBackpropInputFusion) &&
        !CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
      continue;
    }
    auto conv_cnode = node->cast<CNodePtr>();
    MS_ASSERT(conv_cnode->inputs().size() > kConvWeightIndex);
    auto weight_node = conv_cnode->input(kConvWeightIndex);
    MS_ASSERT(weight_node != nullptr);
    auto weight_value = GetLiteParamValue(weight_node);
    if (weight_value == nullptr) {
      MS_LOG(ERROR) << "weight node must param value";
      return false;
    }
    MS_ASSERT(weight_value->tensor_type() == TypeId::kNumberTypeFloat32 ||
              weight_value->tensor_type() == TypeId::kNumberTypeUInt8);
    lite::STATUS status;
    schema::Format src_format = static_cast<schema::Format>(weight_value->format());
    schema::Format weight_dst_format = schema::Format::Format_KHWC;
    if (dst_format != schema::Format::Format_NUM_OF_FORMAT) {
      weight_dst_format = dst_format;
    }
    status = TransFilterFormat(weight_value, weight_dst_format);
    if (status == RET_OK) {
      weight_value->set_format(weight_dst_format);
    } else {
      MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(schema::EnumValuesFormat()[weight_value->format()]) << "To"
                    << EnumNameFormat(weight_dst_format) << " failed, node : " << node->fullname_with_scope()
                    << "quant type:" << quant_type;
      return ERROR;
    }
    status = HandleWeightSharing(graph, weight_node->cast<ParameterPtr>(), src_format, weight_dst_format);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "handle weight-sharing failed.";
      return false;
    }
    auto type_id = static_cast<TypeId>(weight_value->tensor_type());
    auto type_ptr = TypeIdToType(type_id);
    auto shape = weight_value->tensor_shape();
    std::vector<int64_t> shape_vector;
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                         [](const int32_t &value) { return static_cast<int64_t>(value); });
    auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
    weight_node->set_abstract(abstract_tensor);
  }
  return RET_OK;
}

bool WeightFormatTransformPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto status = ConvWeightFormatTrans(func_graph);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Conv2D weight FormatTrans failed: " << status;
    return status;
  }
  return false;
}
}  // namespace mindspore::opt
