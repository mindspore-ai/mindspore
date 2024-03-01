/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>
#include "ops/array_ops.h"
#include "ops/other_ops.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_name.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "mindspore/core/utils/anf_utils.h"
#include "tools/optimizer/graph/quant_fusion_x_offset_to_bias_pass.h"

namespace mindspore::opt {
namespace {
constexpr auto kAttrNameOffset = "offset";
constexpr auto kAttrNameTransposeB = "transpose_b";
constexpr auto kAttrNameNeedFusedXoffsetToBias = "need_fused_x_offset_to_bias";
}  // namespace

ParameterPtr QuantFusionXOffsetToBias::NewQuantFusionXOffsetToBiasNode(const FuncGraphPtr &func_graph, CNodePtr cnode,
                                                                       int bias_index, float x_offset,
                                                                       const tensor::TensorPtr weight, bool transpose) {
  auto shape = weight->shape_c();
  int channel_in = shape[0];
  int channel_out = shape[1];
  if (transpose) {
    channel_in = shape[1];
    channel_out = shape[0];
  }
  std::vector<int32_t> bias_with_x_offset(channel_out);
  auto weight_data = static_cast<int8_t *>(weight->data_c());
  for (int i = 0; i < channel_out; i++) {
    for (int j = 0; j < channel_in; j++) {
      if (transpose) {
        bias_with_x_offset[i] += static_cast<int32_t>(*(weight_data + j + i * channel_in));
      } else {
        bias_with_x_offset[i] += static_cast<int32_t>(*(weight_data + i + j * channel_out));
      }
    }
    bias_with_x_offset[i] *= static_cast<int32_t>(-x_offset);
  }
  if (cnode->size() > static_cast<size_t>(bias_index) && cnode->input(bias_index)->isa<Parameter>()) {
    auto bias_node = cnode->input(bias_index);
    ParameterPtr parameter;
    tensor::TensorPtr bias;
    lite::quant::GetParameterAndTensor(bias_node, &parameter, &bias);
    MS_EXCEPTION_IF_NULL(parameter);
    MS_EXCEPTION_IF_NULL(bias);
    if (bias->Size() != static_cast<size_t>(channel_out)) {
      MS_LOG(ERROR) << "bias shape error, bias size: " << bias->Size() << " channel out size: " << channel_out;
      return nullptr;
    }
    auto bias_data = static_cast<int32_t *>(bias->data_c());
    for (int i = 0; i < channel_out; i++) {
      bias_with_x_offset[i] += *(bias_data + i);
      MS_LOG(DEBUG) << "bias[" << i << "] = " << bias_with_x_offset[i];
    }
  }
  auto bias_with_x_offset_node = opt::BuildIntVecParameterNode(
    func_graph, bias_with_x_offset, cnode->fullname_with_scope() + "_bias_with_x_offset_node");
  auto bias_abstract = bias_with_x_offset_node->abstract();
  MS_EXCEPTION_IF_NULL(bias_abstract);
  std::vector<int64_t> bias_shape = {channel_out};
  bias_abstract->set_shape(std::make_shared<abstract::Shape>(bias_shape));
  return bias_with_x_offset_node;
}

STATUS QuantFusionXOffsetToBias::RunQuantFusionXOffsetToBias(const FuncGraphPtr &func_graph,
                                                             const FuncGraphManagerPtr &manager) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(manager);
  auto support_x_offset_to_bias_primitive_types = {prim::kPrimMatMulAllReduce};
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!lite::quant::CheckNodeInSet(cnode, support_x_offset_to_bias_primitive_types)) {
      continue;
    }
    auto prim = GetCNodePrimitive(cnode);
    if (!prim->HasAttr(kAttrNameNeedFusedXoffsetToBias)) {
      continue;
    }
    MS_LOG(INFO) << "Cnode: " << cnode->fullname_with_scope() << " will do quant x offset to bias";
    int quant_index = 1;
    int weight_index = 2;
    int bias_index = 3;
    auto quant_cnode = cnode->input(quant_index)->cast<CNodePtr>();
    if (!CheckPrimitiveType(quant_cnode, prim::kPrimQuant)) {
      MS_LOG(ERROR) << "cnode input index : " << quant_index << " is not Quant node";
      return RET_ERROR;
    }
    auto quant_primitive = GetValueNode<PrimitivePtr>(quant_cnode->input(0));
    auto x_offset = GetValue<float>(quant_primitive->GetAttr(kAttrNameOffset));
    auto weight_node = cnode->input(weight_index);
    ParameterPtr parameter;
    tensor::TensorPtr weight;
    lite::quant::GetParameterAndTensor(weight_node, &parameter, &weight);
    CHECK_NULL_RETURN(parameter);
    CHECK_NULL_RETURN(weight);
    bool transpose = false;
    if (quant_primitive->HasAttr(kAttrNameTransposeB)) {
      transpose = GetValue<bool>(quant_primitive->GetAttr(kAttrNameTransposeB));
    }
    auto bias_offset_node = NewQuantFusionXOffsetToBiasNode(func_graph, cnode, bias_index, x_offset, weight, transpose);
    CHECK_NULL_RETURN(bias_offset_node);

    // Set New Offset Bias
    manager->SetEdge(cnode, bias_index, bias_offset_node);
  }
  return lite::RET_OK;
}

bool QuantFusionXOffsetToBias::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto ret = RunQuantFusionXOffsetToBias(func_graph, manager);
  MS_CHECK_TRUE_RET(ret == lite::RET_OK, false);
  return true;
}
}  // namespace mindspore::opt
