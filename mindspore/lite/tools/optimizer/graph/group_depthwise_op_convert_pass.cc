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
#include "tools/optimizer/graph/group_depthwise_op_convert_pass.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/tensor.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "src/common/log_adapter.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvInputIndex = 1;
}  // namespace

bool GroupDepthwiseOpConvertPass::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimConv2DFusion)) {
      continue;
    }

    auto conv_cnode = node->cast<CNodePtr>();
    auto prim_node = conv_cnode->input(0);
    MS_ASSERT(prim_node != nullptr);
    auto prim_value_node = prim_node->cast<ValueNodePtr>();
    MS_ASSERT(prim_value_node != nullptr && prim_value_node->value != nullptr);
    auto conv2d_fusion = prim_value_node->value()->cast<std::shared_ptr<mindspore::ops::Conv2DFusion>>();
    if (conv2d_fusion == nullptr) {
      MS_LOG(ERROR) << "the input of depthwiseConv2d is null";
      return false;
    }
    if (conv2d_fusion->GetAttr(ops::kIsDepthWise) == nullptr ||
        !GetValue<bool>(conv2d_fusion->GetAttr(ops::kIsDepthWise))) {
      continue;
    }
    auto data_node = conv_cnode->input(kConvInputIndex)->abstract();
    if (data_node == nullptr) {
      MS_LOG(ERROR) << "the node input is invalid.";
      return false;
    }
    auto data_shape = utils::cast<abstract::ShapePtr>(data_node->GetShapeTrack())->shape();
    if (data_shape.empty()) {
      MS_LOG(DEBUG) << "the tensor's shape is dynamic.";
      return true;
    }
    auto weight_data_node = conv_cnode->input(kConvWeightIndex)->abstract();
    if (weight_data_node == nullptr) {
      MS_LOG(ERROR) << "the weight node input is invalid.";
      return false;
    }
    auto weight_shape = utils::cast<abstract::ShapePtr>(weight_data_node->GetShapeTrack())->shape();
    if (weight_shape.empty()) {
      MS_LOG(DEBUG) << "the weight's shape is dynamic.";
      return true;
    }
    if (data_shape[3] == 1 || data_shape[3] != weight_shape[3]) {
      conv2d_fusion->EraseAttr(ops::kIsDepthWise);
      conv2d_fusion->set_group(static_cast<int64_t>(data_shape[3]));
      conv2d_fusion->set_in_channel(data_shape[3]);
      MS_ASSERT(conv_cnode->inputs().size() > kConvWeightIndex);
      auto weight_node = conv_cnode->input(kConvWeightIndex);
      MS_ASSERT(weight_node != nullptr);
      auto weight_value = GetLiteParamValue(weight_node);
      if (weight_value == nullptr) {
        MS_LOG(ERROR) << "weight node must param value";
        return false;
      }
      MS_ASSERT(weight_value->tensor_type() == TypeId::kNumberTypeFloat32 ||
                weight_value->tensor_type() == TypeId::kNumberTypeInt8);
      lite::STATUS status;
      schema::Format weight_dst_format = schema::Format::Format_CHWK;
      weight_value->set_format(schema::Format_KHWC);
      status = TransFilterFormat(weight_value, weight_dst_format);
      if (status == RET_OK) {
        weight_value->set_format(weight_dst_format);
      } else {
        MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(schema::EnumValuesFormat()[weight_value->format()]) << "To"
                      << EnumNameFormat(weight_dst_format) << " failed, node : " << node->fullname_with_scope();
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
  }
  return true;
}
}  // namespace mindspore::opt
