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
#include "tools/optimizer/graph/weight_format_transform_pass.h"
#include <memory>
#include <algorithm>
#include <vector>
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
constexpr size_t kConvWeightIndex = 2;
}  // namespace
void WeightFormatTransformPass::SetQuantType(QuantType type) { this->quant_type = type; }
void WeightFormatTransformPass::SetFmkType(FmkType type) { this->fmk_type = type; }
void WeightFormatTransformPass::SetDstFormat(schema::Format format) { this->dst_format = format; }
lite::STATUS WeightFormatTransformPass::ConvWeightFormatTrans(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto type = opt::GetCNodeType(node);
    if (type != schema::PrimitiveType_Conv2D && type != schema::PrimitiveType_DepthwiseConv2D &&
        type != schema::PrimitiveType_Conv2DGradInput && type != schema::PrimitiveType_GroupConv2DGradInput &&
        type != schema::PrimitiveType_DeConv2D && type != schema::PrimitiveType_DeDepthwiseConv2D) {
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
