/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/input_data_type_trans_pass.h"
#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "ops/cast.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "src/common/utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kNotEqualMinIndex = 3;
const std::vector<TypeId> kFloatDataType = {kNumberTypeFloat, kNumberTypeFloat16, kNumberTypeFloat32,
                                            kNumberTypeFloat64};
const std::vector<TypeId> kIntDataType = {kNumberTypeInt, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};

STATUS CastDataType(const FuncGraphPtr &graph, const AnfNodePtr &node, TypeId in_data_type, TypeId out_data_type) {
  MS_ASSERT(graph != nullptr && node != nullptr);
  if (in_data_type == out_data_type) {
    return RET_OK;
  }
  bool need_cast = (std::find(kFloatDataType.begin(), kFloatDataType.end(), in_data_type) != kFloatDataType.end() &&
                    std::find(kFloatDataType.begin(), kFloatDataType.end(), out_data_type) != kFloatDataType.end()) ||
                   (std::find(kIntDataType.begin(), kIntDataType.end(), in_data_type) != kIntDataType.end() &&
                    std::find(kIntDataType.begin(), kIntDataType.end(), out_data_type) != kIntDataType.end());
  if (need_cast) {
    auto abstract = node->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "The abstract is nullptr.");
    if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
      MS_LOG(ERROR) << "abstract is not AbstractTensor";
      return RET_ERROR;
    }
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
    MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed.");
    auto element = abstract_tensor->element();
    MS_CHECK_TRUE_MSG(element != nullptr, RET_ERROR, "The element of abstract tensor is nullptr.");
    element->set_type(TypeIdToType(in_data_type));
    auto new_abstract = abstract->Clone();
    MS_CHECK_TRUE_MSG(new_abstract != nullptr, RET_ERROR, "Clone abstract failed!");
    new_abstract->set_value(std::make_shared<ValueAny>());
    if (GenCastNode(graph, node, node->fullname_with_scope() + "_post_cast", out_data_type, new_abstract) == nullptr) {
      MS_LOG(ERROR) << "GenCastNode failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace

STATUS InOutDTypeTransPass::HandleGraphInput(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto graph_inputs = graph->get_inputs();
  for (const auto &input : graph_inputs) {
    TypeId input_data_type;
    if (GetDataTypeFromAnfNode(input, &input_data_type) != RET_OK) {
      MS_LOG(ERROR) << "get input node data type failed for node: " << input->fullname_with_scope();
      return RET_ERROR;
    }
    // convert int64 to int32 and float64 to float32 by default.
    if (input_data_type == kNumberTypeInt64 &&
        std::find(kIntDataType.begin(), kIntDataType.end(), dst_input_data_type_) == kIntDataType.end()) {
      dst_input_data_type_ = kNumberTypeInt32;
    }
    if (input_data_type == kNumberTypeFloat64 &&
        std::find(kFloatDataType.begin(), kFloatDataType.end(), dst_input_data_type_) == kFloatDataType.end()) {
      dst_input_data_type_ = kNumberTypeFloat32;
    }
    if (CastDataType(graph, input, dst_input_data_type_, input_data_type) != RET_OK) {
      MS_LOG(ERROR) << "Cast input data type failed for node: " << input->fullname_with_scope();
      return RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS InOutDTypeTransPass::HandleGraphOutput(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto return_node = graph->get_return();
  MS_CHECK_TRUE_RET(return_node != nullptr, RET_ERROR);
  bool has_make_tuple = false;
  std::vector<AnfNodePtr> graph_outputs;
  if (lite::GetFlattenInputsIfMakeTuple(return_node, &graph_outputs, &has_make_tuple) != RET_OK) {
    MS_LOG(ERROR) << "Get graph output nodes failed.";
    return RET_ERROR;
  }
  for (const auto &output : graph_outputs) {
    MS_CHECK_TRUE_RET(output != nullptr, RET_ERROR);
    if (!output->isa<CNode>() || IsMonadNode(output) || opt::CheckPrimitiveType(output, prim::kPrimUpdateState) ||
        opt::CheckPrimitiveType(output, prim::kPrimDepend) || opt::CheckPrimitiveType(output, prim::kPrimLoad)) {
      continue;
    }
    TypeId output_data_type;
    if (GetDataTypeFromAnfNode(output, &output_data_type) != RET_OK) {
      MS_LOG(ERROR) << "get input node data type failed for node: " << output->fullname_with_scope();
      return RET_ERROR;
    }
    if (CastDataType(graph, output, output_data_type, dst_output_data_type_) != RET_OK) {
      MS_LOG(ERROR) << "Cast input data type failed for node: " << output->fullname_with_scope();
      return RET_ERROR;
    }
  }
  return lite::RET_OK;
}

bool InOutDTypeTransPass::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = Manage(graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  MS_LOG(INFO) << "The input data type will unified to " << dst_input_data_type_
               << " and the output data type will unified to " << dst_output_data_type_;
  if (HandleGraphInput(graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "transfer graph input format from nhwc to nchw failed.";
    return false;
  }
  if (HandleGraphOutput(graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "transfer graph input format from nhwc to nchw failed.";
    return false;
  }
  return true;
}
}  // namespace mindspore::opt
