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

#define USE_DEPRECATED_API

#include "tools/converter/quantizer/quant_strategy.h"
#include <set>
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::lite::quant {
bool QuantStrategy::CanTensorQuantized(const CNodePtr &cnode, const AnfNodePtr &input_node, int preferred_dim) {
  if (input_node == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized input is nullptr!";
    return false;
  }
  ParameterPtr param_node = nullptr;
  if (input_node->isa<Parameter>()) {
    param_node = input_node->cast<ParameterPtr>();
  }
  if (param_node == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized invalid param_node!";
    return false;
  }
  if (!param_node->has_default()) {
    MS_LOG(INFO) << "param_node don't has default.";
    return false;
  }
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(INFO) << "abstract is nullptr";
    return false;
  }
  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << param_node->name();
    return false;
  }
  auto weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  MS_ASSERT(weight_shape != nullptr);
  if (weight_shape.size() < DIMENSION_2D) {  // do not quant single dim tensors
    return false;
  }
  int total_shape_size = 1;
  auto ret = GetElementNumFromShape(ConvertShapeVectorToInt32(weight_shape), &total_shape_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get element num from shape failed.";
    return false;
  }
  if (total_shape_size < 0 || static_cast<size_t>(total_shape_size) <= min_quant_weight_size_) {
    MS_LOG(INFO) << "shape_size " << total_shape_size << " less min_quant_weight_size_ " << min_quant_weight_size_;
    return false;
  }

  static const std::set<PrimitivePtr> check_channel_ops = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion};

  if (CheckNodeInSet(cnode, check_channel_ops) && weight_shape.size() >= DIMENSION_2D &&
      weight_shape[preferred_dim] <= static_cast<int>(min_quant_weight_channel_)) {
    MS_LOG(INFO) << "preferred_dim shape:" << weight_shape[preferred_dim] << " less min_quant_weight_channel_ "
                 << min_quant_weight_channel_;
    return false;
  }
  return true;
}

bool QuantStrategy::CanOpFullQuantized(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_int8_ops,
                                       const std::set<PrimitivePtr> &skip_check_dtype_ops,
                                       const std::set<mindspore::ActivationType> &support_activation) {
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  // The return node does not need to be quantified.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimReturn) || opt::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
    return false;
  }
  auto type = NodePrimitiveType(cnode);
  if (!support_int8_ops.empty() && !CheckNodeInSet(cnode, support_int8_ops)) {
    MS_LOG(WARNING) << "node:" << cnode->fullname_with_scope() << " type:" << type << " will not quantify.";
    return false;
  }
  if (CheckNodeInSet(cnode, skip_check_dtype_ops)) {
    MS_LOG(INFO) << " type:" << type << " node name is" << cnode->fullname_with_scope()
                 << " not need to check data type.";
    return true;
  }
  TypeId type_id;
  auto ret = opt::GetDataTypeFromAnfNode(cnode, &type_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fetch DataType from cnode failed.";
    return false;
  }

  bool is_data_type_fp32 = type_id == kNumberTypeFloat32;
  if (!is_data_type_fp32) {
    MS_LOG(WARNING) << " type:" << type << " node name is " << cnode->fullname_with_scope() << ", type_id " << type_id
                    << " is not float32 and will not be quantified.";
    return false;
  }

  // Check Activation
  if (!support_activation.empty() && opt::CheckPrimitiveType(cnode, prim::kPrimActivation)) {
    auto value_ptr = GetValueNode<PrimitivePtr>(cnode->input(0))->GetAttr(ops::kActivationType);
    if (value_ptr == nullptr) {
      return false;
    }
    auto activation = mindspore::ActivationType(GetValue<int64_t>(value_ptr));
    if (support_activation.find(activation) == support_activation.end()) {
      return false;
    }
  }
  return true;
}

bool QuantStrategy::IsSkipOp(const std::string &skip_node_name) {
  return !(skip_node_.find(skip_node_name) == skip_node_.end());
}
}  // namespace mindspore::lite::quant
