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
#include <vector>
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

bool QuantStrategy::CheckAscendSpec(const FuncGraphManagerPtr &manager, const CNodePtr &cnode, TypeId type_id,
                                    int min_quant_weight_channel) {
  int ret;

  if (type_id == kNumberTypeFloat16) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimMatMulFusion)) {
      return false;
    }
    MS_LOG(INFO) << cnode->fullname_with_scope() << " will update to fp32";
    ret = lite::quant::ConvertCNodeFp16ToFp32(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Converter fp16 to fp32 failed.";
      return false;
    }
    ret = UpdateDataType(cnode, kNumberTypeFloat32);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " set new dtype failed.";
      return false;
    }
  } else if (type_id != kNumberTypeFloat32) {
    MS_LOG(WARNING) << " node name is " << cnode->fullname_with_scope() << ", type_id " << type_id
                    << " is not float32 and will not be quantified.";
    return false;
  }

  if (opt::CheckPrimitiveType(cnode, prim::kPrimMatMulFusion)) {
    auto prim_ptr = GetCNodePrimitive(cnode);
    // Don't support transpose.
    CHECK_NULL_RETURN(prim_ptr);
    auto transpose_a = prim_ptr->GetAttr(mindspore::ops::kTransposeA);
    auto transpose_b = prim_ptr->GetAttr(mindspore::ops::kTransposeB);
    if (transpose_a != nullptr && GetValue<bool>(transpose_a)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " transposeA is true.";
      return false;
    }
    if (transpose_b != nullptr && GetValue<bool>(transpose_b)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " transposeB is true.";
      return false;
    }

    auto weight = cnode->input(kWeightIndex + kPrimOffset);
    // activation
    if (weight->isa<CNode>()) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " both activation.";
      return false;
    } else {
      // shared weight
      auto node_map = manager->node_users();
      auto node_user = node_map[weight];
      if (node_user.size() > 1) {
        MS_LOG(INFO) << weight->fullname_with_scope() << " is shared.";
        return false;
      }
    }
    return true;
  }

  static const std::set<PrimitivePtr> check_channel_ops = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion};

  size_t min_rank = 3;
  if (CheckNodeInSet(cnode, check_channel_ops) && cnode->size() >= min_rank) {
    auto weight = cnode->input(kWeightIndex + kPrimOffset);
    auto abstract = weight->abstract();
    MS_CHECK_TRUE_RET(abstract != nullptr, false);
    std::vector<int64_t> weight_shape;
    ret = opt::FetchShapeFromAbstract(abstract, &weight_shape);
    if (ret != RET_OK) {
      MS_LOG(INFO) << "Dynamic Shape.";
      return true;
    }
    if (weight_shape[0] < static_cast<int>(min_quant_weight_channel)) {
      MS_LOG(WARNING) << weight->fullname_with_scope() << " preferred_dim shape:" << weight_shape[0]
                      << " less min_quant_weight_channel_ " << min_quant_weight_channel << " will not quant.";
      return false;
    }
  }
  return true;
}

bool QuantStrategy::CanOpFullQuantized(const FuncGraphManagerPtr &manager, const CNodePtr &cnode,
                                       const std::set<PrimitivePtr> &support_int8_ops,
                                       const std::set<PrimitivePtr> &skip_check_dtype_ops,
                                       const std::set<mindspore::ActivationType> &support_activation) {
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  MS_CHECK_TRUE_RET(manager != nullptr, false);
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

  TypeId type_id;
  auto ret = opt::GetDataTypeFromAnfNode(cnode, &type_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fetch DataType from cnode failed.";
    return false;
  }

  // CheckAscendSpec
  if (target_device_ == ASCEND) {
    return CheckAscendSpec(manager, cnode, type_id, min_quant_weight_channel_);
  } else {
    bool is_data_type_fp32 = type_id == kNumberTypeFloat32;
    if (!is_data_type_fp32) {
      MS_LOG(WARNING) << " type:" << type << " node name is " << cnode->fullname_with_scope() << ", type_id " << type_id
                      << " is not float32 and will not be quantified.";
      return false;
    }
  }
  return true;
}

bool QuantStrategy::IsSkipOp(const std::string &skip_node_name) {
  return !(skip_node_.find(skip_node_name) == skip_node_.end());
}
}  // namespace mindspore::lite::quant
