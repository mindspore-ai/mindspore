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

#include "tools/converter/quantizer/quant_strategy.h"
#include <set>
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "base/core_ops.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"

namespace mindspore::lite::quant {
bool QuantStrategy::CanTensorQuantized(const AnfNodePtr &input_node, int preferred_dim) {
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
  int64_t total_shape_size = 1;
  for (auto shape : weight_shape) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(total_shape_size, shape), RET_ERROR, "Int mul overflow");
    total_shape_size *= shape;
  }
  if (total_shape_size < 0 || static_cast<size_t>(total_shape_size) < min_quant_weight_size_) {
    MS_LOG(INFO) << "shape_size " << total_shape_size << " less min_quant_weight_size_ " << min_quant_weight_size_;
    return false;
  }

  // min_quant_weight_channel_ only supports convolution
  if (weight_shape.size() > DIMENSION_2D &&
      weight_shape[preferred_dim] <= static_cast<int>(min_quant_weight_channel_)) {
    MS_LOG(INFO) << "preferred_dim shape:" << weight_shape[preferred_dim] << " less min_quant_weight_channel_ "
                 << min_quant_weight_channel_;
    return false;
  }
  return true;
}

bool QuantStrategy::CanOpFullQuantized(const AnfNodePtr &node) {
  MS_CHECK_TRUE_RET(node != nullptr, false);
  if (!node->isa<mindspore::CNode>()) {
    return false;
  }
  const auto cnode = std::dynamic_pointer_cast<mindspore::CNode>(node);
  MS_ASSERT(cnode != nullptr);
  auto type = NodePrimitiveType(cnode);
  static const std::set<PrimitivePtr> support_int8_ops = {prim::kPrimAddFusion,     prim::kPrimActivation,
                                                          prim::kPrimAvgPoolFusion, prim::kPrimConcat,
                                                          prim::kPrimConv2DFusion,  prim::kPrimConv2dTransposeFusion,
                                                          prim::kPrimCrop,          prim::kPrimFullConnection,
                                                          prim::kPrimGather,        prim::kPrimLayerNormFusion,
                                                          prim::kPrimMatMul,        prim::kPrimMaxPoolFusion,
                                                          prim::kPrimMulFusion,     prim::kPrimReshape,
                                                          prim::kPrimSplit,         prim::kPrimTranspose,
                                                          prim::kPrimReduceFusion,  prim::kPrimDivFusion,
                                                          prim::kPrimSqrt,          prim::kPrimPowFusion,
                                                          prim::kPrimUnsqueeze,     prim::kPrimAffine};
  // The return node does not need to be quantified.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimReturn) || opt::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
    return false;
  }
  // These operators do not need to check the data type.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimShape) || opt::CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
    return true;
  }
  auto is_support_node = CheckNodeInSet(cnode, support_int8_ops);
  if (!is_support_node && type != "Eltwise") {
    MS_LOG(WARNING) << "node:" << cnode->fullname_with_scope() << " type:" << type << " is not support quantization.";
    return false;
  }
  TypeId type_id;
  auto ret = opt::GetDataTypeFromAnfNode(cnode, &type_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fetch DataType from cnode failed.";
    return false;
  }

  bool is_data_type_fp32 = type_id == kNumberTypeFloat32;
  if (!is_data_type_fp32) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << "  type_id is " << type_id << " , and is not float32.";
  }
  return is_data_type_fp32;
}

bool QuantStrategy::IsSkipOp(const AnfNodePtr &input_node) {
  if (skip_node_.find(input_node->fullname_with_scope()) == skip_node_.end()) {
    return false;
  }
  return true;
}
}  // namespace mindspore::lite::quant
