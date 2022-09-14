/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/dynamic_quantizer.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"

namespace mindspore::lite::quant {
int DynamicQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  // Dynamic dont support filters.
  param_->commonQuantParam.min_quant_weight_channel = 0;
  param_->commonQuantParam.min_quant_weight_size = 0;
  auto quantizer = WeightQuantizer(param_);
  const std::set<PrimitivePtr> support_weight_quant_nodes = {prim::kPrimMatMulFusion, prim::kPrimGather};
  const std::set<PrimitivePtr> symmetric_nodes = {prim::kPrimMatMulFusion};
  auto ret = quantizer.WeightQuant(func_graph, support_weight_quant_nodes, {}, symmetric_nodes, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight Quant failed.";
    return ret;
  }
  InsertQuantNodeManager manager;
  const std::set<PrimitivePtr> support_dynamic_quant_ops = {
    prim::kPrimMatMulFusion,
  };
  ret = manager.InsertDynamicQuantNode(func_graph, support_dynamic_quant_ops, param_->commonQuantParam.skip_quant_node);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Insert dynamic quant failed.";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
