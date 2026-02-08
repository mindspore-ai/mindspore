/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "backend/ge_backend/pass/adaptive_max_pool2d_check.h"
#include <memory>
#include <vector>
#include <string>
#include "mindspore/ops/op_def/conv_pool_ops.h"
#include "include/backend/common/pass_manager/helper.h"
#include "abstract/dshape.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/utils/anfalgo.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/value.h"
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace opt {
std::vector<std::string> AdaptiveMaxPool2DGeCheck::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimAdaptiveMaxPool2D->name()};
  return ret;
}

const BaseRef AdaptiveMaxPool2DGeCheck::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  return VectorRef({prim::kPrimAdaptiveMaxPool2D, X, Y});
}

const bool IsDynamicShapeGe(const CNodePtr &adaptive_max_pool2d) {
  auto origin_primitive = GetCNodePrimitive(adaptive_max_pool2d);
  MS_EXCEPTION_IF_NULL(origin_primitive);
  if (common::AnfAlgo::IsDynamicShape(adaptive_max_pool2d)) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass, due to dynamic attr:" << origin_primitive->GetAttrsText();
    return true;
  }

  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(adaptive_max_pool2d, kIndex0);
  if (IsDynamicShape(input_shape) || IsDynamicRank(input_shape)) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass due to dynamic input shape.";
    return true;
  }

  return false;
}

const AnfNodePtr AdaptiveMaxPool2DGeCheck::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto adaptive_max_pool2d = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(adaptive_max_pool2d);
  if (IsDynamicShapeGe(adaptive_max_pool2d)) {
    return node;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(adaptive_max_pool2d, kIndex0);
  if (input_shape.size() != kShape3dDims && input_shape.size() != kShape4dDims) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's input shape must equal to 3 or 4, but got " << input_shape.size();
  }

  auto output_size_node = common::AnfAlgo::GetInputNode(adaptive_max_pool2d, kIndex1);
  MS_EXCEPTION_IF_NULL(output_size_node);
  auto output_size_abstract = output_size_node->abstract();
  MS_EXCEPTION_IF_NULL(output_size_abstract);
  auto output_size_val = output_size_abstract->GetValue();
  MS_EXCEPTION_IF_NULL(output_size_val);
  const auto output_size_opt = GetArrayValue<int64_t>(output_size_val);
  if (!output_size_opt.has_value()) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass due to tuple value cannot get.";
    return node;
  }

  auto output_size_val_get = output_size_opt.value();
  if (output_size_val_get.IsValueUnknown(kDim0) || output_size_val_get.IsValueUnknown(kDim1)) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass due to tuple value unknown.";
    return node;
  }

  auto output_size = output_size_val_get;
  if (output_size.size() != kShape2dDims) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's output_size shape should equal to 2.";
  }

  if (output_size[kDim0] == -1 || output_size[kDim1] == -1) {
    size_t w_index = input_shape.size() - kIndex1;
    size_t h_index = input_shape.size() - kIndex2;
    int64_t height = input_shape.at(h_index);
    int64_t width = input_shape.at(w_index);
    int64_t output_h = (output_size[kDim0] == -1) ? height : output_size[kDim0];
    int64_t output_w = (output_size[kDim1] == -1) ? width : output_size[kDim1];
    if ((output_h != -1 && output_h <= 0) || (output_w != -1 && output_w <= 0)) {
      MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's output_size value is invalid.";
    }

    ShapeVector output_size_new = {output_h, output_w};
    auto output_size_new_node = kernel_graph->NewValueNode(MakeValue<std::vector<int64_t>>(output_size_new));
    output_size_new_node->set_abstract(output_size_node->abstract());
    output_size_new_node->set_scope(output_size_node->scope());
    auto manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->SetEdge(node, kIndex2, output_size_new_node);
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
