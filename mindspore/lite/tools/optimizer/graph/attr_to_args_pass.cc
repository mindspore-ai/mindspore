/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/graph/attr_to_args_pass.h"
#include <utility>
#include "tools/common/node_util.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "ops/primitive_c.h"
#include "ops/base_operator.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
namespace {
static const std::map<std::string, std::vector<string>> kAttrMapNeedAdjust = {
  {"ArgMin",
   {
     "axis",
     "output_type",
   }},
  {"BroadcastTo",
   {
     "shape",
   }},
  {"ArgMaxV2",
   {
     "axis",
     "output_type",
   }},
  {"ArgMaxWithValue",
   {
     "axis",
     "keep_dims",
   }},
  {"AvgPool",
   {
     "kernel_size",
     "strides",
     "pad_mode",
     "data_format",
   }},
  {"StridedSlice",
   {
     "begin_mask",
     "end_mask",
     "ellipsis_mask",
     "new_axis_mask",
     "shrink_axis_mask",
   }},
  {"BatchNorm",
   {
     "is_training",
     "epsilon",
     "momentum",
     "data_format",
   }},
  {"FusedBatchNorm",
   {
     "is_training",
     "epsilon",
     "momentum",
     "data_format",
   }},
  {"Elu",
   {
     "alpha",
   }},
  {"Gather",
   {
     "batch_dims",
   }},
  {"LayerNorm",
   {
     "begin_norm_axis",
     "begin_params_axis",
     "epsilon",
   }},
  {"LayerNormV3",
   {
     "begin_norm_axis",
     "begin_params_axis",
     "epsilon",
   }},
  {"Range",
   {
     "maxlen",
   }},
  {"Concat",
   {
     "axis",
   }},
  {"ConcatV2",
   {
     "axis",
   }},
  {"CumSum",
   {
     "exclusive",
     "reverse",
   }},
  {"ReduceAll",
   {
     "keep_dims",
   }},
  {"ReduceMax",
   {
     "keep_dims",
   }},
  {"ReduceMin",
   {
     "keep_dims",
   }},
  {"ReduceMean",
   {
     "keep_dims",
   }},
  {"ReduceSum",
   {
     "keep_dims",
     "skip_mode",
   }},
  {"Split",
   {
     "axis",
     "output_num",
   }},
  {"ResizeBicubic",
   {
     "align_corners",
     "half_pixel_centers",
   }},
  {"ResizeBilinear",
   {
     "size",
     "align_corners",
     "half_pixel_centers",
   }},
  {"ResizeNearestNeighbor",
   {
     "size",
     "align_corners",
     "half_pixel_centers",
   }},
  {"ResizeBilinearV2",
   {
     "align_corners",
     "half_pixel_centers",
   }},
  {"ResizeNearestNeighborV2",
   {
     "align_corners",
     "half_pixel_centers",
   }},
  {"ReverseV2",
   {
     "axis",
   }},
  {"MatMul",
   {
     "transpose_a",
     "transpose_b",
   }},
  {"MatMulV2",
   {
     "transpose_a",
     "transpose_b",
   }},
  {"BatchMatMul",
   {
     "transpose_a",
     "transpose_b",
   }},
  {"Softmax",
   {
     "axis",
   }},
  {"FusedInferAttentionScore",
   {
     "num_heads",
     "scale_value",
     "pre_tokens",
     "next_tokens",
     "input_layout",
     "num_key_value_heads",
     "sparse_mode",
     "inner_precise",
     "block_size",
     "antiquant_mode",
     "softmax_lse_flag",
   }},
  {"GridSampler3D",
   {
     "interpolation_mode",
     "padding_mode",
     "align_corners",
   }},
  {"GridSampler2D",
   {
     "interpolation_mode",
     "padding_mode",
     "align_corners",
   }},
  {"WeightQuantBatchMatmul",
   {
     "transpose_x",
     "transpose_weight",
     "antiquant_group_size",
   }},
  {"QuantBatchMatmul",
   {
     "transpose_x1",
     "transpose_x2",
     "dtype",
   }},
  {"GroupedMatmul",
   {
     "split_item",
     "group_type",
   }},
  {"BinaryCrossEntropy",
   {
     "reduction",
   }},
};

constexpr size_t kMatMulInputSizeWithBias = 6;  // primitive, x1, x2, bias, transpose_a, transpose_b
constexpr auto kMatMulOpName = "MatMul";
constexpr auto kMatMulV2OpName = "MatMulV2";

void RearrangeBiasForMatMul(const FuncGraphManagerPtr &manager, const CNodePtr &cnode) {
  auto node_inputs = cnode->inputs();
  auto bias_add_node_it = node_inputs.begin() + kIndexThree;
  std::rotate(bias_add_node_it, bias_add_node_it + 1, node_inputs.end());
  cnode->set_inputs(node_inputs);
}

int ConvertAttrToArgsForNode(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  auto cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr");
  const auto &origin_prim = GetCNodePrimitive(node);
  MS_CHECK_TRUE_MSG(origin_prim != nullptr, RET_ERROR, "origin_prim is nullptr");
  auto prim_name = origin_prim->name();
  const auto &attrs_adjust = kAttrMapNeedAdjust.at(prim_name);
  const auto &origin_attrs = origin_prim->attrs();

  // Create new primitive and inherit the origin attributes.
  MS_LOG(INFO) << "Begin to convert Primitive to Primitive_Func for node: " << node->DebugString()
               << "new name: " << prim_name;
  for (const auto &attr : attrs_adjust) {
    if (origin_attrs.count(attr) == 0) {
      MS_LOG(INFO) << "Origin primitive: " << prim_name << " has no attribute : " << attr;
    } else {
      // Convert the specific attr to input and erase the specific attr.
      auto attr_value = origin_prim->GetAttr(attr);
      MS_CHECK_TRUE_MSG(attr_value != nullptr, RET_ERROR, "attr_value is nullptr");
      auto new_value_node = std::make_shared<ValueNode>(attr_value);
      MS_CHECK_TRUE_MSG(new_value_node != nullptr, RET_ERROR, "new_value_node is nullptr");
      new_value_node->set_abstract(attr_value->ToAbstract());
      manager->AddEdge(cnode, new_value_node);
    }
  }

  if ((prim_name == kMatMulOpName || prim_name == kMatMulV2OpName) &&
      cnode->inputs().size() == kMatMulInputSizeWithBias) {
    RearrangeBiasForMatMul(manager, cnode);
  }
  MS_LOG(INFO) << "End, new node: " << node->DebugString();
  return RET_OK;
}
}  // namespace

bool AttrToArgsPass::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return false;
  }

  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "get func graph manager is nullptr";
    return false;
  }

  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      continue;
    }
    if (kAttrMapNeedAdjust.find(prim->name()) == kAttrMapNeedAdjust.end()) {
      continue;
    }
    if (ConvertAttrToArgsForNode(node, manager) != RET_OK) {
      MS_LOG(ERROR) << "Convert attr to args for node " << node->fullname_with_scope() << "failed.";
      return false;
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
