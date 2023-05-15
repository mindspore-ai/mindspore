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

#include "plugin/device/ascend/optimizer/ge/convert_attr_to_input.h"

#include <memory>
#include <vector>
#include <map>
#include <utility>
#include <string>

#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {
namespace {
struct AttrToInputInfo {
  std::string to_node_type;
  std::vector<std::pair<std::string, string>> attr_input_info;
};
// old version node name | new version node name | {ori_attr_name, new_input_name}
static const std::map<std::string, AttrToInputInfo> kNeedTransMap = {
  {"ArgMaxV2", {"ArgMaxV2", {{"axis", "dimension"}}}},
  {"ArgMin", {"ArgMin", {{"axis", "dimension"}}}},
  {"ResizeNearestNeighborV2", {"ResizeNearestNeighborV2", {{"size", "size"}}}},
  {"ResizeNearestNeighborV2Grad", {"ResizeNearestNeighborV2Grad", {{"size", "size"}}}},
  {"ResizeBilinearV2", {"ResizeBilinearV2", {{"size", "size"}}}},
  {"CropAndResize", {"CropAndResize", {{"crop_size", "crop_size"}}}},
  {"Conv2DBackpropInput", {"Conv2DBackpropInput", {{"input_sizes", "input_size"}}}},
  {"Conv2DBackpropFilter", {"Conv2DBackpropFilter", {{"filter_sizes", "filter_size"}}}},
  {"Conv3DTranspose", {"Conv3DTranspose", {{"input_size", "input_size"}}}},
  {"Conv3DBackpropFilter", {"Conv3DBackpropFilter", {{"filter_size", "filter_size"}}}},
  {"AvgPoolGrad", {"AvgPoolGrad", {{"x_origin", "orig_input_shape"}}}},
  {"AvgPool3DGrad", {"AvgPool3DGrad", {{"origin_input_shape", "orig_input_shape"}}}},
  {"SparseApplyFtrlD", {"SparseApplyFtrl", {{"lr", "lr"}, {"l1", "l1"}, {"l2", "l2"}, {"lr_power", "lr_power"}}}},
  {"SparseApplyRMSPropD", {"SparseApplyRMSProp", {{"rho", "rho"}, {"momentum", "momentum"}, {"epsilon", "epsilon"}}}},
  {"SparseApplyAdagradD", {"SparseApplyAdagrad", {{"lr", "lr"}}}},
  {"SparseApplyAdagradV2D", {"SparseApplyAdagradV2", {{"lr", "lr"}, {"epsilon", "epsilon"}}}},
  {"ApplyAdamWithAmsgradD", {"ApplyAdamWithAmsgrad", {{"beta1", "beta1"}, {"beta2", "beta2"}, {"epsilon", "epsilon"}}}},
  {"ApplyAdagradV2D", {"ApplyAdagradV2", {{"epsilon", "epsilon"}}}},
  {"ApplyRMSPropD", {"ApplyRMSProp", {{"rho", "rho"}, {"momentum", "momentum"}, {"epsilon", "epsilon"}}}},
  {"SparseApplyAdadelta", {"SparseApplyAdadelta", {{"epsilon", "epsilon"}}}},
  {"SparseApplyFtrlV2D",
   {"SparseApplyFtrlV2",
    {{"lr", "lr"}, {"l1", "l1"}, {"l2", "l2"}, {"l2_shrinkage", "l2_shrinkage"}, {"lr_power", "lr_power"}}}},
  {"Pad", {"Pad", {{"paddings", "paddings"}}}},
  {"BroadcastTo", {"BroadcastTo", {{"shape", "shape"}}}},
  {"ReduceAny", {"ReduceAny", {{"axis", "axes"}}}},
  {"ReduceSum", {"ReduceSum", {{"axis", "axes"}}}},
  {"ReduceAll", {"ReduceAll", {{"axis", "axes"}}}},
  {"ReduceMean", {"ReduceMean", {{"axis", "axes"}}}},
  {"ReduceMin", {"ReduceMin", {{"axis", "axes"}}}},
  {"ReduceMax", {"ReduceMax", {{"axis", "axes"}}}},
  {"ReduceProd", {"ReduceProd", {{"axis", "axes"}}}},
  {"Cumsum", {"Cumsum", {{"axis", "axis"}}}},
  {"Cumprod", {"Cumprod", {{"axis", "axis"}}}},
  {"Tile", {"Tile", {{"multiples", "multiples"}}}},
  {"OneHot", {"OneHot", {{"depth", "depth"}}}},
  {"GatherV2", {"GatherV2", {{"axis", "axis"}}}},
  {"ScatterNd", {"ScatterNd", {{"shape", "shape"}}}},
  {"StridedSlice", {"StridedSlice", {{"begin", "begin"}, {"end", "end"}, {"strides", "strides"}}}},
  {"UnsortedSegmentSum", {"UnsortedSegmentSum", {{"num_segments", "num_segments"}}}},
  {"UnsortedSegmentProd", {"UnsortedSegmentProd", {{"num_segments", "num_segments"}}}},
  {"ReverseV2", {"ReverseV2", {{"axis", "axis"}}}},
  {"InplaceAddD", {"InplaceAdd", {{"indices", "indices"}}}},
  {"InplaceSubD", {"InplaceSub", {{"indices", "indices"}}}},
  {"InplaceUpdateD", {"InplaceUpdate", {{"indices", "indices"}}}},
  {"UnsortedSegmentMax", {"UnsortedSegmentMax", {{"num_segments", "num_segments"}}}},
  {"SplitD", {"Split", {{"axis", "split_dim"}}}},
  {"ConcatD", {"Concat", {{"axis", "concat_dim"}}}},
  {"SplitVD", {"SplitV", {{"size_splits", "size_splits"}, {"split_dim", "split_dim"}}}},
  {"SpaceToBatchND", {"SpaceToBatchND", {{"block_shape", "block_shape"}, {"paddings", "paddings"}}}},
  {"BatchToSpace", {"BatchToSpace", {{"crops", "crops"}}}},
  {"SpaceToBatch", {"SpaceToBatch", {{"paddings", "paddings"}}}},
  {"BatchToSpaceND", {"BatchToSpaceND", {{"block_shape", "block_shape"}, {"crops", "crops"}}}}};

bool NeedConvert(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    auto node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (!IsValueNode<Primitive>(node)) {
      return false;
    }
    auto prim = GetValuePtr<Primitive>(node);
    MS_EXCEPTION_IF_NULL(prim);
    return kNeedTransMap.find(prim->name()) != kNeedTransMap.cend();
  }
  return false;
}
}  // namespace

const BaseRef ConvertAttrToInput::DefinePattern() const {
  VarPtr convert = std::make_shared<CondVar>(NeedConvert);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({convert, inputs});
}

const AnfNodePtr ConvertAttrToInput::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &origin_prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(origin_prim);
  const auto &item = kNeedTransMap.at(origin_prim->name());
  std::string new_prim_name = item.to_node_type;
  auto attr_input_info = item.attr_input_info;
  const auto &origin_attrs = origin_prim->attrs();
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Create new primitive and inherit the origin attributes.
  auto new_prim = std::make_shared<Primitive>(new_prim_name);
  MS_EXCEPTION_IF_NULL(new_prim);
  (void)new_prim->SetAttrs(origin_attrs);
  MS_LOG(INFO) << "Begin to convert attr to input for node: " << node->DebugString() << "new name: " << new_prim_name;
  for (const auto &attr_item : attr_input_info) {
    if (origin_attrs.count(attr_item.first) == 0) {
      MS_LOG(WARNING) << "Origin primitive: " << origin_prim->name() << "has no attr : " << attr_item.first
                      << ", so only update to new primitive: " << new_prim_name;
    } else {
      // Convert the specific attr to input and erase the specific attr.
      auto attr_value = new_prim->GetAttr(attr_item.first);
      MS_EXCEPTION_IF_NULL(attr_value);
      auto new_value_node = std::make_shared<ValueNode>(attr_value);
      MS_EXCEPTION_IF_NULL(new_value_node);
      new_value_node->set_abstract(attr_value->ToAbstract());
      manager->AddEdge(cnode, new_value_node);
      new_prim->EraseAttr(attr_item.first);
    }
  }
  // Update the primitive of cnode
  manager->SetEdge(cnode, kIndex0, std::make_shared<ValueNode>(new_prim));
  cnode->set_fullname_with_scope("");
  MS_LOG(INFO) << "End, new node: " << node->DebugString();
  return node;
}
}  // namespace mindspore::opt
