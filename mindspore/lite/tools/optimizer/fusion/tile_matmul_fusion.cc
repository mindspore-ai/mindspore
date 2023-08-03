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
#include "tools/optimizer/fusion/tile_matmul_fusion.h"
#include <memory>
#include "mindspore/core/ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "tools/lite_exporter/fetch_content.h"

namespace mindspore {
namespace opt {
bool TileMatMulFusion::CheckCanFuse(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  auto tile_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(tile_cnode != nullptr, false);
  auto tile_primc = GetCNodePrimitive(tile_cnode);
  MS_CHECK_TRUE_RET(tile_primc != nullptr, false);
  if (IsQuantParameterNode(tile_primc)) {
    MS_LOG(INFO) << tile_primc->name() << " is quant node";
    return false;
  }
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto node_users = manager->node_users()[tile_cnode];
  for (auto &node_user : node_users) {
    auto post_node = node_user.first;
    auto post_node_index = node_user.second;
    if (!utils::isa<CNode>(post_node) || !CheckPrimitiveType(post_node, prim::kPrimMatMulFusion) ||
        post_node_index != C2NUM) {
      MS_LOG(INFO) << "The post node of tile must be matmul's matirxB.";
      return false;
    }
    auto matmul_primc = GetCNodePrimitive(post_node);
    MS_CHECK_TRUE_RET(matmul_primc != nullptr, false);
    if (IsQuantParameterNode(matmul_primc)) {
      MS_LOG(INFO) << matmul_primc->name() << " is quant node";
      return false;
    }
  }

  lite::DataInfo data_info;
  auto status = lite::FetchConstData(tile_cnode, C2NUM, converter::kFmkTypeMs, &data_info, true);
  MS_CHECK_TRUE_MSG(status == RET_OK, false, "Fetch tile_cnode third input's const data failed.");
  if ((data_info.data_type_ != kNumberTypeInt32 && data_info.data_type_ != kNumberTypeInt) ||
      data_info.data_.size() / sizeof(int) < DIMENSION_2D) {
    MS_LOG(INFO) << "Tile index data is invalid.";
    return false;
  }
  auto data = reinterpret_cast<int *>(data_info.data_.data());
  int dim = static_cast<int>(data_info.data_.size() / sizeof(int));
  for (int i = dim - C1NUM; i > dim - C3NUM; --i) {
    if (data[i] != C1NUM) {
      return false;
    }
  }
  lite::DataInfo weights_info;
  auto left_pre_node = tile_cnode->input(C1NUM);
  if (left_pre_node->isa<Parameter>() || left_pre_node->isa<ValueNode>()) {
    status = lite::FetchConstData(tile_cnode, C1NUM, converter::kFmkTypeMs, &weights_info, false);
  } else {
    status = lite::FetchDataFromCNode(tile_cnode, C1NUM, &weights_info);
  }
  MS_CHECK_TRUE_RET(status == RET_OK, false);
  MS_CHECK_TRUE_MSG(weights_info.shape_.size() == static_cast<size_t>(dim), false,
                    "Tile_cnode second input's shape size is invalid.");
  for (int i = 0; i < dim - C2NUM; i++) {
    if (data[i] != C1NUM && weights_info.shape_[i] != C1NUM) {
      return false;
    }
  }
  return true;
}

bool TileMatMulFusion::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimTileFusion)) {
      continue;
    }
    if (!CheckCanFuse(func_graph, node)) {
      continue;
    }
    auto tile_cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(tile_cnode != nullptr, false);
    auto left_pre_node = tile_cnode->input(SECOND_INPUT);
    auto manage = func_graph->manager();
    MS_CHECK_TRUE_RET(manage != nullptr, false);
    auto success = manage->Replace(tile_cnode, left_pre_node);
    MS_CHECK_TRUE_MSG(success, false, "Replace old node failed.");
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
