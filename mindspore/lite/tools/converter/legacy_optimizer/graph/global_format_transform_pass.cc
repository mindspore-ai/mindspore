/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/legacy_optimizer/graph/global_format_transform_pass.h"
#include <algorithm>
#include "third_party/securec/include/securec.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "tools/common/node_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace {
std::vector<int> nchw2nhwc_perm = {0, 2, 3, 1};
std::vector<int> nhwc2nchw_perm = {0, 3, 1, 2};
}  // namespace
namespace lite {

STATUS GlobalFormatTransformPass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  std::set<size_t> need_del_nodes;
  std::set<size_t> need_trans_format_nodes;
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    auto type = node->primitive->value.type;
    if (type != PrimitiveType_Transpose) {
      continue;
    }
    if (GetTransposePerm(graph, node) != nchw2nhwc_perm) {
      continue;
    }
    std::vector<size_t> pre_nh2nc_nodes;
    std::vector<size_t> pre_not_trans_nodes;
    auto status = FindPreNh2NcNodes(graph, iter - graph->nodes.begin(), &pre_nh2nc_nodes, &pre_not_trans_nodes);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "GenNewScaleTensor failed: " << status;
      return status;
    }
    std::copy(pre_nh2nc_nodes.begin(), pre_nh2nc_nodes.end(), std::inserter(need_del_nodes, need_del_nodes.end()));
    std::copy(pre_not_trans_nodes.begin(), pre_not_trans_nodes.end(),
              std::inserter(need_trans_format_nodes, need_trans_format_nodes.end()));
    if (!pre_nh2nc_nodes.empty()) {
      need_del_nodes.insert(iter - graph->nodes.begin());
    }
  }
  if (need_del_nodes.empty()) {
    return RET_OK;
  }
  for (auto del_node_index : need_del_nodes) {
    auto node_name = graph->nodes.at(del_node_index)->name;
    auto status = IsolateOneWayNode(graph, del_node_index);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Isolate Node failed, node: " << node_name << ", error: " << status;
      return status;
    }
  }

  auto status = TransWeightToNhwc(graph, need_trans_format_nodes);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "trans weight to nhwc failed";
    return status;
  }
  return RET_OK;
}

STATUS ConvertNcTensor2Nh(TensorT *tensor, const std::vector<int> &pad_dims) {
  if (pad_dims.size() != 4) {
    MS_LOG(ERROR) << "pad dims error";
    return RET_ERROR;
  }
  auto batch = pad_dims[NCHW_N];
  auto channel = pad_dims[NCHW_C];
  auto area = pad_dims[NCHW_H] * pad_dims[NCHW_W];
  auto size = batch * channel * area;
  auto new_nhwc_data = new (std::nothrow) float[size];
  if (new_nhwc_data == nullptr) {
    MS_LOG(ERROR) << "create new nhwc data failed";
    delete[] new_nhwc_data;
    return RET_ERROR;
  }
  if (memset_s(new_nhwc_data, sizeof(float) * size, 0, sizeof(float) * size) != EOK) {
    MS_LOG(ERROR) << "create new nhwc data failed";
    delete[] new_nhwc_data;
    return RET_ERROR;
  }
  auto nchw_data = reinterpret_cast<float *>(tensor->data.data());
  // nchw to nhwc
  for (auto i = 0; i < batch; i++) {
    float *src_batch = nchw_data + i * channel * area;
    float *dst_batch = new_nhwc_data + i * channel * area;
    for (int j = 0; j < area; ++j) {
      float *src_area = src_batch + i;
      float *dst_area = dst_batch + i * channel;
      for (int k = 0; k < channel; ++k) {
        dst_area[k] = src_area[k * area];
      }
    }
  }
  if (memcpy_s(nchw_data, tensor->data.size(), new_nhwc_data, sizeof(float) * size) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    delete[] new_nhwc_data;
    return RET_ERROR;
  }
  delete[] new_nhwc_data;
  return RET_OK;
}

STATUS GlobalFormatTransformPass::TransWeightToNhwc(MetaGraphT *graph, const std::set<size_t> &pre_not_trans_nodes) {
  MS_ASSERT(graph != nullptr);
  if (pre_not_trans_nodes.empty()) {
    return RET_OK;
  }
  for (auto index : pre_not_trans_nodes) {
    auto &cur_node = graph->nodes.at(index);
    // need change axis from nchw to nhwc like concat,slice
    auto ret = ChangeOpAxis(graph, cur_node);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ChangeOpAxis error";
      return ret;
    }
    auto node_input_indexs = cur_node->inputIndex;
    for (auto input_index : node_input_indexs) {
      // weight data need trans nhwc layerout
      if (!IsContain(graph->inputIndex, input_index) &&
          graph->allTensors.at(input_index)->nodeType == NodeType_ValueNode) {
        auto &weight_tensor = graph->allTensors.at(input_index);
        auto origin_dims = weight_tensor->dims;
        weight_tensor->format = Format_NHWC;
        if (origin_dims.size() > 4) {
          MS_LOG(ERROR) << "tensor origin tensor size error";
          return RET_ERROR;
        }
        if (origin_dims.empty()) {
          continue;
        }
        auto pad_dims = origin_dims;
        if (origin_dims.size() == 1) {
          pad_dims = {1, 1, 1, origin_dims[0]};
        } else if (origin_dims.size() == 2) {
          pad_dims = {1, 1, origin_dims[0], origin_dims[1]};
        } else if (origin_dims.size() == 3) {
          pad_dims = {1, origin_dims[0], origin_dims[1], origin_dims[2]};
        }
        if (ConvertNcTensor2Nh(weight_tensor.get(), pad_dims) != RET_OK) {
          MS_LOG(ERROR) << "Convert nchw to nhwc failed";
          return RET_ERROR;
        }
        weight_tensor->dims = {pad_dims[NCHW_N], pad_dims[NCHW_H], pad_dims[NCHW_W], pad_dims[NCHW_C]};
      }
    }
  }
  return RET_OK;
}

STATUS GlobalFormatTransformPass::FindPreNh2NcNodes(MetaGraphT *graph, size_t nc2nh_index,
                                                    std::vector<size_t> *pre_nh2nc_nodes,
                                                    std::vector<size_t> *pre_not_trans_nodes) {
  MS_ASSERT(graph != nullptr);
  std::vector<size_t> bfs_queue = {nc2nh_index};
  // find pre node nh2nc start nodes
  while (!bfs_queue.empty()) {
    auto cur_node_index = bfs_queue.back();
    auto &cur_node = graph->nodes.at(cur_node_index);
    bfs_queue.pop_back();
    auto input_node_indexes = GetInputNodeIdx(*graph, *cur_node);
    for (auto input_node_index : input_node_indexes) {
      MS_ASSERT(graph->nodes.size() > input_node_index);
      auto &pre_node = graph->nodes.at(input_node_index);
      MS_ASSERT(pre_node != nullptr);
      auto node_type = pre_node->primitive->value.type;
      if (node_type == schema::PrimitiveType_Transpose && GetTransposePerm(graph, pre_node) == nhwc2nchw_perm) {
        if (!IsContain(*pre_nh2nc_nodes, input_node_index)) {
          pre_nh2nc_nodes->emplace_back(input_node_index);
        }
      } else if (IsContain(GetInsertOpList(), node_type)) {
        if (!IsContain(bfs_queue, input_node_index)) {
          bfs_queue.emplace_back(input_node_index);
        }
        auto pre_node_output_indexs = GetOutputNodeIdx(*graph, *pre_node);
        if (pre_node_output_indexs.size() != 1) {
          if (node_type == schema::PrimitiveType_Activation || node_type == schema::PrimitiveType_Concat) {
            pre_nh2nc_nodes->clear();
            pre_not_trans_nodes->clear();
            return RET_OK;
          }
          for (auto pre_node_output_index : pre_node_output_indexs) {
            MS_ASSERT(graph->nodes.size() > pre_node_output_index);
            if (graph->nodes.at(pre_node_output_index)->primitive->value.type == schema::PrimitiveType_PadFusion) {
              pre_nh2nc_nodes->clear();
              pre_not_trans_nodes->clear();
              return RET_OK;
            }
          }
        }
      } else {
        pre_nh2nc_nodes->clear();
        pre_not_trans_nodes->clear();
        return RET_OK;
      }
      if (!IsContain(*pre_not_trans_nodes, cur_node_index) && cur_node_index != nc2nh_index) {
        pre_not_trans_nodes->emplace_back(cur_node_index);
      }
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
