/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/slice_prepose_pass.h"
#include <vector>
#include <memory>
#include "mindspore/lite/include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "backend/optimizer/common/helper.h"
#include "src/ops/primitive_c.h"
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"

using mindspore::lite::PrimitiveC;
namespace mindspore::opt {
namespace {
std::vector<int32_t> GetCNodeInputShape(const CNodePtr &cnode, size_t index = 1) {
  MS_ASSERT(cnode != nullptr);
  std::vector<int32_t> empty_shape;
  if (index < 1 || cnode->inputs().size() <= index) {
    MS_LOG(ERROR) << "out of index";
    return empty_shape;
  }
  auto abstract = GetCNodeInputAbstract(cnode, index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Abstract of CNode is nullptr";
    return empty_shape;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
    MS_LOG(DEBUG) << "abstract is not AbstractTensor";
    return empty_shape;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
  if (!utils::isa<ParamValueLitePtr>(abstract_tensor->GetValueTrack())) {
    MS_LOG(DEBUG) << "Value of abstract is not ParamValueLite, indicate that infershape has failed";
    return empty_shape;
  }
  auto param_value_lite = utils::cast<ParamValueLitePtr>(abstract_tensor->GetValueTrack());
  if (param_value_lite == nullptr) {
    MS_LOG(ERROR) << "ParamValueLite of abstract is nullptr";
    return empty_shape;
  }
  return param_value_lite->tensor_shape();
}
}  // namespace

schema::SliceT *SlicePreposePass::GetSliceT(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    return nullptr;
  }
  auto primc = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  if (primc == nullptr) {
    return nullptr;
  }
  auto primt = primc->GetPrimitiveT();
  if (primt == nullptr || primt->value.AsSlice() == nullptr) {
    return nullptr;
  }
  return primt->value.AsSlice();
}

STATUS SlicePreposePass::SwapSliceWithPreceed(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                              const CNodePtr &preceed_cnode, const int index,
                                              const TransactionPtr &tr) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(preceed_cnode != nullptr);
  if (slice_cnode->input(1) != preceed_cnode) {
    MS_LOG(ERROR) << "preceed node must be slice node's direct parent";
    return RET_ERROR;
  }
  if (IsMultiOutputTensors(graph, preceed_cnode)) {
    MS_LOG(ERROR) << "preceed node referenced by multi nodes not support swap";
    return RET_ERROR;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return RET_ERROR;
  }
  auto node_users = manager->node_users()[slice_cnode];
  if (tr != nullptr) {  // do swap with transaction
    for (auto &node_user : node_users) {
      tr->SetEdge(node_user.first, node_user.second, preceed_cnode);
    }
    tr->SetEdge(slice_cnode, 1, preceed_cnode->input(index));
    tr->SetEdge(preceed_cnode, index, slice_cnode);
  } else {
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, preceed_cnode);
    }
    manager->SetEdge(slice_cnode, 1, preceed_cnode->input(index));
    manager->SetEdge(preceed_cnode, index, slice_cnode);
  }
  return RET_OK;
}

/*
 * Prepose condition:
 *  the softmax axis is not sliced
 */
bool SlicePreposePass::PreposeWithSoftmax(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                          const CNodePtr &softmax_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(softmax_cnode != nullptr);
  auto softmax_primc = GetValueNode<std::shared_ptr<PrimitiveC>>(softmax_cnode->input(0));
  if (softmax_primc == nullptr) {
    MS_LOG(ERROR) << "softmax_primc is nullptr";
    return false;
  }
  auto softmax_primt = softmax_primc->GetPrimitiveT();
  if (softmax_primt == nullptr || softmax_primt->value.AsSoftMax() == nullptr) {
    MS_LOG(ERROR) << "softmax_primt is nullptr";
    return false;
  }
  auto softmax_attr = softmax_primt->value.AsSoftMax();
  auto softmax_axis = softmax_attr->axis;
  auto shape = GetCNodeInputShape(softmax_cnode, 1);
  if (softmax_axis == -1) {
    if (shape.empty()) {  // when softmax axis == -1, shape info is needed to determine whether slice can be preposed
      return false;
    }
    softmax_axis += shape.size();
  }

  auto slice_t = GetSliceT(slice_cnode);
  MS_ASSERT(slice_t != nullptr);
  auto slice_axes = slice_t->axes;
  auto slice_begin = slice_t->begin;
  auto slice_size = slice_t->size;

  for (size_t i = 0; i < slice_axes.size(); ++i) {
    if (slice_axes[i] == softmax_axis) {
      if (slice_begin[i] != 0) {
        return false;
      }
      if (slice_size[i] != -1) {
        if (shape.empty() || slice_axes[i] >= static_cast<int>(shape.size())) {
          return false;
        }
        if (slice_size[i] < shape[slice_axes[i]]) {
          return false;
        }
      }
    }
  }
  auto status = SwapSliceWithPreceed(graph, slice_cnode, softmax_cnode, 1);
  return status == RET_OK;
}

bool SlicePreposePass::DoPrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                 const CNodePtr &preceed_cnode) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(slice_cnode != nullptr);
  MS_ASSERT(preceed_cnode != nullptr);
  auto preceed_node_type = GetCNodeType(preceed_cnode);
  switch (preceed_node_type) {
    case schema::PrimitiveType_SoftMax: {
      return PreposeWithSoftmax(graph, slice_cnode, preceed_cnode);
    }
    default: {
      MS_LOG(DEBUG) << "Node type " << preceed_node_type << " currently not support SlicePrepose";
    }
  }
  return false;
}

bool SlicePreposePass::Run(const FuncGraphPtr &graph) {
  if (fmk_type != lite::converter::FmkType_TF && fmk_type != lite::converter::FmkType_TFLITE) {
    MS_LOG(INFO) << "The framework type of model should be tf/tflite.";
    return false;
  }
  MS_ASSERT(graph != nullptr);
  bool changed = false;
  while (true) {
    bool this_time_changed = false;
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (node->func_graph() != graph) {
        continue;
      }
      if (!utils::isa<CNodePtr>(node) || GetCNodeType(node) != schema::PrimitiveType_Slice) {
        continue;
      }
      auto slice_cnode = node->cast<CNodePtr>();
      if (slice_cnode->inputs().size() != lite::kDoubleNum) {  // only support params from attrs now
        MS_LOG(INFO) << "SlicePrepose not support more than two inputs now";
        continue;
      }
      auto primt = GetSliceT(slice_cnode);
      if (primt == nullptr) {
        MS_LOG(ERROR) << "primitive_t of slice is nullptr";
        continue;
      }
      auto preceed_node = slice_cnode->input(1);
      if (preceed_node == nullptr) {
        MS_LOG(ERROR) << "preceed node is nullptr";
        continue;
      }
      auto output_tensor_num = GetOutputTensorNum(preceed_node);
      if (output_tensor_num > 1) {
        continue;
      }
      auto output_node_list = GetRealNodeUsedList(graph, utils::cast<AnfNodePtr>(preceed_node));
      if (output_node_list->size() > 1) {  // referenced by multi nodes
        continue;
      } else {
        if (utils::isa<ParameterPtr>(preceed_node)) {
          /*
           * if preceed_node is parameter without default param, it's input placeholder, so we can't prepose
           * if preceed_node is parameter with default param, constant_folding will process it
           */
          continue;
        }
        auto preceed_cnode = preceed_node->cast<CNodePtr>();
        if (preceed_cnode == nullptr) {
          MS_LOG(ERROR) << "preceed_cnode is nullptr";
          continue;
        }
        if (DoPrepose(graph, slice_cnode, preceed_cnode)) {
          this_time_changed = true;
          break;
        }
      }
    }
    if (this_time_changed) {
      changed = true;
    } else {
      break;
    }
  }
  return changed;
}
}  // namespace mindspore::opt
