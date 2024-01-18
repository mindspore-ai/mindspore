/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/converter/quantizer/quant_helper/ffn_full_quant.h"
#include <memory>
#include <vector>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/backend/optimizer/graph_optimizer.h"
#include "nnacl/op_base.h"
#include "ops/tuple_get_item.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/import/remove_public_primitive.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/converter/quantizer/quant_helper/qat_transform.h"
#include "tools/converter/quantizer/quant_helper/quant_node_pass.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "tools/optimizer/fusion/matmul_add_fusion.h"
#include "tools/optimizer/graph/remove_load_pass.h"

namespace mindspore::lite::quant {
namespace {
constexpr int kFFNWeight1Index = 2;
constexpr int kFFNWeight2Index = 3;
constexpr int kFFNBias1Index = 5;
constexpr int kFFNBias2Index = 6;
};  // namespace

int FFNFullQuant::PreProcess(const FuncGraphPtr &func_graph) {
  auto remove_shared_primitve = RemovePublicPrimitiveInterference();
  if (!remove_shared_primitve.Run(func_graph)) {
    MS_LOG(ERROR) << "RemovePublicPrimitiveInterference";
    return RET_ERROR;
  }

  // Matmul Add Fusion
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto convert_pm = std::make_shared<opt::LitePassManager>("anf graph convert pass manager", true);
  CHECK_NULL_RETURN(convert_pm);
  convert_pm->AddPass(std::make_shared<opt::RemoveLoadPass>());
  optimizer->AddPassManager(convert_pm);
  if (optimizer->Optimize(func_graph) == nullptr) {
    MS_LOG(ERROR) << "run graph convert pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FFNFullQuant::DoWeightQuantWithFakeQuantNode(const FuncGraphPtr &func_graph, const CNodePtr ffn_cnode, int index) {
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  auto quant_node_pass = QuantNodePass(func_graph);

  auto fake_quant_cnode = ffn_cnode->input(index)->cast<CNodePtr>();
  const std::set<PrimitivePtr> fake_quant_types = {prim::kPrimFakeQuantPerLayer, prim::kPrimFakeQuantPerChannel};
  if (!CheckNodeInSet(fake_quant_cnode, fake_quant_types)) {
    return RET_ERROR;
  }

  auto primitive = GetValueNode<PrimitivePtr>(fake_quant_cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  if (!primitive->HasAttr(kChannelAxis)) {
    MS_LOG(ERROR) << "FakeQuantNode " << fake_quant_cnode->fullname_with_scope() << " dont have channel_axis attr";
    return RET_ERROR;
  }
  int preferred_dim = GetValue<int64_t>(primitive->GetAttr(kChannelAxis));
  MS_LOG(INFO) << "FakeQuantNode " << fake_quant_cnode->fullname_with_scope() << " 'channel_axis attr is "
               << preferred_dim;

  auto ret = ConvertCNodeFp16ToFp32(fake_quant_cnode);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to convert cnode fp16 to fp32";
    return ret;
  }
  auto weight_node = fake_quant_cnode->input(kPrimOffset);
  if (!utils::isa<ParameterPtr>(weight_node)) {
    MS_LOG(ERROR) << "weight node is not Parameter: " << weight_node->fullname_with_scope();
    return RET_ERROR;
  }
  auto quant_params = GetQuantParamWithFakeQuantNode(fake_quant_cnode, true);
  ParameterPtr parameter;
  tensor::TensorPtr weight;
  GetParameterAndTensor(weight_node, &parameter, &weight);
  auto status = quant_node_pass.QuantFilter(parameter, weight, quant_params, preferred_dim);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed : " << status;
    return status;
  }

  // Remove FakeQuant Node
  manager->SetEdge(ffn_cnode, index, weight_node);
  return RET_OK;
}

int FFNFullQuant::IsFullQuantNode(const CNodePtr &cnode) {
  const std::set<PrimitivePtr> fake_quant_types = {prim::kPrimFakeQuantPerLayer, prim::kPrimFakeQuantPerChannel};
  std::vector param_index_list = {kFFNWeight1Index, kFFNWeight2Index};
  for (auto param_index : param_index_list) {
    auto fake_quant_node = cnode->input(param_index)->cast<CNodePtr>();
    if (!CheckNodeInSet(fake_quant_node, fake_quant_types)) {
      return false;
    }
  }
  return true;
}

bool FFNFullQuant::CheckFFNNeedFullQuant(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node) || !opt::CheckPrimitiveType(node, prim::kPrimFFN)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsFullQuantNode(cnode)) {
      return true;
    }
  }
  return false;
}

int FFNFullQuant::DoSingleGraphFFNFullQuantTransform(const FuncGraphPtr &func_graph) {
  CHECK_NULL_RETURN(func_graph);
  if (!CheckFFNNeedFullQuant(func_graph)) {
    MS_LOG(INFO) << "Dont need DoSingleGraphFFNFullQuantTransform";
    return RET_OK;
  }

  auto ret = PreProcess(func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Pre process failed.";
    return ret;
  }

  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node) || !opt::CheckPrimitiveType(node, prim::kPrimFFN)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsFullQuantNode(cnode)) {
      return lite::RET_OK;
    }

    if (DoWeightQuantWithFakeQuantNode(func_graph, cnode, kFFNWeight1Index) != RET_OK) {
      MS_LOG(ERROR) << "Fail to Quant FFN weight1, FFN: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (DoWeightQuantWithFakeQuantNode(func_graph, cnode, kFFNWeight2Index) != RET_OK) {
      MS_LOG(ERROR) << "Fail to Quant FFN weight2, FFN: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    MS_LOG(INFO) << "Success to Quant FFN node: " << cnode->fullname_with_scope();
  }
  return RET_OK;
}

int FFNFullQuant::Transform() {
  std::set<FuncGraphPtr> all_func_graphs{};
  GetFuncGraphs(func_graph_, &all_func_graphs);
  // Support for multi-subgraph models
  for (auto &item : all_func_graphs) {
    auto status = DoSingleGraphFFNFullQuantTransform(item);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do AscendDistributeFakeQuantTransform failed.";
      return status;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
