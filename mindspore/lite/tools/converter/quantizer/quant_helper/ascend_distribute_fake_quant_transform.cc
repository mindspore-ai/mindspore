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

#include "tools/converter/quantizer/quant_helper/ascend_distribute_fake_quant_transform.h"
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
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/quantizer/smooth_quant.h"

namespace mindspore::lite::quant {
namespace {
constexpr int kMinIndex = 1;
constexpr int kMaxIndex = 2;
}  // namespace

std::vector<schema::QuantParamT> AscendDistributeFakeQuantTransform::CalQuantParam(const tensor::TensorPtr &min_value,
                                                                                   const tensor::TensorPtr &max_value,
                                                                                   bool symmetric) {
  std::vector<schema::QuantParamT> quant_params;
  // Ascend fake quant transform support PerLayer && PerChannel quant param
  if (min_value->ElementsNum() != max_value->ElementsNum()) {
    MS_LOG(ERROR) << "min value size not equal max value size";
    return {};
  }
  int size = min_value->ElementsNum();
  auto min_data = reinterpret_cast<float *>(min_value->data_c());
  auto max_data = reinterpret_cast<float *>(max_value->data_c());
  for (int i = 0; i < size; i++) {
    float real_min = *(min_data + i);
    float real_max = *(max_data + i);
    schema::QuantParamT quant_param;
    int bit_num = k8Bit;

    MS_LOG(DEBUG) << "min: " << real_min << " max: " << real_max << " bit_num: " << bit_num << " symmetric"
                  << symmetric;
    auto ret = CalQuantizationParams(&quant_param, real_min, real_max, bit_num, symmetric);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to calculate quant params";
      return {};
    }
    quant_params.push_back(quant_param);
  }
  return quant_params;
}

std::vector<schema::QuantParamT> AscendDistributeFakeQuantTransform::GetQuantParamWithFakeQuantNode(
  const CNodePtr &fake_quant_node, bool symmetric) {
  tensor::TensorPtr min_value;
  tensor::TensorPtr max_value;
  auto min_input = fake_quant_node->input(kMinIndex + kPrimOffset);
  if (utils::isa<ParameterPtr>(min_input) && min_input->cast<ParameterPtr>()->has_default() &&
      min_input->cast<ParameterPtr>()->default_param() != nullptr) {
    min_value = min_input->cast<ParameterPtr>()->default_param()->cast<tensor::TensorPtr>();
  } else {
    MS_LOG(ERROR) << "Quant param get min value failed";
    return {};
  }
  auto max_input = fake_quant_node->input(kMaxIndex + kPrimOffset);
  if (utils::isa<ParameterPtr>(max_input) && max_input->cast<ParameterPtr>()->has_default() &&
      max_input->cast<ParameterPtr>()->default_param() != nullptr) {
    max_value = max_input->cast<ParameterPtr>()->default_param()->cast<tensor::TensorPtr>();
  } else {
    MS_LOG(ERROR) << "Quant param get max value failed";
    return {};
  }
  auto quant_params = CalQuantParam(min_value, max_value, symmetric);
  return quant_params;
}

int AscendDistributeFakeQuantTransform::RemoveWeightRedundantNode(const FuncGraphPtr &func_graph,
                                                                  const CNodePtr &cnode) {
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  // Remove Weight Parameter-Cast Node
  auto weight_node = cnode->input(kWeightIndex + kPrimOffset);
  if (opt::CheckPrimitiveType(weight_node, prim::kPrimCast)) {
    auto weight_cnode = weight_node->cast<CNodePtr>();
    MS_LOG(INFO) << "Remove Cast Node: " << weight_cnode->fullname_with_scope();
    auto success = manager->Replace(weight_node, weight_cnode->input(kInputIndex + kPrimOffset));
    if (!success) {
      MS_LOG(ERROR) << "Fail to remove Cast node";
      return RET_ERROR;
    }
  }

  // Remove Matmul Weight Load Node
  weight_node = cnode->input(kWeightIndex + kPrimOffset);
  if (opt::CheckPrimitiveType(weight_node, prim::kPrimLoad)) {
    auto weight_cnode = weight_node->cast<CNodePtr>();
    MS_LOG(INFO) << "Remove Load Node: " << weight_cnode->fullname_with_scope();
    auto success = manager->Replace(weight_node, weight_cnode->input(kInputIndex + kPrimOffset));
    if (!success) {
      MS_LOG(ERROR) << "Fail to remove Load node";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
int AscendDistributeFakeQuantTransform::FetchWeightQuantParamFromFakeQuant(const FuncGraphPtr &func_graph) {
  const std::set<PrimitivePtr> fake_quant_types = {prim::kPrimFakeQuantPerLayer, prim::kPrimFakeQuantPerChannel};
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  auto quant_node_pass = QuantNodePass(func_graph);
  auto weight_quantizer = WeightQuantizer(param_);
  auto ret = RET_NO_CHANGE;
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    CHECK_NULL_RETURN(primitive);
    if (!CheckNodeInSet(cnode, fake_quant_types)) {
      continue;
    }
    int index = 0;
    auto input_node = cnode->input(index + kPrimOffset);
    if (!utils::isa<ParameterPtr>(input_node)) {
      continue;
    }
    // Store Quant Param
    auto quant_params = GetQuantParamWithFakeQuantNode(cnode, true);
    if (quant_params.empty()) {
      MS_LOG(ERROR) << "Fail to get quantParam with fakeQuantNode.";
      return RET_ERROR;
    }

    if (SetInputNodeQuantParam(cnode, kPrimOffset, quant_params) != RET_OK) {
      MS_LOG(ERROR) << "Failed to set weight quant param.";
      return RET_ERROR;
    }

    auto node_users = manager->node_users()[cnode];
    if (node_users.empty()) {
      MS_LOG(WARNING) << cnode->fullname_with_scope() << " cnode is isolated.";
      continue;
    }

    // Remove FakeQuant Node
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, input_node);
      auto quant_param_holder = GetCNodeQuantHolder(node_user.first->cast<CNodePtr>());
      CHECK_NULL_RETURN(quant_param_holder);
      quant_param_holder->set_input_quant_param(node_user.second - kPrimOffset, quant_params);
      ret = ConvertCNodeFp16ToFp32(node_user.first->cast<CNodePtr>());
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Fail to convert cnode fp16 to fp32";
        return ret;
      }
    }
    ret = RET_OK;
  }
  return ret;
}

int AscendDistributeFakeQuantTransform::SetWeightQuantParam(const FuncGraphPtr &func_graph) {
  const std::set<PrimitivePtr> fake_quant_types = {prim::kPrimFakeQuantPerLayer, prim::kPrimFakeQuantPerChannel};
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  auto quant_node_pass = QuantNodePass(func_graph);
  auto weight_quantizer = WeightQuantizer(param_);
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    CHECK_NULL_RETURN(primitive);
    if (!CheckNodeInSet(cnode, fake_quant_types)) {
      continue;
    }
    int index = 0;
    auto input_node = cnode->input(index + kPrimOffset);
    if (!utils::isa<ParameterPtr>(input_node)) {
      continue;
    }
    // Store Quant Param
    auto quant_params = GetQuantParamWithFakeQuantNode(cnode, false);
    if (quant_params.empty()) {
      MS_LOG(ERROR) << "Fail to get quantParam with fakeQuantNode.";
      return RET_ERROR;
    }

    if (SetInputNodeQuantParam(cnode, kPrimOffset, quant_params) != RET_OK) {
      MS_LOG(ERROR) << "Failed to set weight quant param.";
      return RET_ERROR;
    }

    auto ret = ConvertCNodeFp16ToFp32(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to convert cnode fp16 to fp32";
      return ret;
    }
    auto node_users = manager->node_users()[cnode];
    if (node_users.empty()) {
      MS_LOG(WARNING) << cnode->fullname_with_scope() << " cnode is isolated.";
      continue;
    }
    // Save weight quant axis in quant param
    tensor::TensorPtr input_tensor = quant::GetNodeTensor(input_node);
    CHECK_NULL_RETURN(input_tensor);
    auto quantization_params = input_tensor->quant_params();
    if (quantization_params.empty()) {
      MS_LOG(WARNING) << input_node->fullname_with_scope() << " quantization param is empty.";
      return {};
    }
    auto quantization_param = quantization_params.front();
    if (primitive->HasAttr(kChannelAxis)) {
      quantization_param->AddAttr(kChannelAxis, primitive->GetAttr(kChannelAxis));
    } else {
      MS_LOG(ERROR) << "fakequant node dont have kChannelAxis attr";
      return RET_ERROR;
    }

    // Remove FakeQuant Node
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, input_node);
    }

    const std::set<PrimitivePtr> support_weight_quant_types = {prim::kPrimMatMul, prim::kPrimBatchMatMul,
                                                               prim::kPrimGather};
    for (auto &node_user : node_users) {
      auto follow_cnode = node_user.first->cast<CNodePtr>();
      if (!CheckNodeInSet(follow_cnode, support_weight_quant_types)) {
        MS_LOG(INFO) << cnode->fullname_with_scope() << " of type: " << primitive->name() << " dont need weight quant.";
        continue;
      }

      ParameterPtr parameter;
      tensor::TensorPtr weight;
      GetParameterAndTensor(input_node, &parameter, &weight);
      int preferred_dim = GetValue<int64_t>(primitive->GetAttr(kChannelAxis));
      MS_LOG(INFO) << "cnode : " << cnode->fullname_with_scope() << " preferred_dim is: " << preferred_dim;

      auto status = quant_node_pass.QuantFilter(parameter, weight, quant_params, preferred_dim);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "QuantFilter failed : " << status;
        return status;
      }

      // insert Ascend AntiQuant Node
      TypeId type_id;
      if (opt::GetDataTypeFromAnfNode(follow_cnode, &type_id) != RET_OK) {
        MS_LOG(WARNING) << follow_cnode->fullname_with_scope() << " Get data type failed.";
        return RET_NO_CHANGE;
      }
      InsertQuantNodeManager quant_manager;
      status = quant_manager.InsertAscendAntiQuantNode(func_graph, follow_cnode, node_user.second, kNumberTypeInt8,
                                                       type_id, preferred_dim, param_->chip_name);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertAscendDequantNode failed : " << status;
        return status;
      }
      break;
    }
  }
  return RET_OK;
}

int AscendDistributeFakeQuantTransform::SetInputQuantParam(const FuncGraphPtr &func_graph) {
  const std::set<PrimitivePtr> support_gold_stick_quant_types = {prim::kPrimMatMulFusion, prim::kPrimMatMul,
                                                                 prim::kPrimBatchMatMul};
  const std::set<PrimitivePtr> communication_types = {prim::kPrimStridedSlice, prim::kPrimAllGather,
                                                      prim::kPrimAllReduce, prim::kPrimSplit};
  const std::set<PrimitivePtr> fake_quant_types = {prim::kPrimFakeQuantPerLayer, prim::kPrimFakeQuantPerChannel};
  auto manager = func_graph->manager();
  CHECK_NULL_RETURN(manager);
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto op_name = cnode->fullname_with_scope();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    CHECK_NULL_RETURN(primitive);

    if (!CheckNodeInSet(cnode, support_gold_stick_quant_types)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " of type: " << primitive->name() << " dont need weight quant.";
      continue;
    }

    int index = kInputIndex;
    auto input_node = cnode->input(index + kPrimOffset);
    auto pre_cnode = cnode;
    if (!utils::isa<CNodePtr>(input_node)) {
      continue;
    }

    auto fq_cnode = input_node->cast<mindspore::CNodePtr>();
    while (CheckNodeInSet(fq_cnode, communication_types)) {
      pre_cnode = fq_cnode;
      input_node = fq_cnode->input(kInputIndex + kPrimOffset);
      if (utils::isa<CNodePtr>(input_node)) {
        fq_cnode = input_node->cast<mindspore::CNodePtr>();
      } else {
        break;
      }
    }

    if (!CheckNodeInSet(fq_cnode, fake_quant_types)) {
      MS_LOG(INFO) << op_name << "dont have quant info, which save in fake quant node, it can not quant";
      continue;
    }

    // ascend fake quant transform not support matmul input is parameter
    if (fq_cnode->input(kInputIndex + kPrimOffset)->isa<Parameter>()) {
      MS_LOG(INFO) << "Ascend fake quant transform not support input[0] is Parameter. MatMul "
                   << cnode->fullname_with_scope() << " will not quant ";
      continue;
    }

    // Save cnode dtype to origin_type attr
    TypeId type_id = kTypeUnknown;
    auto ret = opt::GetDataTypeFromAnfNode(cnode, &type_id);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << "Fetch DataType from cnode failed.";
      return ret;
    }
    if (type_id != kTypeUnknown) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " origin type is " << type_id;
      cnode->AddAttr("origin_type", MakeValue(static_cast<int>(type_id)));
    }

    auto quant_params = GetQuantParamWithFakeQuantNode(fq_cnode, false);
    if (quant_params.empty()) {
      MS_LOG(ERROR) << "Fail to get quantParam with fakeQuantNode.";
      return RET_ERROR;
    }

    // set cnode quant_type
    auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_MSG(cnode_primitive != nullptr, RET_NULL_PTR, "Primitive is nullptr.");
    cnode_primitive->AddAttr(quant::kQuantType, MakeValue(static_cast<int>(quant::QUANT_ALL)));

    // support quant param holder
    auto quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(quant_param_holder);
    quant_param_holder->set_quant_type(quant::QUANT_ALL);
    quant_param_holder->set_input_quant_param(0, quant_params);

    MS_LOG(INFO) << "Remove FakeQuant Node: " << pre_cnode->fullname_with_scope();
    bool success = manager->Replace(input_node, fq_cnode->input(kInputIndex + kPrimOffset));
    if (!success) {
      MS_LOG(ERROR) << "Fail to remove FakeQuant node";
      return RET_ERROR;
    }

    ret = RemoveWeightRedundantNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to Remove Weight RedundantNode";
      return ret;
    }
  }
  return RET_OK;
}

int AscendDistributeFakeQuantTransform::InsertAscendQuantDeQuantNode(const FuncGraphPtr &func_graph) {
  // Insert QuantDtypeCast for matmul
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    lite::quant::QuantType curr_quant_type;
    if (GetQuantTypeNew(cnode, &curr_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type != lite::quant::QUANT_ALL) {
      continue;
    }
    lite::quant::InsertQuantNodeManager insert_node_manager;
    auto ret = insert_node_manager.InsertAscendDeQuantNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert AscendDeQuant node failed, cnode name: " << cnode->fullname_with_scope();
      return ret;
    }
    ret = insert_node_manager.InsertAscendQuantNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert AscendQuant node failed, cnode name: " << cnode->fullname_with_scope();
      return ret;
    }
    ret = UpdateDataType(cnode, kNumberTypeInt32);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Update datatype failed, cnode name: " << cnode->fullname_with_scope();
      return ret;
    }
    MS_LOG(INFO) << "Insert Qunat&DeQuant node: " << cnode->fullname_with_scope();
    for (size_t i = 1; i < cnode->size(); i++) {
      ret = RemoveInputNodeQuantParam(cnode, i);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Remove input node quant param failed, cnode name: " << cnode->fullname_with_scope();
        return ret;
      }
    }
  }
  return RET_OK;
}

int AscendDistributeFakeQuantTransform::MatMulWeightTranspose(const FuncGraphPtr &func_graph) {
  // Ascend device not support weight transpose is true
  lite::quant::InsertQuantNodeManager quant_manager;
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_CHECK_TRUE_MSG(cnode_primitive != nullptr, RET_NULL_PTR, "Primitive is nullptr.");
    if (!cnode_primitive->HasAttr(quant::kQuantType)) {
      continue;
    }
    auto quant_type_attr = cnode_primitive->GetAttr(quant::kQuantType);
    MS_CHECK_TRUE_MSG(quant_type_attr != nullptr, RET_NULL_PTR, "quant_type attr not exist.");
    auto quant_type = static_cast<quant::QuantType>(GetValue<int32_t>(quant_type_attr));

    if (quant_type != quant::QUANT_WEIGHT && quant_type != quant::QUANT_ALL) {
      MS_LOG(DEBUG) << "Invalid quant type, dont need transpose weight.";
      continue;
    }
    auto ret = quant_manager.AdjustTransposeNodeForSingleMatMulNode(func_graph, cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " Adjust Transpose Node failed.";
      return ret;
    }
  }
  return RET_OK;
}

int AscendDistributeFakeQuantTransform::NeedAscendDistributeFakeQuantTransform(const FuncGraphPtr &func_graph) {
  const std::set<PrimitivePtr> fake_quant_types = {prim::kPrimFakeQuantPerLayer, prim::kPrimFakeQuantPerChannel};
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto op_name = cnode->fullname_with_scope();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    CHECK_NULL_RETURN(primitive);
    if (CheckNodeInSet(cnode, fake_quant_types)) {
      return true;
    }
  }
  return false;
}
int AscendDistributeFakeQuantTransform::PreProcess(const FuncGraphPtr &func_graph) {
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
  // To adapt lite split tools, the weight quant will not do matmuladdfusion
  if (param_->weightQuantParam.dequant_strategy != ON_THE_FLY) {
    convert_pm->AddPass(std::make_shared<opt::MatMulAddFusion>());
  }
  optimizer->AddPassManager(convert_pm);
  if (optimizer->Optimize(func_graph) == nullptr) {
    MS_LOG(ERROR) << "run graph convert pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
int AscendDistributeFakeQuantTransform::DoSingleGraphAscendDistributeFakeQuantTransform(
  const FuncGraphPtr &func_graph) {
  if (!NeedAscendDistributeFakeQuantTransform(func_graph)) {
    MS_LOG(INFO) << "it dont need AscendDistributeFakeQuantTransform";
    return RET_OK;
  }

  auto ret = PreProcess(func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Pre process failed.";
    return ret;
  }

  // Set Weight Node quant Param
  if (param_->weightQuantParam.dequant_strategy == ON_THE_FLY) {
    ret = SetWeightQuantParam(func_graph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to SetWeightQuantParam";
      return ret;
    }
    return RET_OK;
  }

  // Set input quant param
  ret = SetInputQuantParam(func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to SetInputQuantParam";
    return ret;
  }

  ret = FetchWeightQuantParamFromFakeQuant(func_graph);
  if (ret == RET_OK) {
    ret = MatMulWeightTranspose(func_graph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to MatMulWeightTranspose";
      return ret;
    }
  } else if (ret == RET_NO_CHANGE) {
    ret = MatMulWeightTranspose(func_graph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to MatMulWeightTranspose";
      return ret;
    }
    // only static in the future.
    std::shared_ptr<ConverterPara> param;
    auto qat_transform = quant::QATTransform(func_graph, param);
    std::set<PrimitivePtr> per_channel_primitive_types = {prim::kPrimMatMul, prim::kPrimBatchMatMul};
    ret = qat_transform.StaticWeightQuantInfo(func_graph, per_channel_primitive_types);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Static weight quantization info failed.";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << " Fetch weight quant param from FQ node failed.";
    return ret;
  }

  MS_LOG(INFO) << "Start smooth quant.";
  SmoothQuant sq;
  ret = sq.Run(func_graph, param_->fullQuantParam.smooth_alpha, param_->fullQuantParam.enable_smooth_shift);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run smooth quant failed.";
    return ret;
  }
  MS_LOG(INFO) << "End smooth quant.";
  auto quant_node_pass = QuantNodePass(func_graph);
  ret = quant_node_pass.Quant();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run quantization node pass failed.";
    return ret;
  }

  ret = InsertAscendQuantDeQuantNode(func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Insert AscendQuant&DeQuant node failed.";
    return ret;
  }
  return RET_OK;
}

int AscendDistributeFakeQuantTransform::Transform() {
  std::set<FuncGraphPtr> all_func_graphs{};
  GetFuncGraphs(func_graph_, &all_func_graphs);
  // Support for multi-subgraph models
  for (auto &item : all_func_graphs) {
    auto status = DoSingleGraphAscendDistributeFakeQuantTransform(item);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do AscendDistributeFakeQuantTransform failed.";
      return status;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
