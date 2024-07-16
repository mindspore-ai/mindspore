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

#include "tools/converter/parser/onnx/onnx_quantize_linear_adjust.h"
#include <memory>
#include <utility>
#include <set>
#include "ops/primitive_c.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/fusion/scale_fusion.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "tools/converter/ops/ops_def.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "mindspore/core/ops/array_ops.h"

namespace mindspore::lite {
namespace {
const std::set<PrimitivePtr> kSelfInferOperator = {prim::kPrimTranspose};
}

void OnnxQuantizeLinearAdjust::RemoveDequantizeLinear(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto inputs = cnode->inputs();
  MS_CHECK_TRUE_RET_VOID(inputs.size() >= kIndex2);
  auto pre_cnode = inputs[1];
  MS_CHECK_PTR_IF_NULL(pre_cnode);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  MS_CHECK_PTR_IF_NULL(manager);
  auto node_users = manager->node_users()[cnode];
  if (node_users.empty()) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " cnode is isolated.";
    return;
  }
  // remove fake quant node
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, pre_cnode);
  }
}

bool OnnxQuantizeLinearAdjust::SetQuantParam(const CNodePtr &cnode, const QuantParamHolderPtr &quant_param_holder,
                                             bool is_next_node, size_t index) {
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, false, "Primitive quant params holder nullptr.");
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto scale_value = primitive->GetAttr(kAttrScale);
  auto scale_vec_value = primitive->GetAttr(kAttrScaleVec);
  auto zero_point_value = primitive->GetAttr(kAttrZeroPoint);
  auto zero_point_vec_value = primitive->GetAttr(kAttrZeroPointVec);
  if (scale_value != nullptr && zero_point_value != nullptr) {
    auto scale = GetValue<float>(scale_value);
    auto zero_point = GetValue<int>(zero_point_value);

    std::vector<schema::QuantParamT> quant_params;
    schema::QuantParamT quant_param;
    quant_param.scale = scale;
    quant_param.zeroPoint = zero_point;
    quant_param.inited = true;
    quant_params.push_back(quant_param);
    if (is_next_node) {
      quant_param_holder->set_input_quant_param(index, quant_params);
    } else {
      quant_param_holder->set_output_quant_param(index, quant_params);
    }
    return true;
  } else if (scale_vec_value != nullptr && zero_point_vec_value != nullptr) {
    auto scale_channel = GetValue<std::vector<float>>(scale_vec_value);
    auto zero_point_vec = GetValue<std::vector<int8_t>>(zero_point_vec_value);
    if (scale_channel.size() != zero_point_vec.size()) {
      MS_LOG(ERROR) << "Scale size is not equal to zero_point size! scale_channel.size(): " << scale_channel.size()
                    << ", zero_point_vec.size(): " << zero_point_vec.size();
      return false;
    }
    std::vector<schema::QuantParamT> quant_params;
    for (uint64_t i = 0; i < scale_channel.size(); i++) {
      schema::QuantParamT quant_param;
      quant_param.scale = scale_channel.at(i);
      quant_param.zeroPoint = zero_point_vec.at(i);
      quant_param.inited = true;
      quant_params.emplace_back(quant_param);
    }
    if (is_next_node) {
      quant_param_holder->set_input_quant_param(index, quant_params);
    } else {
      quant_param_holder->set_output_quant_param(index, quant_params);
    }
    return true;
  } else {
    MS_LOG(ERROR) << "Can not get quantization parameter attribute from this node: " << cnode->fullname_with_scope();
    return false;
  }
}

bool OnnxQuantizeLinearAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);

  // remove dequantizer
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!opt::CheckPrimitiveType(cnode, std::make_shared<Primitive>(lite::kNameDequantizeLinear))) {
      continue;
    }
    RemoveDequantizeLinear(func_graph, cnode);
  }

  if (!FoldQuantParms(func_graph)) {
    MS_LOG(ERROR) << "Fold quant params failed!";
    return false;
  }
  if (lite::ConverterInnerContext::GetInstance()->GetTargetDevice() == "Ascend") {
    if (!DoWeightQuant(func_graph)) {
      MS_LOG(ERROR) << "Do weight quant failed!";
      return false;
    }
    return true;
  }
  if (!DoWeightQuantDeQuant(func_graph)) {
    MS_LOG(ERROR) << "Do weight quant and dequant failed!";
    return false;
  }
  return true;
}

bool OnnxQuantizeLinearAdjust::FoldQuantParms(const FuncGraphPtr &func_graph) {
  // fold quant params to input/output tensor
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (!opt::CheckPrimitiveType(cnode, std::make_shared<Primitive>(lite::kNameQuantizeLinear))) {
      continue;
    }
    for (size_t i = 1; i < cnode->size(); i++) {
      auto input_cnode = cnode->input(i);
      MS_CHECK_TRUE_RET(input_cnode != nullptr, false);
      if (IsGraphInput(input_cnode) || !input_cnode->isa<mindspore::CNode>()) {
        continue;
      }
      auto previous_quant_holder = GetCNodeQuantHolder(input_cnode->cast<CNodePtr>());
      if (!SetQuantParam(cnode, previous_quant_holder, false, 0)) {
        MS_LOG(ERROR) << "Set output quant param failed.";
        return false;
      }
    }

    auto manager = func_graph->manager();
    if (manager == nullptr) {
      manager = Manage(func_graph, true);
    }
    MS_CHECK_TRUE_RET(manager != nullptr, false);
    auto node_users = manager->node_users()[cnode];
    for (auto &node_user : node_users) {
      auto next_quant_holder = GetCNodeQuantHolder(node_user.first->cast<CNodePtr>());
      if (!SetQuantParam(cnode, next_quant_holder, true, (node_user.second - kPrimOffset))) {
        MS_LOG(ERROR) << "Set input quant param failed.";
        return false;
      }
      manager->SetEdge(node_user.first, node_user.second, cnode->inputs()[kIndex1]);
    }
  }
  return true;
}

bool OnnxQuantizeLinearAdjust::DoWeightQuantDeQuant(const FuncGraphPtr &func_graph) {
  auto nodes = func_graph->GetOrderedCnodes();
  for (auto const &cnode : nodes) {
    auto quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(quant_param_holder);
    if (!quant_param_holder->IsInputQuantParamsInited()) {
      continue;
    }
    // weight constant folding
    for (size_t i = 1; i < cnode->size(); i++) {
      auto input_node = cnode->input(i);
      CHECK_NULL_RETURN(input_node);
      if (IsGraphInput(input_node) || input_node->isa<mindspore::CNode>()) {
        continue;
      }
      if (input_node->isa<mindspore::Parameter>()) {
        auto ret = DoParameterQuantDeQuant(cnode, input_node->cast<ParameterPtr>(), i, quant_param_holder);
        if (ret != RET_OK && ret != RET_NO_CHANGE) {
          MS_LOG(ERROR) << input_node->fullname_with_scope() << " parameter quant dequant failed.";
          return false;
        }
      }
    }
  }
  return true;
}

int OnnxQuantizeLinearAdjust::DoParameterQuantDeQuant(const CNodePtr &cnode, const ParameterPtr &input_node,
                                                      size_t input_index,
                                                      const QuantParamHolderPtr &quant_param_holder) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(input_node);
  if (input_index == THIRD_INPUT + 1 && quant::CheckNodeInSet(cnode, quant::kHasBiasOperator)) {
    return RET_NO_CHANGE;
  }
  auto tensor_info = input_node->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " can not get value";
    return RET_NULL_PTR;
  }
  if (tensor_info->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " is not float32, data will not quantify or dequantify.";
    return RET_OK;
  }
  int preferred_dim =
    quant::GetPreferredDim(cnode, input_index - 1, quant::ConvertShapeVectorToInt32(tensor_info->shape()));
  MS_CHECK_GT(static_cast<int>(quant_param_holder->get_input_quant_params().size()), static_cast<int>(input_index) - 1,
              RET_ERROR);
  auto quant_params = quant_param_holder->get_input_quant_params().at(input_index - 1);
  MS_CHECK_FALSE_MSG(quant_params.empty(), RET_ERROR, "quant_params is empty.");

  auto status = QuantDeQuantFilter(input_node, tensor_info, quant_params, preferred_dim);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed : " << status;
    return status;
  }
  return RET_OK;
}

// quant and dequant weight params
int OnnxQuantizeLinearAdjust::QuantDeQuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight,
                                                 const std::vector<schema::QuantParamT> &quant_params,
                                                 int preferred_dim) {
  size_t elem_count = weight->DataSize();
  auto raw_datas = static_cast<float *>(weight->data_c());
  std::vector<float> new_datas(elem_count);
  auto dims = quant::ConvertShapeVectorToInt32(weight->shape());
  MS_CHECK_FALSE_MSG(raw_datas == nullptr, RET_ERROR, "raw_data is nullptr.");

  size_t bit_num = quant_params.front().numBits;
  int quant_min = QuantMin(bit_num, false, false);
  int quant_max = QuantMax(bit_num, false);
  if (quant::IsPerchannelWeight(quant_params, weight, preferred_dim)) {
    auto count = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<>());
    MS_CHECK_FALSE_MSG(static_cast<size_t>(count) != elem_count, RET_ERROR, "element != count.");
    CHECK_LESS_RETURN(dims.size(), static_cast<size_t>(preferred_dim + 1));
    for (size_t i = 0; i < elem_count; i++) {
      float raw_data = raw_datas[i];
      auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
      auto quant_param = quant_params.at(bucket_index);
      auto new_data = quant::QuantDeQuantData<float>(raw_data, &quant_param, quant_max, quant_min);
      new_datas[i] = new_data;
    }
  } else {
    auto quant_param = quant_params.front();
    for (uint32_t i = 0; i < elem_count; i++) {
      float raw_data = raw_datas[i];
      auto new_data = quant::QuantDeQuantData<float>(raw_data, &quant_param, quant_max, quant_min);
      new_datas[i] = new_data;
    }
  }
  auto new_size = new_datas.size() * sizeof(float);
  if (memcpy_s(weight->data_c(), new_size, new_datas.data(), new_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

bool OnnxQuantizeLinearAdjust::DoWeightQuant(const FuncGraphPtr &func_graph) {
  MS_CHECK_FALSE_MSG(func_graph == nullptr, false, "func_graph is nullptr!");
  auto nodes = func_graph->GetOrderedCnodes();
  for (auto const &cnode : nodes) {
    auto quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(quant_param_holder);
    if (!quant_param_holder->IsInputQuantParamsInited()) {
      continue;
    }
    // weight constant folding
    for (size_t i = 1; i < cnode->size(); i++) {
      auto input_node = cnode->input(i);
      MS_CHECK_FALSE_MSG(input_node == nullptr, false, "input_node is nullptr!");
      if (IsGraphInput(input_node) || input_node->isa<mindspore::CNode>()) {
        continue;
      }
      // Only quantize the weights.
      if (input_node->isa<mindspore::Parameter>()) {
        auto ret = DoParameterQuant(cnode, input_node->cast<ParameterPtr>(), i, quant_param_holder);
        if (ret != RET_OK && ret != RET_NO_CHANGE) {
          MS_LOG(ERROR) << input_node->fullname_with_scope() << " parameter quant dequant failed.";
          return false;
        }
      }
    }
    // If cnode is constfold type (e.g. Const->Q->DQ->transpose->cnode2), copy its(transpose) input_quant_param to
    // cnode2.
    if (quant::CheckNodeInSet(cnode, kSelfInferOperator)) {
      auto manager = func_graph->manager();
      if (manager == nullptr) {
        manager = Manage(func_graph, true);
      }
      auto node_users = manager->node_users()[cnode];
      for (auto &node_user : node_users) {
        auto next_quant_holder = GetCNodeQuantHolder(node_user.first->cast<CNodePtr>());
        MS_CHECK_TRUE_RET(quant_param_holder != nullptr, false);
        auto fold_node_input_quant_param = quant_param_holder->get_input_quant_params();
        if (fold_node_input_quant_param.empty()) {
          MS_LOG(ERROR)
            << "quant_param_holder is InputQuantParamsInited, but its input_quant_param is empty! CNode name: "
            << cnode->fullname_with_scope();
          return false;
        }
        auto fold_quant_param = fold_node_input_quant_param.at(kIndex0);
        size_t index = node_user.second - 1;
        next_quant_holder->set_input_quant_param(index, fold_quant_param);
      }
    }
  }
  return true;
}

int OnnxQuantizeLinearAdjust::DoParameterQuant(const CNodePtr &cnode, const ParameterPtr &input_node,
                                               size_t input_index, const QuantParamHolderPtr &quant_param_holder) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(input_node);
  if (input_index == THIRD_INPUT + 1 && quant::CheckNodeInSet(cnode, quant::kHasBiasOperator)) {
    return RET_NO_CHANGE;
  }
  auto tensor_info = input_node->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " can not get value";
    return RET_NULL_PTR;
  }
  if (tensor_info->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " is not float32, data will not quantify or dequantify.";
    return RET_OK;
  }
  int preferred_dim =
    quant::GetPreferredDim(cnode, input_index - 1, quant::ConvertShapeVectorToInt32(tensor_info->shape()));
  MS_CHECK_GT(static_cast<int>(quant_param_holder->get_input_quant_params().size()), static_cast<int>(input_index) - 1,
              RET_ERROR);
  auto quant_params = quant_param_holder->get_input_quant_params().at(input_index - 1);
  MS_CHECK_FALSE_MSG(quant_params.empty(), RET_ERROR, "quant_params is empty.");

  auto status = QuantFilter(input_node, tensor_info, quant_params, preferred_dim);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed : " << status;
    return status;
  }
  return RET_OK;
}

// quant weight params
int OnnxQuantizeLinearAdjust::QuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight,
                                          const std::vector<schema::QuantParamT> &quant_params, int preferred_dim) {
  size_t elem_count = weight->DataSize();
  auto raw_datas = static_cast<float *>(weight->data_c());
  MS_CHECK_FALSE_MSG(raw_datas == nullptr, RET_ERROR, "raw_data is nullptr!");
  std::vector<int8_t> new_datas(elem_count);
  auto dims = quant::ConvertShapeVectorToInt32(weight->shape());

  size_t bit_num = quant_params.front().numBits;
  int quant_min = QuantMin(bit_num, false, false);
  int quant_max = QuantMax(bit_num, false);
  MS_LOG(INFO) << " quant_params.size(): " << quant_params.size() << ", weight->shape(): " << weight->shape()
               << ", preferred_dim: " << preferred_dim;
  if (quant::IsPerchannelWeight(quant_params, weight, preferred_dim)) {
    auto count = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<>());
    MS_CHECK_FALSE_MSG(static_cast<size_t>(count) != elem_count, RET_ERROR, "element != count.");
    CHECK_LESS_RETURN(dims.size(), static_cast<size_t>(preferred_dim + 1));
    for (size_t i = 0; i < elem_count; i++) {
      float raw_data = raw_datas[i];
      auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
      auto quant_param = quant_params.at(bucket_index);
      auto new_data = quant::QuantDeQuantData<int8_t>(raw_data, &quant_param, quant_max, quant_min, false);
      new_datas[i] = new_data;
    }
  } else {
    MS_LOG(ERROR) << "quant_params.size() !=  weight->shape()[preferred_dim], quant_params.size(): "
                  << quant_params.size() << ", weight->shape(): " << weight->shape()
                  << ", preferred_dim: " << preferred_dim;
    return lite::RET_ERROR;
  }
  auto new_size = new_datas.size() * sizeof(int8_t);
  std::vector<int64_t> dims_int64(dims.begin(), dims.end());
  auto new_weight_tensor = CreateTensorInfo(new_datas.data(), new_size, dims_int64, kNumberTypeInt8);
  if (new_weight_tensor == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return lite::RET_ERROR;
  }
  auto param_ptr = parameter_node->cast<ParameterPtr>();
  MS_CHECK_FALSE_MSG(param_ptr == nullptr, RET_ERROR, "param_ptr is nullptr!");
  param_ptr->set_default_param(new_weight_tensor);
  param_ptr->set_abstract(new_weight_tensor->ToAbstract());
  return RET_OK;
}
}  // namespace mindspore::lite
