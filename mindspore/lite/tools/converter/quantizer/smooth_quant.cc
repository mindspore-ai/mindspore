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

#include "tools/converter/quantizer/smooth_quant.h"
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include "include/errorcode.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindrt/src/thread/threadpool.h"
#include "nnacl/fp32/scale_fp32.h"
#include "ops/fusion/scale_fusion.h"

namespace mindspore::lite::quant {
namespace {
constexpr float EPSILON_MIN = 1e-5;
constexpr float EPSILON_MAX = 100;
constexpr size_t CHANNEL_IN_AXIS = 0;
constexpr size_t CHANNEL_OUT_AXIS = 1;
}  // namespace

int GetPerAxisQuantParams(const float *raw_datas, size_t elem_count, const std::vector<int> &dims, int preferred_dim,
                          std::vector<schema::QuantParamT> *quant_params) {
  // the key is bucket_index
  std::map<int, MinMax> per_channel_min_max;
  GetAllChannelMinMax(raw_datas, elem_count, dims, preferred_dim, &per_channel_min_max);

  // Cal Quant param
  for (auto min_max_map : per_channel_min_max) {
    float min = min_max_map.second.min;
    float max = min_max_map.second.max;
    schema::QuantParamT quant_param;
    auto ret = CalQuantizationParams(&quant_param, min, max, k8Bit, true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Cal quantization params failed.";
      return ret;
    }
    quant_params->push_back(quant_param);
  }
  return RET_OK;
}

int SmoothQuant::Run(const FuncGraphPtr &func_graph, double smooth_alpha) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto cnode_name = cnode->fullname_with_scope();
    auto holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(holder);
    if (holder->quant_type() != quant::QUANT_ALL) {
      continue;
    }
    std::set<PrimitivePtr> support_primitive_types = {prim::kPrimMatMulFusion, prim::kPrimMatMul};
    if (!CheckNodeInSet(cnode, support_primitive_types)) {
      continue;
    }
    auto ret = LinearSmooth(func_graph, cnode, smooth_alpha);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode_name << " linear smooth quant failed.";
      return ret;
    }
  }
  return RET_OK;
}

int SmoothWeight(const CNodePtr &cnode, const std::vector<float> &smooth_scales, float *weight_tensor_data) {
  auto cnode_name = cnode->fullname_with_scope();

  ShapeVector matrix_b_shape;
  auto ret = opt::FetchShapeFromAbstract(cnode->input(kWeightIndex + kPrimOffset)->abstract(), &matrix_b_shape);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode_name << " Fetch matrix b shape failed.";
    return ret;
  }
  if (matrix_b_shape.size() != DIMENSION_2D) {
    MS_LOG(ERROR) << "Only support shape equal 2 and matrix b shape is " << matrix_b_shape;
    return RET_ERROR;
  }
  // Adjust Weight
  // transpose_b is false.
  for (int i = 0; i < matrix_b_shape.at(0); ++i) {    // deep
    for (int j = 0; j < matrix_b_shape.at(1); ++j) {  // col
      auto index = i * matrix_b_shape.at(1) + j;
      weight_tensor_data[index] *= smooth_scales.at(i);
    }
  }
  return RET_OK;
}

int InsertScaleForActivation(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                             const std::vector<float> &smooth_scales) {
  auto cnode_name = cnode->fullname_with_scope();
  ParameterPtr scale_ratio;
  auto dtype = kNumberTypeFloat32;
  if (cnode->HasAttr("origin_type")) {
    auto value = cnode->GetAttr("origin_type");
    dtype = static_cast<TypeId>(opt::CastToInt(value).front());
  }
  if (dtype == kNumberTypeFloat16) {
    std::vector<float16> smooth_scales_fp16(smooth_scales.size());
    for (size_t i = 0; i < smooth_scales.size(); ++i) {
      smooth_scales_fp16[i] = static_cast<float16>(1.0f / smooth_scales[i]);
    }
    scale_ratio = opt::BuildFloat16VecParameterNode(func_graph, smooth_scales_fp16, cnode_name + "_scales");
  } else {
    std::vector<float> smooth_scales_fp32(smooth_scales.size());
    for (size_t i = 0; i < smooth_scales.size(); ++i) {
      smooth_scales_fp32[i] = 1.0f / smooth_scales[i];
    }
    scale_ratio = opt::BuildFloatVecParameterNode(func_graph, smooth_scales_fp32, cnode_name + "_scales");
  }
  // Insert scale node
  InsertQuantNodeManager insert_manager;
  auto mul_cnode = insert_manager.NewMulNode(func_graph, cnode->input(kInputIndex + kPrimOffset), scale_ratio);
  CHECK_NULL_RETURN(mul_cnode);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  manager->SetEdge(cnode, kInputIndex + kPrimOffset, mul_cnode);
  return RET_OK;
}

int UpdateQuantParam(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const tensor::TensorPtr &tensor_info,
                     const std::vector<schema::QuantParamT> &matrix_a_quants, const std::vector<float> &smooth_scales) {
  auto cnode_name = cnode->fullname_with_scope();
  auto holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(holder);
  float act_min = FLT_MAX;
  float act_max = FLT_MIN;
  for (size_t i = 0; i < smooth_scales.size(); ++i) {
    auto quant = matrix_a_quants.at(i);
    float adjust_min = static_cast<float>(quant.min) / smooth_scales.at(i);
    float adjust_max = static_cast<float>(quant.max) / smooth_scales.at(i);
    act_min = std::min(act_min, adjust_min);
    act_min = std::min(act_min, adjust_max);
    act_max = std::max(act_max, adjust_min);
    act_max = std::max(act_max, adjust_max);
  }
  schema::QuantParamT act_quant_param;
  auto ret = CalQuantizationParams(&act_quant_param, act_min, act_max, k8Bit, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode_name << " calculate quantization param failed.";
    return ret;
  }
  holder->set_input_quant_param(0, {act_quant_param});
  // update weight quant info
  auto weight_tensor_data = static_cast<float *>(tensor_info->data_c());
  std::vector<schema::QuantParamT> weight_quant_params;
  GetPerAxisQuantParams(weight_tensor_data, tensor_info->DataSize(), ConvertShapeVectorToInt32(tensor_info->shape_c()),
                        CHANNEL_OUT_AXIS, &weight_quant_params);
  holder->set_input_quant_param(kWeightIndex, weight_quant_params);
  // Bias dont need adjust. Y = (1 / Scale)X * (Scale)W + Bias
  if (cnode->size() > kBiasIndex + kPrimOffset) {
    std::vector<schema::QuantParamT> bias_weight_quant_params;
    ret = CalBiasQuantParams({act_quant_param}, weight_quant_params, &bias_weight_quant_params);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode_name << " calculate bias quantization param failed.";
      return ret;
    }
    holder->set_input_quant_param(kBiasIndex, bias_weight_quant_params);
  }
  return RET_OK;
}

// All tensors must float32.
int SmoothQuant::LinearSmooth(const FuncGraphPtr &func_graph, const CNodePtr &cnode, double alpha) {
  CHECK_NULL_RETURN(cnode);
  auto cnode_name = cnode->fullname_with_scope();
  std::set<PrimitivePtr> support_primitive_types = {prim::kPrimMatMulFusion, prim::kPrimMatMul};
  if (!CheckNodeInSet(cnode, support_primitive_types)) {
    MS_LOG(ERROR) << cnode_name << " linear smooth only support MatMulFusion|MatMul";
    return RET_ERROR;
  }
  auto holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(holder);
  auto input_quant_params = holder->get_input_quant_params();
  // quant param validation.
  if (input_quant_params.size() < Num2) {
    MS_LOG(ERROR) << cnode_name << " quant param size is less than 2.";
    return RET_ERROR;
  }
  auto matrix_a_quants = input_quant_params.at(kInputIndex);
  if (matrix_a_quants.size() <= Num1) {
    MS_LOG(INFO) << cnode_name << " will not smooth quant.";
    return RET_OK;
  }
  // Calculate matrix b smooth scale.
  ParameterPtr param_node;
  tensor::TensorPtr tensor_info;
  GetParameterAndTensor(cnode->input(kWeightIndex + kPrimOffset), &param_node, &tensor_info);
  CHECK_NULL_RETURN(param_node);
  CHECK_NULL_RETURN(tensor_info);
  if (tensor_info->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << cnode_name << " weight data type " << tensor_info->data_type() << " is not kNumberTypeFloat32.";
    return RET_ERROR;
  }
  float *weight_tensor_data = static_cast<float *>(tensor_info->data_c());
  std::vector<schema::QuantParamT> smooth_weight_quant_params;
  GetPerAxisQuantParams(weight_tensor_data, tensor_info->DataSize(), ConvertShapeVectorToInt32(tensor_info->shape_c()),
                        CHANNEL_IN_AXIS, &smooth_weight_quant_params);

  auto matrix_b_quants = smooth_weight_quant_params;
  if (matrix_a_quants.size() != matrix_b_quants.size()) {
    MS_LOG(INFO) << cnode_name
                 << " quant param size is not equal, and don't support smooth quant optimize. input size is "
                 << matrix_a_quants.size() << " and weight quant param size is " << matrix_b_quants.size();
    return RET_OK;
  }

  // calculate smooth
  size_t scale_size = matrix_a_quants.size();
  std::vector<float> smooth_scales(scale_size);
  for (size_t i = 0; i < scale_size; ++i) {
    auto abs_act_max = std::max(std::abs(matrix_a_quants.at(i).min), std::abs(matrix_a_quants.at(i).max));
    auto abs_w_max = std::max(std::abs(matrix_b_quants.at(i).min), std::abs(matrix_b_quants.at(i).max));
    float scale = std::pow(abs_act_max, alpha) / std::pow(abs_w_max, 1 - alpha);
    smooth_scales.at(i) = std::min(std::max(scale, EPSILON_MIN), EPSILON_MAX);
  }

  auto ret = SmoothWeight(cnode, smooth_scales, weight_tensor_data);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode_name << " smooth weight failed.";
    return ret;
  }
  ret = InsertScaleForActivation(func_graph, cnode, smooth_scales);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode_name << " insert scale for activation failed.";
    return ret;
  }

  // update act quant param.
  ret = UpdateQuantParam(func_graph, cnode, tensor_info, matrix_a_quants, smooth_scales);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode_name << " update quant param failed.";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
