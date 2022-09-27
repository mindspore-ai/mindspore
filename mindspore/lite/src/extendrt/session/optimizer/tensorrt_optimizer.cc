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
#include "extendrt/session/optimizer/tensorrt_optimizer.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
tensor::TensorPtr TensorRtOptimizer::GetParameterValue(const CNodePtr &node, size_t parameter_index) {
  if (node == nullptr) {
    return nullptr;
  }
  auto input = common::AnfAlgo::GetInputNode(node, parameter_index);
  if (!input->isa<Parameter>()) {
    return nullptr;
  }
  auto parameter = input->cast<ParameterPtr>();
  if (parameter == nullptr && !parameter->has_default()) {
    return nullptr;
  }
  auto param_val = parameter->default_param();
  if (!param_val->isa<tensor::Tensor>()) {
    return nullptr;
  }
  return param_val->cast<tensor::TensorPtr>();
}

std::vector<int32_t> TensorRtOptimizer::GetParameterIntValue(const CNodePtr &node, size_t parameter_index) {
  auto tensor = GetParameterValue(node, parameter_index);
  auto elem_num = tensor->ElementsNum();
  if (elem_num < 1) {
    return {};
  }
  auto data_c = tensor->data_c();
  if (data_c == nullptr) {
    return {};
  }
  std::vector<int32_t> ints;
  auto type_id = tensor->Dtype()->type_id();
  if (type_id == TypeId::kNumberTypeInt32) {
    auto int_val = reinterpret_cast<const int32_t *>(data_c);
    for (int64_t i = 0; i < elem_num; i++) {
      ints.push_back(int_val[i]);
    }
  } else if (type_id == TypeId::kNumberTypeInt64) {
    auto int_val = reinterpret_cast<const int64_t *>(data_c);
    for (int64_t i = 0; i < elem_num; i++) {
      ints.push_back(LongToInt(int_val[i]));
    }
  } else {
    return {};
  }
  return ints;
}

std::vector<float> TensorRtOptimizer::GetParameterFloatValue(const CNodePtr &node, size_t parameter_index) {
  auto tensor = GetParameterValue(node, parameter_index);
  auto elem_num = tensor->ElementsNum();
  if (elem_num < 1) {
    return {};
  }
  auto data_c = tensor->data_c();
  if (data_c == nullptr) {
    return {};
  }
  std::vector<float> floats;
  auto type_id = tensor->Dtype()->type_id();
  if (type_id == TypeId::kNumberTypeInt32) {
    auto int_val = reinterpret_cast<const int32_t *>(data_c);
    for (int64_t i = 0; i < elem_num; i++) {
      floats.push_back(IntToFloat(int_val[i]));
    }
  } else if (type_id == TypeId::kNumberTypeInt64) {
    auto int_val = reinterpret_cast<const int64_t *>(data_c);
    for (int64_t i = 0; i < elem_num; i++) {
      floats.push_back(LongToFloat(int_val[i]));
    }
  } else if (type_id == TypeId::kNumberTypeFloat32) {
    auto float_val = reinterpret_cast<const float *>(data_c);
    for (int64_t i = 0; i < elem_num; i++) {
      floats.push_back(float_val[i]);
    }
  } else {
    return {};
  }
  return floats;
}

bool TensorRtOptimizer::GetMatmulFactor(const AnfNodePtr &pack_input, float *matmul_factor, int32_t *sclice_index,
                                        AnfNodePtr *shape_input) {
  constexpr size_t expect_matmul_input_size = 2;
  if (!common::AnfAlgo::CheckPrimitiveType(pack_input, prim::kPrimMulFusion)) {
    return false;
  }
  auto matmul = pack_input->cast<CNodePtr>();
  if (common::AnfAlgo::GetInputNum(matmul) != expect_matmul_input_size) {
    return false;
  }
  auto matmul_factors = GetParameterFloatValue(matmul, kIndex1);
  if (matmul_factors.size() != 1) {
    return false;
  }
  *matmul_factor = matmul_factors[0];
  auto matmul_input0 = common::AnfAlgo::GetInputNode(matmul, kIndex0);
  if (!common::AnfAlgo::CheckPrimitiveType(matmul_input0, prim::kPrimStridedSlice)) {
    return false;
  }
  auto slice_node = matmul_input0->cast<CNodePtr>();
  constexpr size_t slice_input_size = 4;
  if (common::AnfAlgo::GetInputNum(slice_node) != slice_input_size) {
    return false;
  }
  auto begin_vec = GetParameterIntValue(slice_node, kIndex1);
  auto end_vec = GetParameterIntValue(slice_node, kIndex2);
  auto stride_vec = GetParameterIntValue(slice_node, kIndex3);
  if (begin_vec.size() != 1 || end_vec.size() != 1 || stride_vec.size() != 1) {
    return false;
  }
  if (begin_vec[0] + 1 != end_vec[0]) {
    return false;
  }
  *sclice_index = begin_vec[0];
  auto slice_input = common::AnfAlgo::GetInputNode(slice_node, kIndex0);
  if (!common::AnfAlgo::CheckPrimitiveType(slice_input, prim::kPrimShape) &&
      !common::AnfAlgo::CheckPrimitiveType(slice_input, prim::kPrimTensorShape) &&
      !common::AnfAlgo::CheckPrimitiveType(slice_input, prim::kPrimDynamicShape)) {
    return false;
  }
  auto shape_node = slice_input->cast<CNodePtr>();
  constexpr size_t reshape_input_size = 1;
  if (common::AnfAlgo::GetInputNum(shape_node) < reshape_input_size) {
    return false;
  }
  auto shape_input0 = common::AnfAlgo::GetInputNode(shape_node, kIndex0);
  while (common::AnfAlgo::CheckPrimitiveType(shape_input0, prim::kPrimTranspose)) {
    shape_input0 = common::AnfAlgo::GetInputNode(shape_input0->cast<CNodePtr>(), kIndex0);
  }
  *shape_input = shape_input0;
  return true;
}

bool TensorRtOptimizer::OptResizeScales(const FuncGraphPtr &func_graph, const CNodePtr &resize_node) {
  auto resize_input1 = common::AnfAlgo::GetInputNode(resize_node, kIndex1);
  if (resize_input1 == nullptr) {
    return false;
  }
  if (!common::AnfAlgo::CheckPrimitiveType(resize_input1, prim::kPrimStack)) {
    return false;
  }
  auto pack_node = resize_input1->cast<CNodePtr>();
  constexpr size_t expect_pack_input_size = 2;
  if (common::AnfAlgo::GetInputNum(pack_node) != expect_pack_input_size) {
    return false;
  }
  auto pack_input0 = common::AnfAlgo::GetInputNode(pack_node, kIndex0);
  auto pack_input1 = common::AnfAlgo::GetInputNode(pack_node, kIndex1);

  float matmul_factor0 = 0.0;
  int32_t slice_dim_input0 = 0;
  AnfNodePtr shape0_input = nullptr;
  if (!GetMatmulFactor(pack_input0, &matmul_factor0, &slice_dim_input0, &shape0_input)) {
    return false;
  }
  float matmul_factor1 = 0.0;
  int32_t slice_dim_input1 = 0;
  AnfNodePtr shape1_input = nullptr;
  if (!GetMatmulFactor(pack_input1, &matmul_factor1, &slice_dim_input1, &shape1_input)) {
    return false;
  }
  auto resize_input0 = common::AnfAlgo::GetInputNode(resize_node, kIndex0);
  while (common::AnfAlgo::CheckPrimitiveType(resize_input0, prim::kPrimTranspose)) {
    resize_input0 = common::AnfAlgo::GetInputNode(resize_input0->cast<CNodePtr>(), kIndex0);
  }
  if (resize_input0 != shape0_input || resize_input0 != shape1_input) {
    return false;
  }
  if (matmul_factor0 <= 0.0f || matmul_factor1 <= 0.0f) {
    return false;
  }
  std::vector<float> scales;
  scales.push_back(1);
  scales.push_back(1);
  if ((slice_dim_input0 == kNCHW_H && slice_dim_input1 == kNCHW_W) ||
      (slice_dim_input0 == kNHWC_H && slice_dim_input1 == kNHWC_W)) {  // 1,2 or 2,3
    scales.push_back(matmul_factor0);
    scales.push_back(matmul_factor1);
  } else if ((slice_dim_input1 == kNCHW_H && slice_dim_input0 == kNCHW_W) ||
             (slice_dim_input1 == kNHWC_H && slice_dim_input0 == kNHWC_W)) {
    scales.push_back(matmul_factor1);
    scales.push_back(matmul_factor0);
  } else {
    return false;
  }
  common::AnfAlgo::SetNodeAttr(kAttrScales, MakeValue(scales), resize_node);
  return true;
}

bool TensorRtOptimizer::OptResizeHeightWidth(const FuncGraphPtr &func_graph, const CNodePtr &resize_node) {
  auto resize_input1 = common::AnfAlgo::GetInputNode(resize_node, kIndex1);
  if (resize_input1 == nullptr) {
    return false;
  }
  if (!common::AnfAlgo::CheckPrimitiveType(resize_input1, prim::kPrimGather)) {
    return false;
  }
  auto gather_node = resize_input1->cast<CNodePtr>();
  constexpr size_t expect_gather_input_size = 2;
  if (common::AnfAlgo::GetInputNum(gather_node) < expect_gather_input_size) {
    return false;
  }
  auto gather_const = GetParameterIntValue(gather_node, kIndex1);
  if (gather_const.size() != kDim2 || gather_const[0] != kNCHW_H || gather_const[1] != kNCHW_W) {
    return false;
  }
  auto gather_input0 = common::AnfAlgo::GetInputNode(gather_node, kIndex0);
  if (!common::AnfAlgo::CheckPrimitiveType(gather_input0, prim::kPrimConcat)) {
    return false;
  }
  auto manager = func_graph->manager();
  // input 0 is primitive, real input 1 index is kIndex1 + 1
  manager->SetEdge(resize_node, kIndex1 + 1, gather_input0);
  return true;
}

void TensorRtOptimizer::RunOptimizer(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  constexpr size_t resize_input_size = 2;
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimResize)) {
      continue;
    }
    auto resize_cnode = node->cast<CNodePtr>();
    if (common::AnfAlgo::GetInputNum(resize_cnode) != resize_input_size) {
      MS_LOG_WARNING << "Input size " << common::AnfAlgo::GetInputNum(resize_cnode) << " of resize node "
                     << resize_cnode->fullname_with_scope() << " != 2";
      continue;
    }
    if (OptResizeScales(func_graph, resize_cnode)) {
      continue;
    }
    if (OptResizeHeightWidth(func_graph, resize_cnode)) {
      continue;
    }
  }
}
}  // namespace mindspore
