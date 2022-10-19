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
#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/resize_fusion.h"
#include <functional>
#include <memory>
#include <vector>
#include "tools/lite_exporter/fetch_content.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/tensor_util.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/resize.h"
#include "mindapi/base/types.h"

namespace mindspore::opt {
const BaseRef ResizeFusion::DefinePattern() const {
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, false);

  auto is_shape = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimShape>);
  MS_CHECK_TRUE_RET(is_shape != nullptr, {});
  VectorRef shape_ref = VectorRef({is_shape, input_});

  auto is_cast = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast != nullptr, {});
  auto var = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var != nullptr, false);
  VectorRef shape_cast_ref = VectorRef({is_cast, shape_ref, var});

  // h
  auto is_slice_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_slice_1 != nullptr, {});
  auto is_seq_var1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var1 != nullptr, {});
  VectorRef slice_ref_1 = VectorRef({is_slice_1, shape_cast_ref, is_seq_var1});

  auto is_mul_1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_1 != nullptr, {});
  auto var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var1 != nullptr, false);
  VectorRef mul_ref_1 = VectorRef({is_mul_1, slice_ref_1, var1});

  auto is_cast_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_2 != nullptr, {});
  auto var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var2 != nullptr, false);
  VectorRef shape_cast_ref_2 = VectorRef({is_cast_2, mul_ref_1, var2});

  // w
  auto is_slice_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStridedSlice>);
  MS_CHECK_TRUE_RET(is_slice_2 != nullptr, {});
  auto is_seq_var2 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var2 != nullptr, {});
  VectorRef slice_ref_2 = VectorRef({is_slice_2, shape_cast_ref, is_seq_var2});

  auto is_mul_2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul_2 != nullptr, {});
  auto var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var3 != nullptr, false);
  VectorRef mul_ref_2 = VectorRef({is_mul_2, slice_ref_2, var3});

  auto is_cast_4 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  MS_CHECK_TRUE_RET(is_cast_4 != nullptr, {});
  auto var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var4 != nullptr, false);
  VectorRef shape_cast_ref_4 = VectorRef({is_cast_4, mul_ref_2, var4});

  // concat h and w
  auto is_stack = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimStack>);
  MS_CHECK_TRUE_RET(is_stack != nullptr, {});
  VectorRef stack_ref = VectorRef({is_stack, shape_cast_ref_2, shape_cast_ref_4});

  auto is_resize = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimResize>);
  MS_CHECK_TRUE_RET(is_resize != nullptr, {});
  VectorRef resize_ref = VectorRef({is_resize, input_, stack_ref});

  return resize_ref;
}
CNodePtr ResizeFusion::GetAddCnode(const AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  if (!utils::isa<CNode>(node)) {
    return nullptr;
  }
  return node->cast<CNodePtr>();
}

int ResizeFusion::DoFuison(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const {
  MS_ASSERT(node != nullptr);
  auto resize_cnode = node->cast<CNodePtr>();
  MS_ASSERT(resize_cnode != nullptr);
  auto stack_cnode = resize_cnode->input(kInputIndexTwo)->cast<CNodePtr>();

  auto cast1_cnode = stack_cnode->input(1)->cast<CNodePtr>();
  auto mul1_cnode = cast1_cnode->input(1)->cast<CNodePtr>();
  auto mul_factor = mul1_cnode->input(kInputIndexTwo)->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(mul_factor != nullptr, lite::RET_ERROR);
  auto mul_factor_tensor = std::dynamic_pointer_cast<tensor::Tensor>(mul_factor);
  MS_CHECK_TRUE_RET(mul_factor_tensor != nullptr, lite::RET_ERROR);
  if (mul_factor_tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "scale factor data size is not equal to 1";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(mul_factor_tensor->data_c() != nullptr, lite::RET_ERROR);
  float mul_factor_data = (reinterpret_cast<float *>(mul_factor_tensor->data_c()))[0];

  auto cast2_cnode = stack_cnode->input(kInputIndexTwo)->cast<CNodePtr>();
  auto mul2_cnode = cast2_cnode->input(1)->cast<CNodePtr>();
  mul_factor = mul2_cnode->input(kInputIndexTwo)->cast<ParameterPtr>()->default_param();
  MS_CHECK_TRUE_RET(mul_factor != nullptr, lite::RET_ERROR);
  mul_factor_tensor = std::dynamic_pointer_cast<tensor::Tensor>(mul_factor);
  MS_CHECK_TRUE_RET(mul_factor_tensor != nullptr, lite::RET_ERROR);
  if (mul_factor_tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "scale factor data size is not equal to 1";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(mul_factor_tensor->data_c() != nullptr, lite::RET_ERROR);
  float mul_factor_data2 = (reinterpret_cast<float *>(mul_factor_tensor->data_c()))[0];
  if (mul_factor_data != mul_factor_data2) {
    MS_LOG(ERROR) << "two mul factor not equal";
    return lite::RET_ERROR;
  }

  std::vector<int64_t> shape = {kInputIndexTwo};
  auto tensor_info = lite::CreateTensorInfo(nullptr, 0, shape, kNumberTypeFloat32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return RET_ERROR;
  }
  auto tensor_data = static_cast<float *>(tensor_info->data_c());
  for (int i = 0; i < kInputIndexTwo; ++i) {
    tensor_data[i] = mul_factor_data;
  }
  auto shape_tensor = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(shape_tensor != nullptr, lite::RET_ERROR);
  auto status = lite::InitParameterFromTensorInfo(shape_tensor, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return RET_ERROR;
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(resize_cnode, kInputIndexTwo, shape_tensor);

  return lite::RET_OK;
}

const AnfNodePtr ResizeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  if (DoFuison(func_graph, node) != lite::RET_OK) {
    return nullptr;
  }
  return node;
}
}  // namespace mindspore::opt
