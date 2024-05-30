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

#include "plugin/device/ascend/kernel/internal/matmul_elemwise.h"

#include <memory>
#include <string>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "param/matmul_ext_param.h"

namespace mindspore {
namespace kernel {

constexpr auto matmul_elemwise_fusion_relu_str = "relu";
constexpr auto matmul_elemwise_fusion_gelu_str = "gelu";
constexpr auto matmul_elemwise_fusion_biasadd_str = "bias_add";
constexpr auto matmul_elemwise_fusion_biasadd_fastgelu_str = "bias_add_fastgelu";

internal::OpParamPtr InternalMatmulElemBase::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::MatMulExtParam>();
  internal::MatMulParam matmul_param;
  param_ptr->opId = internal::OpId::MatMul;
  // setup matmul param from inputs
  bool transpose_a = primitive_->HasAttr("is_trans_a") ? GetValue<bool>(primitive_->GetAttr("is_trans_a")) : false;
  bool transpose_b = primitive_->HasAttr("is_trans_b") ? GetValue<bool>(primitive_->GetAttr("is_trans_b")) : false;
  auto shape_a = inputs[kIndex0]->GetShapeVector();
  auto shape_b = inputs[kIndex1]->GetShapeVector();
  int m = (!transpose_a) ? shape_a[kIndex0] : shape_a[kIndex1];
  int k = (!transpose_a) ? shape_a[kIndex1] : shape_a[kIndex0];
  int n = (!transpose_b) ? shape_b[kIndex1] : shape_b[kIndex0];

  param_ptr->input_dtype = InternalKernelUtils::ToInternalDType(inputs[kIndex0]->dtype_id());
  param_ptr->weight_dtype = InternalKernelUtils::ToInternalDType(inputs[kIndex1]->dtype_id());
  param_ptr->output_dtype = InternalKernelUtils::ToInternalDType(outputs[kIndex0]->dtype_id());

  bool with_relu = false;
  bool with_gelu = false;
  bool with_bias = false;
  bool with_bias_fastgelu = false;

  auto value_str = primitive_->GetAttr("ElemwiseType");
  MS_EXCEPTION_IF_NULL(value_str);
  std::string elemwise_type = GetValue<std::string>(value_str);
  if (elemwise_type == matmul_elemwise_fusion_relu_str) {
    with_relu = true;
  } else if (elemwise_type == matmul_elemwise_fusion_gelu_str) {
    with_gelu = true;
  } else if (elemwise_type == matmul_elemwise_fusion_biasadd_str) {
    with_bias = true;
    param_ptr->bias_dtype = InternalKernelUtils::ToInternalDType(inputs[kIndex2]->dtype_id());
  } else if (elemwise_type == matmul_elemwise_fusion_biasadd_fastgelu_str) {
    with_bias_fastgelu = true;
    param_ptr->bias_dtype = InternalKernelUtils::ToInternalDType(inputs[kIndex2]->dtype_id());
  }

  param_ptr->with_relu = with_relu;
  param_ptr->with_gelu = with_gelu;
  param_ptr->with_bias = with_bias;
  param_ptr->with_bias_fastgelu = with_bias_fastgelu;

  matmul_param = {
    transpose_a,  // transposeA
    transpose_b,  // transposeB
    {m, k, n},    // oriShape
    with_bias,    // withBias
    false,        // enDequant
    0,            // tilingN
    0,            // tilingK
    false,        // enShuffleK
  };
  param_ptr->specificParam = matmul_param;
  return std::static_pointer_cast<internal::OpParam>(param_ptr);
}

class InternalMatmulElemBinary : public InternalMatmulElemBase {
 public:
  InternalMatmulElemBinary() : InternalMatmulElemBase("MatmulElemBinary") {}
  ~InternalMatmulElemBinary() = default;

  uint64_t GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                             const std::vector<KernelTensor *> &outputs) override {
    // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
    return TilingCacheMgr::GetInstance().GenTilingCacheKey(
      kernel_name_, inputs[kIndex0]->GetShapeVector(), inputs[kIndex0]->dtype_id(), inputs[kIndex1]->GetShapeVector(),
      inputs[kIndex1]->dtype_id(), inputs[kIndex2]->GetShapeVector(), inputs[kIndex2]->dtype_id());
  }
};

class InternalMatmulElemUnary : public InternalMatmulElemBase {
 public:
  InternalMatmulElemUnary() : InternalMatmulElemBase("MatmulElemUnary") {}
  ~InternalMatmulElemUnary() = default;

  uint64_t GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                             const std::vector<KernelTensor *> &outputs) override {
    // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
    return TilingCacheMgr::GetInstance().GenTilingCacheKey(
      kernel_name_, inputs[kIndex0]->GetShapeVector(), inputs[kIndex0]->dtype_id(), inputs[kIndex1]->GetShapeVector(),
      inputs[kIndex1]->dtype_id());
  }
};

MS_INTERNAL_KERNEL_FACTORY_REG(FusedMatMulElemBinary, InternalMatmulElemBinary);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedMatMulElemBinary, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FusedMatMulElemBinary, OUTPUT_NUM_1, INDEX_0);
MS_INTERNAL_KERNEL_FACTORY_REG(FusedMatMulElemUnary, InternalMatmulElemUnary);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedMatMulElemUnary, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FusedMatMulElemUnary, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
