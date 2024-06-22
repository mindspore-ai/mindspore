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

#include "plugin/device/ascend/kernel/internal/inference_multi_matmul.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "param/matmul_qkv_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalInferenceMultiMatmulBase::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                                     const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::MatmulQkv;
  bool transpose_a = false;
  bool transpose_b = true;
  auto n_lens = primitive_->GetAttr("n_lens");
  MS_EXCEPTION_IF_NULL(n_lens);
  auto n_list = GetValue<std::vector<int64_t>>(n_lens);
  if (n_list.size() == 2) {
    n_list.push_back(0);
  }
  bool with_bias = primitive_->HasAttr("with_bias") ? GetValue<bool>(primitive_->GetAttr("with_bias")) : false;
  int32_t silu_position =
    primitive_->HasAttr("silu_position") ? GetValue<int32_t>(primitive_->GetAttr("silu_position")) : -1;
  const auto n_input_zero = 0;
  const auto n_input_one = 1;
  const auto n_input_two = 2;
  internal::MatmulQkvParam op_param = {static_cast<uint32_t>(n_list[n_input_zero]),
                                       static_cast<uint32_t>(n_list[n_input_one]),
                                       static_cast<uint32_t>(n_list[n_input_two]), transpose_a, transpose_b};
  op_param.silu_position = silu_position;
  op_param.with_bias = with_bias;
  param_ptr->specificParam = op_param;
  return param_ptr;
}

// MatmulSplitOut3
class InternalMatmulSplitOut3 : public InternalInferenceMultiMatmulBase {
 public:
  InternalMatmulSplitOut3() : InternalInferenceMultiMatmulBase("InternalMatmulSplitOut3") {}
  ~InternalMatmulSplitOut3() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitOut3, InternalMatmulSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitOut3, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// MatmulSplitOut2
class InternalMatmulSplitOut2 : public InternalInferenceMultiMatmulBase {
 public:
  InternalMatmulSplitOut2() : InternalInferenceMultiMatmulBase("InternalMatmulSplitOut2") {}
  ~InternalMatmulSplitOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitOut2, InternalMatmulSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitOut2, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// QuantBatchMatmulSplitOut3
class InternalQuantBatchMatmulSplitOut3 : public InternalInferenceMultiMatmulBase {
 public:
  InternalQuantBatchMatmulSplitOut3() : InternalInferenceMultiMatmulBase("InternalQuantBatchMatmulSplitOut3") {}
  ~InternalQuantBatchMatmulSplitOut3() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QuantBatchMatmulSplitOut3, InternalQuantBatchMatmulSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantBatchMatmulSplitOut3, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantBatchMatmulSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// MatmulBiasSplitOut2
class InternalMatmulBiasSplitOut2 : public InternalInferenceMultiMatmulBase {
 public:
  InternalMatmulBiasSplitOut2() : InternalInferenceMultiMatmulBase("InternalMatmulBiasSplitOut2") {}
  ~InternalMatmulBiasSplitOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulBiasSplitOut2, InternalMatmulBiasSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitOut2, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulBiasSplitSiluOut2
class InternalMatmulBiasSplitSiluOut2 : public InternalInferenceMultiMatmulBase {
 public:
  InternalMatmulBiasSplitSiluOut2() : InternalInferenceMultiMatmulBase("InternalMatmulBiasSplitSiluOut2") {}
  ~InternalMatmulBiasSplitSiluOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulBiasSplitSiluOut2, InternalMatmulBiasSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitSiluOut2, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// QuantBatchMatmulSplitOut2
class InternalQuantBatchMatmulSplitOut2 : public InternalInferenceMultiMatmulBase {
 public:
  InternalQuantBatchMatmulSplitOut2() : InternalInferenceMultiMatmulBase("InternalQuantBatchMatmulSplitOut2") {}
  ~InternalQuantBatchMatmulSplitOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QuantBatchMatmulSplitOut2, InternalQuantBatchMatmulSplitOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantBatchMatmulSplitOut2, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantBatchMatmulSplitOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// QuantBatchMatmulSplitSiluOut2
class InternalQuantBatchMatmulSplitSiluOut2 : public InternalInferenceMultiMatmulBase {
 public:
  InternalQuantBatchMatmulSplitSiluOut2() : InternalInferenceMultiMatmulBase("InternalQuantBatchMatmulSplitSiluOut2") {}
  ~InternalQuantBatchMatmulSplitSiluOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(QuantBatchMatmulSplitSiluOut2, InternalQuantBatchMatmulSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantBatchMatmulSplitSiluOut2, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantBatchMatmulSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MatmulBiasSplitOut3
class InternalMatmulBiasSplitOut3 : public InternalInferenceMultiMatmulBase {
 public:
  InternalMatmulBiasSplitOut3() : InternalInferenceMultiMatmulBase("InternalMatmulBiasSplitOut3") {}
  ~InternalMatmulBiasSplitOut3() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulBiasSplitOut3, InternalMatmulBiasSplitOut3);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulBiasSplitOut3, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulBiasSplitOut3, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);

// MatmulSplitSiluOut2
class InternalMatmulSplitSiluOut2 : public InternalInferenceMultiMatmulBase {
 public:
  InternalMatmulSplitSiluOut2() : InternalInferenceMultiMatmulBase("InternalMatmulSplitSiluOut2") {}
  ~InternalMatmulSplitSiluOut2() = default;
};

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulSplitSiluOut2, InternalMatmulSplitSiluOut2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulSplitSiluOut2, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulSplitSiluOut2, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
