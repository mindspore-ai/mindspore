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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_ARITHMETIC_DYNAMIC_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_ARITHMETIC_DYNAMIC_FP16_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/op_coder.h"
#include "nnacl/base/cast_base.h"
#include "nnacl/arithmetic_parameter.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/nnacl/dynamic_parameter/arithmetic_dynamic_parameter.h"
#include "nnacl/broadcast_to_parameter.h"

namespace mindspore::lite::micro::nnacl {
using mindspore::schema::PrimitiveType_AddFusion;
using mindspore::schema::PrimitiveType_DivFusion;
using mindspore::schema::PrimitiveType_Eltwise;
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_Mod;
using mindspore::schema::PrimitiveType_MulFusion;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_RealDiv;
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_SubFusion;

class ArithmeticDynamicFP16Coder final : public OperatorCoder {
  typedef struct {
    int primitive_type_;
    int activation_type_;
    std::string func_;
    std::string int_func_;
    std::string bool_func_;
    std::string opt_func_;
    std::string opt_int_func_;
  } ARITHMETIC_FUNC_INFO_FP16;

 public:
  ArithmeticDynamicFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                             const LiteGraph::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ArithmeticDynamicFP16Coder() override = default;

  int DoCode(CoderContext *const context) override;

 private:
  int Prepare(CoderContext *const context) override;

  void InitFunTable();

  void InitRunFunction(int primitive_type);

  void InitDynamicParams();

  void ResetStatus();

  void CalcMultiplesAndStrides();

  void ComputeStrides(const std::vector<std::string> &shape, std::vector<std::string> *strides);

  int ExecuteCode(const std::string &input0, const std::string &input1, const std::string &output,
                  const std::string size, CoderContext *const context, NNaclFp32Serializer *const code);

  int DoBroadcast(NNaclFp32Serializer *const code);

  std::vector<ARITHMETIC_FUNC_INFO_FP16> fun_table_;
  ArithmeticFuncType arithmetic_func_type_{kArithmeticFuncUnknow};
  ArithmeticParameter *param_{nullptr};
  ArithmeticDynamicParameter dynamic_param_;
  BroadcastShapeInfo broadcast_info_;
  BroadcastDynamicShapeInfo dynamic_shape_info_;
  Tensor *filter_tensor_{nullptr};
  std::string input0_ptr_str_;
  std::string input1_ptr_str_;
  std::string output_ptr_str_;
  std::string arithmetic_run_;
  std::string arithmetic_run_int_;
  std::string arithmetic_opt_run_;
  std::string arithmetic_opt_run_int_;
  std::string arithmetic_run_bool_;
  std::string arithmetic_func_str_;
  std::vector<std::string> in0_shape_;
  std::vector<std::string> in1_shape_;
  std::vector<std::string> out_shape_;
  std::vector<std::string> in0_strides_;
  std::vector<std::string> in1_strides_;
  std::vector<std::string> out_strides_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_ARITHMETIC_DYNAMIC_FP16_CODER_H_
