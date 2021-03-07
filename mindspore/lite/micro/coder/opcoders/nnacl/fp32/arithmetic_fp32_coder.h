/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_FP32_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/op_coder.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#define DEFAULT_ARITHMETIC_NDIMS 10
namespace mindspore::lite::micro::nnacl {

using mindspore::schema::PrimitiveType_AddFusion;

using mindspore::schema::PrimitiveType_DivFusion;

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

using mindspore::schema::PrimitiveType_MulFusion;

using mindspore::schema::PrimitiveType_NotEqual;

using mindspore::schema::PrimitiveType_RealDiv;

using mindspore::schema::PrimitiveType_SquaredDifference;

using mindspore::schema::PrimitiveType_SubFusion;

using mindspore::schema::PrimitiveType_Eltwise;

using mindspore::schema::PrimitiveType_Minimum;

class ArithmeticFP32Coder final : public OperatorCoder {
 public:
  ArithmeticFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                      const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ArithmeticFP32Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int Init(CoderContext *const context);

  int BroadcastRun(const std::string &input0, const std::string &input1, const std::string &output, int dim,
                   int out_count, int out_thread_stride, NNaclFp32Serializer *const code);

  int break_pos_{0};

  int outside_{0};

  int out_thread_stride_{0};

  int out_count_{0};

  ArithmeticParameter *arithmetic_parameter_{nullptr};

  Tensor *filter_tensor_{nullptr};

  std::string arithmetic_run_;

  std::string arithmetic_run_int_;

  std::string arithmetic_opt_run_;

  std::string arithmetic_opt_run_int_;

  LiteDataType data_type_{kDataTypeFloat};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_ARITHMETIC_FP32_CODER_H_
