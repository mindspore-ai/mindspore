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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_FP32_SERIALIZER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_FP32_SERIALIZER_H_
#include <string>
#include <sstream>
#include "coder/opcoders/serializers/serializer.h"
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/scale.h"
#include "nnacl/slice_parameter.h"
#include "nnacl/base/tile_base.h"
#include "nnacl/fp32/transpose_fp32.h"
#include "nnacl/pooling_parameter.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/splice_parameter.h"
#include "wrapper/fp32/dequant_int8_to_fp32_wrapper.h"
namespace mindspore::lite::micro::nnacl {

class NNaclFp32Serializer : public Serializer {
 public:
  NNaclFp32Serializer() = default;
  ~NNaclFp32Serializer() override = default;
  void CodeStruct(const std::string &name, const PoolingParameter &pooling_parameter);
  void CodeStruct(const std::string &name, const SoftmaxParameter &softmax_parameter);
  void CodeStruct(const std::string &name, const BatchNormParameter &batch_norm_parameter);
  void CodeStruct(const std::string &name, const ArithmeticParameter &arithmetic_parameter);
  void CodeStruct(const std::string &name, const ConvParameter &conv_parameter);
  void CodeStruct(const std::string &name, const MatMulParameter &mat_mul_parameter);
  void CodeStruct(const std::string &name, const ScaleParameter &scale_parameter);
  void CodeStruct(const std::string &name, const SliceParameter &slice_parameter);
  void CodeStruct(const std::string &name, const TileParameter &tile_parameter);
  void CodeStruct(const std::string &name, const TransposeParameter &transpose_parameter);
  void CodeStruct(const std::string &name, const DeQuantArg &de_quant_arg);
  void CodeStruct(const std::string &name, const SpliceParameter &splice_parameter);
};

}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_FP32_ERIALIZER_H_
