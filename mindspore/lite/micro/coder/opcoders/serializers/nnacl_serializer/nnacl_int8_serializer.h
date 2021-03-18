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
#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_INT8_SERIALIZER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_INT8_SERIALIZER_H_
#include <ostream>
#include <string>
#include "nnacl/pooling_parameter.h"
#include "nnacl/softmax_parameter.h"
#include "coder/opcoders/serializers/serializer.h"
#include "nnacl/op_base.h"
#include "nnacl/int8/add_int8.h"
#include "nnacl/int8/arithmetic_int8.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/int8/concat_int8.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/reshape_parameter.h"
#include "nnacl/slice_parameter.h"
#include "nnacl/batchnorm_parameter.h"
#include "nnacl/int8/relux_int8.h"

namespace mindspore::lite::micro::nnacl {
class NNaclInt8Serializer : public Serializer {
 public:
  NNaclInt8Serializer() = default;
  ~NNaclInt8Serializer() override = default;
  void CodeStruct(const std::string &name, const ConvParameter &conv_parameter);
  void CodeStruct(const std::string &name, const MatMulParameter &matmul_parameter);
  void CodeStruct(const std::string &name, const AddQuantParameter &add_quant_parameter);
  void CodeStruct(const std::string &name, const ArithmeticParameter &arithmetic_parameter);
  void CodeStruct(const std::string &name, const PoolingParameter &pooling_parameter);
  void CodeStruct(const std::string &name, const SoftmaxParameter &softmax_parameter);
  void CodeStruct(const std::string &name, const SliceParameter &slice_parameter);
  void CodeStruct(const std::string &name, const BatchNormParameter &batchnorm_parameter);
  void CodeStruct(const std::string &name, const SoftmaxQuantArg &softmax_quant_parameter);
  void CodeStruct(const std::string &name, const ConcatParameter &concat_parameter, int input_tensors, int in_shape,
                  int out_shape);
  void CodeStruct(const std::string &name, const ::QuantArg &quant_arg);
  void CodeStruct(const std::string &name, const ::QuantMulArg &quant_mul_arg);
  void CodeStruct(const std::string &name, const ReduceQuantArg &reduce_quant_arg);
  void CodeStruct(const std::string &name, const ReshapeQuantArg &reshape_quant_arg);
  void CodeStruct(const std::string &name, const MatmulQuantParameter &matmul_quant_arg, int weight_quant_num);
  void CodeStruct(const std::string &name, const SubQuantArg &sub_quant_arg);
  void CodeStruct(const std::string &name, const DivQuantArg &div_quant_arg);
  void CodeStruct(const std::string &name, const ReluXQuantArg &relu_quant_arg);
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_INT8_SERIALIZER_H_
