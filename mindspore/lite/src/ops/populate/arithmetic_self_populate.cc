/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/ops/arithmetic_self.h"
#include "src/common/log_adapter.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateArithmeticSelf(const mindspore::lite::PrimitiveC *primitive) {
  ArithmeticSelfParameter *arithmetic_self_param =
    reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (arithmetic_self_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticSelfParameter failed.";
    return nullptr;
  }
  memset(arithmetic_self_param, 0, sizeof(ArithmeticSelfParameter));
  arithmetic_self_param->op_parameter_.type_ = primitive->Type();
  return reinterpret_cast<OpParameter *>(arithmetic_self_param);
}

Registry AbsParameterRegistry(schema::PrimitiveType_Abs, PopulateArithmeticSelf);
Registry CosParameterRegistry(schema::PrimitiveType_Cos, PopulateArithmeticSelf);
Registry SinParameterRegistry(schema::PrimitiveType_Sin, PopulateArithmeticSelf);
Registry LogParameterRegistry(schema::PrimitiveType_Log, PopulateArithmeticSelf);
Registry NegParameterRegistry(schema::PrimitiveType_Neg, PopulateArithmeticSelf);
Registry NegGradParameterRegistry(schema::PrimitiveType_NegGrad, PopulateArithmeticSelf);
Registry LogGradParameterRegistry(schema::PrimitiveType_LogGrad, PopulateArithmeticSelf);
Registry AbsGradParameterRegistry(schema::PrimitiveType_AbsGrad, PopulateArithmeticSelf);
Registry SqrtParameterRegistry(schema::PrimitiveType_Sqrt, PopulateArithmeticSelf);
Registry SquareParameterRegistry(schema::PrimitiveType_Square, PopulateArithmeticSelf);
Registry RsqrtParameterRegistry(schema::PrimitiveType_Rsqrt, PopulateArithmeticSelf);
Registry LogicalNotParameterRegistry(schema::PrimitiveType_LogicalNot, PopulateArithmeticSelf);
Registry FloorParameterRegistry(schema::PrimitiveType_Floor, PopulateArithmeticSelf);
Registry CeilParameterRegistry(schema::PrimitiveType_Ceil, PopulateArithmeticSelf);
Registry RoundParameterRegistry(schema::PrimitiveType_Round, PopulateArithmeticSelf);
Registry ReciprocalParameterRegistry(schema::PrimitiveType_Reciprocal, PopulateArithmeticSelf);

}  // namespace lite
}  // namespace mindspore
