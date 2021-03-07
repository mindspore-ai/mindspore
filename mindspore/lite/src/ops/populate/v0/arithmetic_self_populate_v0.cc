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

#include "schema/model_v0_generated.h"
#include "src/common/log_adapter.h"
#include "nnacl/arithmetic_self_parameter.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateArithmeticSelfV0(const void *prim) {
  ArithmeticSelfParameter *arithmetic_self_param =
    reinterpret_cast<ArithmeticSelfParameter *>(malloc(sizeof(ArithmeticSelfParameter)));
  if (arithmetic_self_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticSelfParameter failed.";
    return nullptr;
  }
  memset(arithmetic_self_param, 0, sizeof(ArithmeticSelfParameter));
  auto primitive = static_cast<const schema::v0::Primitive *>(prim);
  int type = primitive->value_type();
  if (type == schema::v0::PrimitiveType_Abs) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Abs;
  } else if (type == schema::v0::PrimitiveType_Cos) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Cos;
  } else if (type == schema::v0::PrimitiveType_Sin) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Sin;
  } else if (type == schema::v0::PrimitiveType_Log) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Log;
  } else if (type == schema::v0::PrimitiveType_Neg) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Neg;
  } else if (type == schema::v0::PrimitiveType_NegGrad) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_NegGrad;
  } else if (type == schema::v0::PrimitiveType_LogGrad) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_LogGrad;
  } else if (type == schema::v0::PrimitiveType_Sqrt) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Sqrt;
  } else if (type == schema::v0::PrimitiveType_Square) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Square;
  } else if (type == schema::v0::PrimitiveType_Rsqrt) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Rsqrt;
  } else if (type == schema::v0::PrimitiveType_LogicalNot) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_LogicalNot;
  } else if (type == schema::v0::PrimitiveType_Floor) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Floor;
  } else if (type == schema::v0::PrimitiveType_Ceil) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Ceil;
  } else if (type == schema::v0::PrimitiveType_Round) {
    arithmetic_self_param->op_parameter_.type_ = schema::PrimitiveType_Round;
  }
  return reinterpret_cast<OpParameter *>(arithmetic_self_param);
}
}  // namespace

Registry g_absV0ParameterRegistry(schema::v0::PrimitiveType_Abs, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_cosV0ParameterRegistry(schema::v0::PrimitiveType_Cos, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_sinV0ParameterRegistry(schema::v0::PrimitiveType_Sin, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_logV0ParameterRegistry(schema::v0::PrimitiveType_Log, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_negV0ParameterRegistry(schema::v0::PrimitiveType_Neg, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_negGradV0ParameterRegistry(schema::v0::PrimitiveType_NegGrad, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_logGradV0ParameterRegistry(schema::v0::PrimitiveType_LogGrad, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_sqrtV0ParameterRegistry(schema::v0::PrimitiveType_Sqrt, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_squareV0ParameterRegistry(schema::v0::PrimitiveType_Square, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_rsqrtV0ParameterRegistry(schema::v0::PrimitiveType_Rsqrt, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_logicalNotV0ParameterRegistry(schema::v0::PrimitiveType_LogicalNot, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_floorV0ParameterRegistry(schema::v0::PrimitiveType_Floor, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_ceilV0ParameterRegistry(schema::v0::PrimitiveType_Ceil, PopulateArithmeticSelfV0, SCHEMA_V0);
Registry g_roundV0ParameterRegistry(schema::v0::PrimitiveType_Round, PopulateArithmeticSelfV0, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
