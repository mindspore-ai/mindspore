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
#include "src/ops/populate/v0/arithmetic_populate_v0.h"
#include "src/common/log_adapter.h"
#include "src/ops/populate/populate_register.h"
#include "src/common/common.h"

namespace mindspore {
namespace lite {
ArithmeticParameter *PopulateArithmeticV0CommonPara(const void *prim) {
  auto *param = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ArithmeticParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ArithmeticParameter));
  const auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  param->op_parameter_.type_ = primitive->value_type();
  param->broadcasting_ = false;
  param->ndim_ = 0;
  param->activation_type_ = 0;
  return param;
}

OpParameter *PopulateArithmeticV0(const void *primitive) {
  ArithmeticParameter *param = PopulateArithmeticV0CommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  int type = param->op_parameter_.type_;
  if (type == schema::v0::PrimitiveType_RealDiv) {
    param->op_parameter_.type_ = schema::PrimitiveType_RealDiv;
  } else if (type == schema::v0::PrimitiveType_LogicalAnd) {
    param->op_parameter_.type_ = schema::PrimitiveType_LogicalAnd;
  } else if (type == schema::v0::PrimitiveType_LogicalOr) {
    param->op_parameter_.type_ = schema::PrimitiveType_LogicalOr;
  } else if (type == schema::v0::PrimitiveType_Equal) {
    param->op_parameter_.type_ = schema::PrimitiveType_Equal;
  } else if (type == schema::v0::PrimitiveType_NotEqual) {
    param->op_parameter_.type_ = schema::PrimitiveType_NotEqual;
  } else if (type == schema::v0::PrimitiveType_Less) {
    param->op_parameter_.type_ = schema::PrimitiveType_Less;
  } else if (type == schema::v0::PrimitiveType_LessEqual) {
    param->op_parameter_.type_ = schema::PrimitiveType_LessEqual;
  } else if (type == schema::v0::PrimitiveType_Greater) {
    param->op_parameter_.type_ = schema::PrimitiveType_Greater;
  } else if (type == schema::v0::PrimitiveType_GreaterEqual) {
    param->op_parameter_.type_ = schema::PrimitiveType_GreaterEqual;
  } else if (type == schema::v0::PrimitiveType_Maximum) {
    param->op_parameter_.type_ = schema::PrimitiveType_Maximum;
  } else if (type == schema::v0::PrimitiveType_Minimum) {
    param->op_parameter_.type_ = schema::PrimitiveType_Minimum;
  } else if (type == schema::v0::PrimitiveType_FloorDiv) {
    param->op_parameter_.type_ = schema::PrimitiveType_FloorDiv;
  } else if (type == schema::v0::PrimitiveType_FloorMod) {
    param->op_parameter_.type_ = schema::PrimitiveType_FloorMod;
  }
  return reinterpret_cast<OpParameter *>(param);
}

Registry g_realDivV0ParameterRegistry(schema::v0::PrimitiveType_RealDiv, PopulateArithmeticV0, SCHEMA_V0);
Registry g_logicalAndV0ParameterRegistry(schema::v0::PrimitiveType_LogicalAnd, PopulateArithmeticV0, SCHEMA_V0);
Registry g_logicalOrV0parameterRegistry(schema::v0::PrimitiveType_LogicalOr, PopulateArithmeticV0, SCHEMA_V0);
Registry g_equalV0ParameterRegistry(schema::v0::PrimitiveType_Equal, PopulateArithmeticV0, SCHEMA_V0);
Registry g_notEqualV0ParameterRegistry(schema::v0::PrimitiveType_NotEqual, PopulateArithmeticV0, SCHEMA_V0);
Registry g_lessV0ParameterRegistry(schema::v0::PrimitiveType_Less, PopulateArithmeticV0, SCHEMA_V0);
Registry g_lessEqualV0ParameterRegistry(schema::v0::PrimitiveType_LessEqual, PopulateArithmeticV0, SCHEMA_V0);
Registry g_greaterV0ParameterRegistry(schema::v0::PrimitiveType_Greater, PopulateArithmeticV0, SCHEMA_V0);
Registry g_greaterEqualV0ParameterRegistry(schema::v0::PrimitiveType_GreaterEqual, PopulateArithmeticV0, SCHEMA_V0);
Registry g_maximumV0ParameterRegistry(schema::v0::PrimitiveType_Maximum, PopulateArithmeticV0, SCHEMA_V0);
Registry g_minimumV0ParameterRegistry(schema::v0::PrimitiveType_Minimum, PopulateArithmeticV0, SCHEMA_V0);
Registry g_floorDivV0ParameterRegistry(schema::v0::PrimitiveType_FloorDiv, PopulateArithmeticV0, SCHEMA_V0);
Registry g_floorModV0ParameterRegistry(schema::v0::PrimitiveType_FloorMod, PopulateArithmeticV0, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
