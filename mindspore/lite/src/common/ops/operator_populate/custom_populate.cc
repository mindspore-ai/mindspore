/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/custom_parameter.h"
#include "nnacl/split_parameter.h"
#include "ops/custom.h"
using mindspore::ops::kNameCustom;
using mindspore::schema::PrimitiveType_Custom;
namespace mindspore {
namespace lite {
namespace {
bool GetDataFromOp(void *dst, size_t len, const ops::Custom *custom_op, std::string index) {
  auto data_bytes = custom_op->get_attr()[index].data();
  auto data_size = custom_op->get_attr()[index].size();
  if (len < data_size) {
    return false;
  }
  std::vector<uint8_t> buf(data_size, 0);
  for (size_t i = 0; i < data_size; ++i) {
    buf[i] = static_cast<char>(data_bytes[i]);
  }
  (void)memcpy(dst, buf.data(), data_size);
  return true;
}
}  // namespace

OpParameter *PopulateCustomOpParameter(const BaseOperatorPtr &base_operator) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base_operator is nullptr";
    return nullptr;
  }
  auto op = dynamic_cast<ops::Custom *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not NLLLoss.";
    return nullptr;
  }

  auto type = op->get_type();
  if (type == "ShapeFusion") {
    auto param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc ShapeParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(OpParameter));
    param->type_ = PrimType_Inner_ShapeFusion;
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "GraphKernel") {
    auto param = static_cast<CustomParameter *>(malloc(sizeof(CustomParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc CustomParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(CustomParameter));
    param->op_parameter_.type_ = PrimType_Inner_GraphKernel;
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "SplitReduceConcatFusion") {
    if (op->get_attr().size() < 1) {
      return nullptr;
    }
    SplitParameter *param = static_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc SplitParameter failed.";
      return nullptr;
    }
    if (!GetDataFromOp(param, sizeof(SplitParameter), op, "0")) {
      MS_LOG(ERROR) << "Get SplitParameter value From prim fail.";
      free(param);
      return nullptr;
    }

    auto split_sizes_size = static_cast<size_t>(param->num_split_) * sizeof(int);
    param->split_sizes_ = reinterpret_cast<int *>(malloc(split_sizes_size));
    if (param->split_sizes_ == nullptr) {
      MS_LOG(ERROR) << "malloc split_sizes_ failed.";
      free(param);
      return nullptr;
    }
    if (!GetDataFromOp(param->split_sizes_, split_sizes_size, op, "1")) {
      MS_LOG(ERROR) << "Get split value From prim fail.";
      free(param->split_sizes_);
      free(param);
      return nullptr;
    }

    param->op_parameter_.type_ = PrimType_Inner_SplitReduceConcatFusion;
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "EncoderLayer") {
    std::cout << "EncoderLayer populate" << std::endl;
    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc EncoderLayer failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(OpParameter));
    param->type_ = PrimType_Inner_EncoderLayer;
    return reinterpret_cast<OpParameter *>(param);
  } else {
    MS_LOG(ERROR) << "Unsupported custom type: " << type;
  }
  return nullptr;
}

REG_OPERATOR_POPULATE(kNameCustom, PrimitiveType_Custom, PopulateCustomOpParameter)
}  // namespace lite
}  // namespace mindspore
