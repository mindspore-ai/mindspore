/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "src/common/ops/populate/populate_register.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "nnacl/custom_parameter.h"
#include "nnacl/split_parameter.h"
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace lite {
bool GetDataFromPrim(void *dst, size_t len, const schema::Custom *custom_prim, size_t index) {
  auto data_bytes = custom_prim->attr()->Get(index)->data();
  auto data_size = data_bytes->size();
  if (len < data_size) {
    return false;
  }
  std::vector<uint8_t> buf(data_size, 0);
  for (size_t i = 0; i < data_size; ++i) {
    buf[i] = static_cast<char>(data_bytes->Get(i));
  }
  (void)memcpy(dst, buf.data(), data_size);
  return true;
}

OpParameter *PopulateCustomParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Custom();
  if (value == nullptr) {
    MS_LOG(ERROR) << "the value is nullptr.";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(value->type() != nullptr, nullptr);
  std::string type = value->type()->c_str();
  if (type == "ShapeFusion") {
    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc ShapeParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(OpParameter));
    param->type_ = PrimType_Inner_ShapeFusion;
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "GraphKernel") {
    auto *param = static_cast<CustomParameter *>(malloc(sizeof(CustomParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc CustomParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(CustomParameter));
    param->op_parameter_.type_ = PrimType_Inner_GraphKernel;
    // Just use the attr_data pointer to save the prim directly, the inner value is parsed as necessary.
    param->attr_data[0] = static_cast<char *>(const_cast<void *>(prim));
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "SplitReduceConcatFusion") {
    if (value->attr()->size() < 1) {
      return nullptr;
    }
    SplitParameter *param = static_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc SplitParameter failed.";
      return nullptr;
    }
    if (!GetDataFromPrim(param, sizeof(SplitParameter), value, 0)) {
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
    if (!GetDataFromPrim(param->split_sizes_, split_sizes_size, value, 1)) {
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
  } else if (type == "FSEDecode") {
    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc FSEDecode failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(OpParameter));
    param->type_ = PrimType_Inner_FseDecode;
    return reinterpret_cast<OpParameter *>(param);
  } else {
    MS_LOG(ERROR) << "Unsupported custom type: " << type;
  }
  return nullptr;
}

REG_POPULATE(PrimType_Custom, PopulateCustomParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
