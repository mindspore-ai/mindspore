/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_OPS_POPULATE_POPULATE_REGISTER_H_
#define MINDSPORE_LITE_SRC_COMMON_OPS_POPULATE_POPULATE_REGISTER_H_

#include <map>
#include <vector>
#include <string>

#include "schema/model_generated.h"
#include "nnacl/op_base.h"
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "src/common/prim_util.h"
#include "src/common/version_manager.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"

namespace mindspore {
constexpr int kOffsetTwo = 2;
constexpr int kOffsetThree = 3;
constexpr size_t kMinShapeSizeTwo = 2;
constexpr size_t kMinShapeSizeFour = 4;
typedef OpParameter *(*BaseOperator2Parameter)(void *base_operator);

static const std::vector<schema::PrimitiveType> string_op = {
  schema::PrimitiveType_CustomExtractFeatures, schema::PrimitiveType_CustomNormalize,
  schema::PrimitiveType_CustomPredict,         schema::PrimitiveType_HashtableLookup,
  schema::PrimitiveType_LshProjection,         schema::PrimitiveType_SkipGram};

class BaseOperatorPopulateRegistry {
 public:
  static BaseOperatorPopulateRegistry *GetInstance();

  void InsertParameterMap(int type, BaseOperator2Parameter creator, int version = lite::SCHEMA_CUR) {
    parameters_[lite::GenPrimVersionKey(type, version)] = creator;
    std::string str = schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
    str_to_type_map_[str] = type;
  }

  BaseOperator2Parameter GetParameterCreator(int type, int version = lite::SCHEMA_CUR) {
    BaseOperator2Parameter param_creator = nullptr;
    auto iter = parameters_.find(lite::GenPrimVersionKey(type, version));
    if (iter == parameters_.end()) {
#ifdef STRING_KERNEL_CLIP
      if (lite::IsContain(string_op, static_cast<schema::PrimitiveType>(type))) {
        MS_LOG(ERROR) << unsupport_string_tensor_log;
        return nullptr;
      }
#endif
      MS_LOG(ERROR) << "Unsupported parameter type in Create : "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
      return nullptr;
    }
    param_creator = iter->second;
    return param_creator;
  }

  int TypeStrToType(const std::string &type_str) {
    auto iter = str_to_type_map_.find(type_str);
    if (iter == str_to_type_map_.end()) {
      MS_LOG(ERROR) << "Unknown type string to type " << type_str;
      return schema::PrimitiveType_NONE;
    }
    return iter->second;
  }

 protected:
  // key:type * 1000 + schema_version
  std::map<int, BaseOperator2Parameter> parameters_;
  std::map<std::string, int> str_to_type_map_;
};

class BaseRegistry {
 public:
  BaseRegistry(int primitive_type, BaseOperator2Parameter creator, int version = lite::SCHEMA_CUR) noexcept {
    BaseOperatorPopulateRegistry::GetInstance()->InsertParameterMap(primitive_type, creator, version);
  }
  ~BaseRegistry() = default;
};

#define REG_BASE_POPULATE(primitive_type, creator) \
  static BaseRegistry g_##primitive_type##base_populate(primitive_type, creator);
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_POPULATE_POPULATE_REGISTER_H_
