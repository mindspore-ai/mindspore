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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ACME_ACME_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ACME_ACME_HELPER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include "acme/include/acme.h"
#include "include/api/format.h"
#include "ir/dtype/type_id.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
namespace kernel {
inline acme::ShapeInfo TransAcmeShape(const ShapeVector &shape) { return shape; }

std::string TransAcmeOpName(const std::string &ms_op_name);

acme::DataType TransAcmeDataType(TypeId ms_type);

acme::TensorFormat TransAcmeFormat(Format format);

inline acme::ArgDescPtr MakeDefaultArgDesc(acme::TensorFormat format, acme::DataType dtype) {
  acme::ShapeInfo shape{1};
  return std::make_shared<acme::ArgDesc>(shape, dtype, format);
}

class NameMapper {
 public:
  NameMapper() = default;
  ~NameMapper() = default;

  static NameMapper &GetInstance() {
    static NameMapper name_mammer;
    return name_mammer;
  }

  inline std::string GetAcmeName(const std::string &ms_name) const {
    auto iter = ms_to_acme_mapper_.find(ms_name);
    if (iter == ms_to_acme_mapper_.end()) {
      return "";
    }

    return iter->second;
  }

  inline void Insert(const std::string &ms_name, const std::string &acme_name) {
    ms_to_acme_mapper_[ms_name] = acme_name;
  }

 private:
  std::unordered_map<std::string, std::string> ms_to_acme_mapper_;
};

class NameMappingRegistrar {
 public:
  NameMappingRegistrar(const std::string &ms_name, const std::string &acme_name) {
    NameMapper::GetInstance().Insert(ms_name, acme_name);
  }
  ~NameMappingRegistrar() = default;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_ACME_ACME_HELPER_H_
