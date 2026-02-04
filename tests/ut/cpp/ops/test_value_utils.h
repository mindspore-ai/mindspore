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

#ifndef MINDSPORE_TESTS_UT_CPP_OPS_TEST_NUMBER_CONTAINER_H_
#define MINDSPORE_TESTS_UT_CPP_OPS_TEST_NUMBER_CONTAINER_H_

#include <utility>
#include <vector>
#include "ir/tensor_new.h"
#include "abstract/abstract_value.h"
#include "mindapi/base/format.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace ops {
class NumberContainer {
 public:
  NumberContainer() {}
  NumberContainer(int8_t v) { value_ = std::make_shared<Int8Imm>(v); }
  NumberContainer(int16_t v) { value_ = std::make_shared<Int16Imm>(v); }
  NumberContainer(int32_t v) { value_ = std::make_shared<Int32Imm>(v); }
  NumberContainer(int64_t v) { value_ = std::make_shared<Int64Imm>(v); }
  NumberContainer(uint8_t v) { value_ = std::make_shared<UInt8Imm>(v); }
  NumberContainer(uint16_t v) { value_ = std::make_shared<UInt16Imm>(v); }
  NumberContainer(uint32_t v) { value_ = std::make_shared<UInt32Imm>(v); }
  NumberContainer(uint64_t v) { value_ = std::make_shared<UInt64Imm>(v); }
  NumberContainer(float v) { value_ = std::make_shared<FP32Imm>(v); }
  NumberContainer(double v) { value_ = std::make_shared<FP64Imm>(v); }
  NumberContainer(bool v) { value_ = std::make_shared<BoolImm>(v); }
  NumberContainer(ValuePtr v) { value_ = std::move(v); }

  ValuePtr value_;
};

template <typename T>
ValuePtr CreateScalar(T v) {
  return std::make_shared<NumberContainer>(v)->value_;
}

template <typename T>
tensor::TensorPtr CreateTensor(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
  return tensor;
}

static inline ValuePtr CreatePyInt(int v) { return std::make_shared<Int64Imm>(v); }

static inline ValuePtr CreateTuple(const std::vector<NumberContainer> &values) {
  std::vector<ValuePtr> value_vec;
  value_vec.reserve(values.size());
  for (const auto &v : values) {
    value_vec.push_back(v.value_);
  }
  return std::make_shared<ValueTuple>(value_vec);
}

template <typename T>
ValuePtrList CreateScalarList(const std::vector<T> &values) {
  ValuePtrList value_list;
  std::transform(values.cbegin(), values.cend(), std::back_inserter(value_list),
                 [](const T &v) { return CreateScalar<T>(v); });
  return value_list;
}

template <typename T>
ValuePtr CreateScalarTuple(const std::vector<T> &values) {
  return std::make_shared<ValueTuple>(CreateScalarList<T>(values));
}

static inline ValuePtr CreateList(const std::vector<NumberContainer> &values) {
  std::vector<ValuePtr> value_vec;
  value_vec.reserve(values.size());
  for (const auto &v : values) {
    value_vec.push_back(v.value_);
  }
  return std::make_shared<ValueList>(value_vec);
}

ValuePtr CreatePyIntList(const std::vector<NumberContainer> &values);

ValuePtr CreatePyIntTuple(const std::vector<NumberContainer> &values);

static inline Format FormatStringToEnum(const std::string &format) {
  std::unordered_map<std::string, mindspore::Format> kStringToEnumMap = {
    {kOpFormat_NHWC, Format::NHWC}, {kOpFormat_NCHW, Format::NCHW}, {kOpFormat_NCDHW, Format::NCDHW}};
  auto iter = kStringToEnumMap.find(format);
  if (iter == kStringToEnumMap.end()) {
    MS_LOG(WARNING) << "Unsupported format [" << format << "].";
    return mindspore::DEFAULT_FORMAT;
  }
  return iter->second;
}
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_TESTS_UT_CPP_OPS_TEST_NUMBER_CONTAINER_H_
