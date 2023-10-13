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

#ifndef MINDSPORE_CCSRC_C_API_SRC_DYNAMIC_OP_INFO_H_
#define MINDSPORE_CCSRC_C_API_SRC_DYNAMIC_OP_INFO_H_

#include <vector>
#include <utility>
#include <string>
#include <map>
#include <memory>
#include "base/base.h"
#include "include/c_api/ms/node.h"
#include "c_api/src/common.h"

struct InnerOpInfo {
  std::string op_name;
  std::vector<ValuePtr> input_values{};
  std::vector<ShapeVector> input_shapes{};
  std::vector<DataTypeC> input_dtypes{};
  std::vector<ShapeVector> output_shapes{};
  std::vector<DataTypeC> output_dtypes{};
  std::vector<std::pair<std::string, ValuePtr>> attrs{};

  InnerOpInfo(const char *op_type, const std::vector<ValuePtr> &inputs, const std::vector<ShapeVector> &out_shapes,
              const std::vector<DataTypeC> &out_dtypes,
              const std::vector<std::pair<std::string, ValuePtr>> &attrs_pair) {
    op_name = op_type;
    for (auto input : inputs) {
      MS_EXCEPTION_IF_NULL(input);
      if (input->isa<TensorImpl>()) {
        auto in_tensor = input->cast<TensorPtr>();
        (void)input_shapes.emplace_back(in_tensor->shape());
        (void)input_dtypes.emplace_back(DataTypeC(in_tensor->data_type_c()));
      } else {
        (void)input_values.emplace_back(input);
      }
    }
    output_shapes = out_shapes;
    output_dtypes = out_dtypes;
    attrs = attrs_pair;
  }

  bool operator==(const InnerOpInfo &op_info) const {
    return op_name == op_info.op_name && input_values == op_info.input_values && input_shapes == op_info.input_shapes &&
           output_shapes == op_info.output_shapes && input_dtypes == op_info.input_dtypes &&
           output_dtypes == op_info.output_dtypes && attrs == op_info.attrs;
  }
};

template <>
struct std::hash<std::vector<ValuePtr>> {
  size_t operator()(const std::vector<ValuePtr> &value_ptr_vec) const {
    size_t res = 17;
    for (const auto &value_ptr : value_ptr_vec) {
      res = res * 31 + std::hash<ValuePtr>()(value_ptr);
    }
    return res;
  }
};

template <>
struct std::hash<std::vector<int64_t>> {
  size_t operator()(const std::vector<int64_t> &value_ptr_vec) const {
    size_t res = 17;
    for (const auto &value_ptr : value_ptr_vec) {
      res = res * 31 + std::hash<int64_t>()(value_ptr);
    }
    return res;
  }
};

template <>
struct std::hash<std::vector<ShapeVector>> {
  size_t operator()(const std::vector<ShapeVector> &shape_vec) const {
    size_t res = 17;
    for (const auto &shape : shape_vec) {
      res = res * 31 + std::hash<ShapeVector>()(shape);
    }
    return res;
  }
};

template <>
struct std::hash<std::vector<DataTypeC>> {
  size_t operator()(const std::vector<DataTypeC> &dtype_vec) const {
    size_t res = 17;
    for (const auto &dtype : dtype_vec) {
      res = res * 31 + std::hash<DataTypeC>()(dtype);
    }
    return res;
  }
};

template <>
struct std::hash<std::vector<std::pair<std::string, ValuePtr>>> {
  size_t operator()(const std::vector<std::pair<std::string, ValuePtr>> &attrs_vec) const {
    size_t res = 17;
    for (const auto &attr : attrs_vec) {
      res = res * 31 + std::hash<std::string>()(attr.first);
      res = res * 31 + std::hash<ValuePtr>()(attr.second);
    }
    return res;
  }
};

template <>
struct std::hash<InnerOpInfo> {
  size_t operator()(const InnerOpInfo &op_info) const {
    size_t res = 17;
    res = res * 31 + std::hash<std::string>()(op_info.op_name);
    res = res * 31 + std::hash<std::vector<ValuePtr>>()(op_info.input_values);
    res = res * 31 + std::hash<std::vector<ShapeVector>>()(op_info.input_shapes);
    res = res * 31 + std::hash<std::vector<ShapeVector>>()(op_info.output_shapes);
    res = res * 31 + std::hash<std::vector<DataTypeC>>()(op_info.input_dtypes);
    res = res * 31 + std::hash<std::vector<DataTypeC>>()(op_info.output_dtypes);
    res = res * 31 + std::hash<std::vector<std::pair<std::string, ValuePtr>>>()(op_info.attrs);
    return res;
  }
};
#endif  // MINDSPORE_CCSRC_C_API_SRC_DYNAMIC_OP_INFO_H_
