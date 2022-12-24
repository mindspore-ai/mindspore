/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OPINFO_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OPINFO_H_
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <utility>
#include <unordered_map>
#include "ir/dtype.h"
#include "kernel/kernel.h"
#include "kernel/oplib/op_info_keys.h"

namespace mindspore::kernel {
class OpAttr {
 public:
  OpAttr() = default;
  ~OpAttr() = default;

  std::string name() const { return name_; }
  std::string param_type() const { return param_type_; }
  std::string type() const { return type_; }
  std::string value() const { return value_; }
  std::string default_value() const { return default_value_; }

  void set_name(const std::string &name) { name_ = name; }
  void set_param_type(const std::string &param_type) { param_type_ = param_type; }
  void set_type(const std::string &type) { type_ = type; }
  void set_value(const std::string &value) { value_ = value; }
  void set_default_value(const std::string &default_value) { default_value_ = default_value; }

 private:
  std::string name_;
  std::string param_type_;
  std::string type_;
  std::string value_;
  std::string default_value_;
};

class OpIOInfo {
 public:
  OpIOInfo() = default;
  ~OpIOInfo() = default;

  int index() const { return index_; }
  const std::string &name() const { return name_; }
  bool need_compile() const { return need_compile_; }
  const std::string &param_type() const { return param_type_; }
  const std::string &reshape_type() const { return reshape_type_; }
  const std::string &shape() const { return shape_; }
  const std::vector<std::string> &dtypes() const { return dtypes_; }
  const std::vector<std::string> &formats() const { return formats_; }
  const std::vector<std::string> &unknown_shape_formats() const { return unknown_shape_formats_; }
  const std::vector<std::string> &object_types() const { return object_types_; }
  const std::string &value_depend() const { return value_depend_; }
  const std::string &shapes_type() const { return shapes_type_; }

  void set_index(const int index) { index_ = index; }
  void set_name(const std::string &name) { name_ = name; }
  void set_need_compile(const bool need_compile) { need_compile_ = need_compile; }
  void set_param_type(const std::string &param_type) { param_type_ = param_type; }
  void set_reshape_type(const std::string &reshape_type) { reshape_type_ = reshape_type; }
  void set_shape(const std::string &shape) { shape_ = shape; }
  void set_dtypes(const std::vector<std::string> &dtype) { dtypes_ = dtype; }
  void set_formats(const std::vector<std::string> &formats) { formats_ = formats; }
  void set_unknown_shape_formats(const std::vector<std::string> &unknown_shape_formats) {
    unknown_shape_formats_ = unknown_shape_formats;
  }
  void set_object_types(const std::vector<std::string> &object_types) { object_types_ = object_types; }
  void set_value_depend(const std::string &value_depend) { value_depend_ = value_depend; }
  void set_shapes_type(const std::string &shapes_type) { shapes_type_ = shapes_type; }

 private:
  int index_ = 0;
  std::string name_;
  bool need_compile_ = false;
  std::string param_type_;
  std::string reshape_type_;
  std::string shape_;
  std::string shapes_type_;
  std::string value_depend_ = kIgnored;
  std::vector<std::string> dtypes_;
  std::vector<std::string> formats_;
  std::vector<std::string> unknown_shape_formats_;
  std::vector<std::string> object_types_;
};

class OpInfo {
 public:
  OpInfo() = default;
  ~OpInfo() = default;
  std::string op_name() const { return op_name_; }
  OpImplyType imply_type() const { return imply_type_; }
  bool async() const { return async_; }
  std::string bin_file() const { return bin_file_; }
  int compute() const { return compute_; }
  bool cube_op() const { return cube_op_; }
  bool dynamic_compile_static() const { return dynamic_compile_static_; }
  bool dynamic_format() const { return dynamic_format_; }
  bool dynamic_rank_support() const { return dynamic_rank_support_; }
  bool dynamic_shape_support() const { return dynamic_shape_support_; }
  bool heavy_op() const { return heavy_op_; }
  bool jit_compile() const { return jit_compile_; }
  std::string kernel() const { return kernel_; }
  bool need_check_support() const { return need_check_support_; }
  OpPattern op_pattern() const { return op_pattern_; }
  std::string op_file() const { return op_file_; }
  std::string op_interface() const { return op_interface_; }
  bool partial() const { return partial_; }
  bool precision_reduce() const { return precision_reduce_; }
  std::string range_limit() const { return range_limit_; }
  const std::vector<std::string> &sagt_key_attrs() const { return sagt_key_attrs_; }
  std::string slice_pattern() const { return slice_pattern_; }
  std::string prebuild_pattern() const { return prebuild_pattern_; }
  // Attr
  std::string impl_path() const { return impl_path_; }
  std::string processor() const { return processor_; }
  const std::vector<size_t> &input_to_attr_index() const { return input_to_attr_index_; }
  const std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>> &real_input_index() const {
    return real_input_index_;
  }
  const std::unordered_map<size_t, size_t> &ref_infos() const { return ref_infos_; }

  std::vector<std::shared_ptr<OpAttr>> attrs_ptr() const { return attrs_ptr_; }
  std::vector<std::shared_ptr<OpIOInfo>> inputs_ptr() const { return inputs_ptr_; }
  std::vector<std::shared_ptr<OpIOInfo>> outputs_ptr() const { return outputs_ptr_; }

  void set_op_name(const std::string &op_name) { op_name_ = op_name; }
  void set_imply_type(OpImplyType imply_type) { imply_type_ = imply_type; }
  void set_async(const bool async) { async_ = async; }
  void set_bin_file(const std::string &bin_file) { bin_file_ = bin_file; }
  void set_compute(const int compute) { compute_ = compute; }
  void set_cube_op(bool cube_op) { cube_op_ = cube_op; }
  void set_dynamic_compile_static(bool dynamic_compile_static) { dynamic_compile_static_ = dynamic_compile_static; }
  void set_dynamic_format(bool dynamic_format) { dynamic_format_ = dynamic_format; }
  void set_dynamic_rank_support(bool dynamic_rank_support) { dynamic_rank_support_ = dynamic_rank_support; }
  void set_dynamic_shape_support(bool flag) { dynamic_shape_support_ = flag; }
  void set_heavy_op(bool heavy_op) { heavy_op_ = heavy_op; }
  void set_jit_compile(bool jit_compile) { jit_compile_ = jit_compile; }
  void set_soft_sync(bool soft_sync) { soft_sync_ = soft_sync; }
  void set_op_impl_switch(const std::string &op_impl_switch) { op_impl_switch_ = op_impl_switch; }
  void set_kernel(const std::string &kernel_name) { kernel_ = kernel_name; }
  void set_need_check_supported(bool need_check_supported) { need_check_support_ = need_check_supported; }
  void set_op_pattern(const OpPattern op_pattern) { op_pattern_ = op_pattern; }
  void set_op_file(const std::string &op_file) { op_file_ = op_file; }
  void set_op_interface(const std::string &op_interface) { op_interface_ = op_interface; }
  void set_partial(const bool partial_flag) { partial_ = partial_flag; }
  void set_precision_reduce(bool precision_reduce) { precision_reduce_ = precision_reduce; }
  void set_range_limit(const std::string &range_limit) { range_limit_ = range_limit; }
  void set_sagt_key_attrs(const std::vector<std::string> &sagt_key_attrs) { sagt_key_attrs_ = sagt_key_attrs; }
  void set_slice_pattern(const std::string &slice_pattern) { slice_pattern_ = slice_pattern; }
  void set_prebuild_pattern(const std::string &prebuild_pattern) { prebuild_pattern_ = prebuild_pattern; }

  void set_impl_path(const std::string &impl_path) { impl_path_ = impl_path; }
  void set_processor(const std::string &processor) { processor_ = processor; }
  void set_input_to_attr_index(const std::vector<size_t> &input_to_attr_index) {
    input_to_attr_index_ = input_to_attr_index;
  }
  void set_real_input_index(const std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>> &real_input_index) {
    real_input_index_ = real_input_index;
  }
  bool is_ref() const { return !ref_infos_.empty(); }
  bool has_ref_index(size_t out_index) const { return ref_infos_.find(out_index) != ref_infos_.end(); }
  void add_ref_pair(size_t out_index, size_t in_index) { (void)ref_infos_.emplace(out_index, in_index); }
  void add_attrs_ptr(const std::shared_ptr<OpAttr> &attr) { attrs_ptr_.push_back(attr); }
  void add_inputs_ptr(const std::shared_ptr<OpIOInfo> &input) { inputs_ptr_.push_back(input); }
  void set_inputs_ptr(const std::vector<std::shared_ptr<OpIOInfo>> &inputs) { inputs_ptr_ = inputs; }
  void add_outputs_ptr(const std::shared_ptr<OpIOInfo> &output) { outputs_ptr_.push_back(output); }

  bool equals_to(const std::shared_ptr<OpInfo> &other_info) const {
    return this->op_name_ == other_info->op_name_ && this->imply_type_ == other_info->imply_type_ &&
           this->processor_ == other_info->processor_ && this->op_pattern_ == other_info->op_pattern_ &&
           this->dynamic_shape_support_ == other_info->dynamic_shape_support_ &&
           this->dynamic_compile_static_ == other_info->dynamic_compile_static_;
  }

 private:
  std::string op_name_;
  OpImplyType imply_type_ = kImplyTBE;
  bool async_ = false;
  std::string bin_file_;
  int compute_ = 0;
  bool cube_op_ = false;
  bool dynamic_compile_static_ = false;
  bool dynamic_format_ = false;
  bool dynamic_rank_support_ = false;
  bool dynamic_shape_support_ = false;
  bool heavy_op_ = false;
  bool jit_compile_ = false;
  bool soft_sync_ = false;
  std::string op_impl_switch_ = "";
  std::string kernel_;
  bool need_check_support_ = false;
  OpPattern op_pattern_ = kCommonPattern;
  std::string op_file_;
  std::string op_interface_;
  bool partial_ = false;
  bool precision_reduce_ = false;
  std::string range_limit_;
  std::vector<std::string> sagt_key_attrs_ = {};
  std::string slice_pattern_;
  std::string prebuild_pattern_;
  // Attr info
  std::vector<std::shared_ptr<OpAttr>> attrs_ptr_;
  // Input/Output info
  std::vector<std::shared_ptr<OpIOInfo>> inputs_ptr_;
  std::vector<std::shared_ptr<OpIOInfo>> outputs_ptr_;

  // Attr not in the json
  std::string impl_path_;
  std::string processor_;
  std::vector<size_t> input_to_attr_index_{};
  std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>> real_input_index_{{}, {}};
  std::unordered_map<size_t, size_t> ref_infos_;
};

using OpAttrPtr = std::shared_ptr<OpAttr>;
using OpIOInfoPtr = std::shared_ptr<OpIOInfo>;
using OpInfoPtr = std::shared_ptr<OpInfo>;
BACKEND_EXPORT std::vector<std::string> SplitStrToVec(const std::string &input);
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OPINFO_H_
