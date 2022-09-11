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

namespace mindspore::kernel {
enum OpImplyType { kAKG = 0, kTBE = 1, kAICPU = 2, kCPU = 3, kGPU = 4, kBISHENG };
enum OpIOType { kInput = 0, kOutput };
constexpr auto kIgnored = "ignored";

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
  const std::string &value_depend() const { return value_depend_; }

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
  void set_value_depend(const std::string &value_depend) { value_depend_ = value_depend; }

 private:
  int index_ = 0;
  std::string name_;
  bool need_compile_ = false;
  std::string param_type_;
  std::string reshape_type_;
  std::string shape_;
  std::vector<std::string> dtypes_;
  std::vector<std::string> formats_;
  std::vector<std::string> unknown_shape_formats_;
  std::string value_depend_ = kIgnored;
};

class OpInfo {
 public:
  OpInfo() = default;
  OpInfo(const OpInfo &opinfo) { *this = opinfo; }
  OpInfo &operator=(const OpInfo &opinfo) {
    if (this == &opinfo) {
      return *this;
    }
    op_name_ = opinfo.op_name();
    imply_type_ = opinfo.imply_type();

    impl_path_ = opinfo.impl_path();
    fusion_type_ = opinfo.fusion_type();
    async_flag_ = opinfo.async_flag_;
    binfile_name_ = opinfo.binfile_name_;
    compute_cost_ = opinfo.compute_cost_;
    kernel_name_ = opinfo.kernel_name();
    partial_flag_ = opinfo.partial_flag_;
    dynamic_shape_ = opinfo.dynamic_shape_;
    dynamic_compile_static_ = opinfo.dynamic_compile_static_;
    op_pattern_ = opinfo.op_pattern();
    processor_ = opinfo.processor_;
    input_to_attr_index_ = opinfo.input_to_attr_index_;
    real_input_index_ = opinfo.real_input_index_;
    need_check_supported_ = opinfo.need_check_supported();
    dynamic_rank_support_ = opinfo.dynamic_rank_support();
    is_dynamic_format_ = opinfo.is_dynamic_format();
    for (const auto &attr : opinfo.attrs_ptr()) {
      attrs_ptr_.push_back(std::make_shared<OpAttr>(*attr));
    }
    for (const auto &input : opinfo.inputs_ptr()) {
      inputs_ptr_.push_back(std::make_shared<OpIOInfo>(*input));
    }
    for (const auto &output : opinfo.outputs_ptr()) {
      outputs_ptr_.push_back(std::make_shared<OpIOInfo>(*output));
    }
    ref_infos_ = opinfo.ref_infos();
    return *this;
  }
  ~OpInfo() = default;
  std::string op_name() const { return op_name_; }
  OpImplyType imply_type() const { return imply_type_; }
  std::string impl_path() const { return impl_path_; }
  std::string fusion_type() const { return fusion_type_; }
  std::string kernel_name() const { return kernel_name_; }
  OpPattern op_pattern() const { return op_pattern_; }
  bool dynamic_shape() const { return dynamic_shape_; }
  bool dynamic_compile_static() const { return dynamic_compile_static_; }
  std::string processor() const { return processor_; }
  bool need_check_supported() const { return need_check_supported_; }
  bool dynamic_rank_support() const { return dynamic_rank_support_; }
  bool is_dynamic_format() const { return is_dynamic_format_; }
  std::vector<std::shared_ptr<OpAttr>> attrs_ptr() const { return attrs_ptr_; }
  std::vector<std::shared_ptr<OpIOInfo>> inputs_ptr() const { return inputs_ptr_; }
  std::vector<std::shared_ptr<OpIOInfo>> outputs_ptr() const { return outputs_ptr_; }
  const std::unordered_map<size_t, size_t> &ref_infos() const { return ref_infos_; }
  const std::vector<size_t> &input_to_attr_index() const { return input_to_attr_index_; }
  std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>> &real_input_index() { return real_input_index_; }
  void set_dynamic_shape(bool dynamic_shape) { dynamic_shape_ = dynamic_shape; }
  void set_dynamic_compile_static_(bool dynamic_compile_static) { dynamic_compile_static_ = dynamic_compile_static; }
  void set_op_name(const std::string &op_name) { op_name_ = op_name; }
  void set_imply_type(const OpImplyType imply_type) { imply_type_ = imply_type; }
  void set_impl_path(const std::string &impl_path) { impl_path_ = impl_path; }
  void set_fusion_type(const std::string &fusion_type) { fusion_type_ = fusion_type; }
  void set_async_flag(const bool async_flag) { async_flag_ = async_flag; }
  void set_binfile_name(const std::string &binfile_name) { binfile_name_ = binfile_name; }
  void set_compute_cost(const int compute_cost) { compute_cost_ = compute_cost; }
  void set_kernel_name(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  void set_partial_flag(const bool partial_flag) { partial_flag_ = partial_flag; }
  void set_op_pattern(const OpPattern op_pattern) { op_pattern_ = op_pattern; }
  void set_processor(const std::string &processor) { processor_ = processor; }
  void set_need_check_supported(bool need_check_supported) { need_check_supported_ = need_check_supported; }
  void set_dynamic_rank_support(bool dynamic_rank_support) { dynamic_rank_support_ = dynamic_rank_support; }
  void set_is_dynamic_format(bool is_dynamic_format) { is_dynamic_format_ = is_dynamic_format; }
  void set_input_to_attr_index(const std::vector<size_t> &input_to_attr_index) {
    input_to_attr_index_ = input_to_attr_index;
  }
  void set_real_input_index(const std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>> &real_input_index) {
    real_input_index_ = real_input_index;
  }
  void add_attrs_ptr(const std::shared_ptr<OpAttr> &attr) { attrs_ptr_.push_back(attr); }
  void add_inputs_ptr(const std::shared_ptr<OpIOInfo> &input) { inputs_ptr_.push_back(input); }
  void add_outputs_ptr(const std::shared_ptr<OpIOInfo> &output) { outputs_ptr_.push_back(output); }
  bool is_ref() const { return !ref_infos_.empty(); }
  bool has_ref_index(size_t out_index) const { return ref_infos_.find(out_index) != ref_infos_.end(); }
  void add_ref_pair(size_t out_index, size_t in_index) { (void)ref_infos_.emplace(out_index, in_index); }
  void ClearInputs() { inputs_ptr_.clear(); }
  void ClearOutputs() { outputs_ptr_.clear(); }
  bool equals_to(const std::shared_ptr<OpInfo> &other_info) const {
    return this->op_name_ == other_info->op_name_ && this->imply_type_ == other_info->imply_type_ &&
           this->processor_ == other_info->processor_ && this->op_pattern_ == other_info->op_pattern_ &&
           this->dynamic_shape_ == other_info->dynamic_shape_ &&
           this->dynamic_compile_static_ == other_info->dynamic_compile_static_;
  }

 private:
  std::string op_name_;
  OpImplyType imply_type_ = kTBE;
  std::string impl_path_;
  std::string fusion_type_;
  bool async_flag_ = false;
  std::string binfile_name_;
  int compute_cost_ = 0;
  std::string kernel_name_;
  bool partial_flag_ = false;
  bool dynamic_shape_ = false;
  bool dynamic_compile_static_ = false;
  bool need_check_supported_ = false;
  bool dynamic_rank_support_ = false;
  bool is_dynamic_format_ = false;
  OpPattern op_pattern_ = kCommonPattern;
  std::string processor_;
  std::vector<size_t> input_to_attr_index_{};
  std::pair<std::map<size_t, size_t>, std::map<size_t, size_t>> real_input_index_{{}, {}};
  std::vector<std::shared_ptr<OpAttr>> attrs_ptr_;
  std::vector<std::shared_ptr<OpIOInfo>> inputs_ptr_;
  std::vector<std::shared_ptr<OpIOInfo>> outputs_ptr_;
  std::unordered_map<size_t, size_t> ref_infos_;
};

using OpAttrPtr = std::shared_ptr<OpAttr>;
using OpIOInfoPtr = std::shared_ptr<OpIOInfo>;
using OpInfoPtr = std::shared_ptr<OpInfo>;
BACKEND_EXPORT std::vector<std::string> SplitStrToVec(const std::string &input);
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_OPLIB_OPINFO_H_
