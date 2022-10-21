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

#include "src/common/config_infos.h"
#include <vector>
#include "src/common/log_adapter.h"
#include "src/common/common.h"

namespace mindspore {
std::vector<std::string> ProfileParser::Split(const std::string &str, const std::string &delim) {
  auto start = 0U;
  auto end = str.find(delim);
  std::vector<std::string> substrs;
  while (end != std::string::npos) {
    substrs.push_back(str.substr(start, end - start));
    start = end + delim.length();
    end = str.find(delim, start);
  }
  substrs.push_back(str.substr(start, end));
  return substrs;
}

bool ProfileParser::StrToInt64(const std::string &str, int64_t *val) {
  if (str.empty()) {
    return false;
  }
  int64_t symbol = 1;
  size_t index = 0;
  if (str[0] == '-') {
    symbol = -1;
    index++;
  }
  int64_t val_ret = 0;
  constexpr int64_t decimal_times = 10;
  for (; index < str.size(); index++) {
    auto c = str[index];
    if (c < '0' || c > '9') {
      return false;
    }
    val_ret = val_ret * decimal_times + (c - '0');
    if (val_ret >= INT32_MAX) {
      return false;
    }
  }
  *val = symbol * val_ret;
  return true;
}

constexpr int KV_NUM = 2;

bool ProfileParser::ParseShapeStr(const std::vector<std::string> &str_dims, ShapeVector *shape_ptr) {
  if (shape_ptr == nullptr) {
    return false;
  }
  if (str_dims.empty()) {
    MS_LOG_ERROR << "Invalid input shape dim, dims number cannot be 0";
    return false;
  }
  auto &shape = *shape_ptr;
  shape.resize(str_dims.size());
  for (size_t i = 0; i != str_dims.size(); ++i) {
    int64_t dim = 0;
    if (!StrToInt64(str_dims[i], &dim)) {
      MS_LOG_ERROR << "Invalid input shape dim, dim value range or format is invalid: " << str_dims[i];
      return false;
    }
    if (dim <= 0 && dim != -1) {
      MS_LOG_ERROR << "Invalid input shape dim, dim can only be -1 when dim < 0： " << str_dims[i];
      return false;
    }
    shape[i] = dim;
  }
  return true;
}

bool ProfileParser::ParseRangeStr(const std::string &range_str, int64_t *min_ptr, int64_t *max_ptr) {
  if (min_ptr == nullptr || max_ptr == nullptr) {
    return false;
  }
  auto min_and_max = Split(range_str, "~");
  auto &min = *min_ptr;
  auto &max = *max_ptr;
  constexpr size_t number_range_count = 1;
  constexpr size_t min_max_range_count = 2;
  if (min_and_max.size() == number_range_count) {
    if (!StrToInt64(min_and_max[0], &min)) {
      MS_LOG_ERROR << "Invalid dynamic dim value range, dim value range or format is invalid: " << range_str
                   << ". It should be 'min~max' or a number.";
      return false;
    }
    max = min;
  } else if (min_and_max.size() == min_max_range_count) {
    if (!StrToInt64(min_and_max[0], &min) || !StrToInt64(min_and_max[1], &max)) {
      MS_LOG_ERROR << "Invalid dynamic dim value range, dim value range or format is invalid: " << range_str
                   << ". It should be 'min~max' or a number.";
      return false;
    }
  } else {
    MS_LOG_ERROR << "Invalid dynamic dim value range, dim value range or format is invalid: " << range_str
                 << ". It should be 'min~max' or a number.";
    return false;
  }
  if (min > max || min <= 0) {
    MS_LOG(ERROR) << "Invalid dimension range string format of '" << lite::kDynamicDims << "': " << range_str;
    return false;
  }
  return true;
}

bool ProfileParser::ParseOptDimStr(const std::string &opt_dim_str, int64_t *opt_ptr) {
  if (opt_ptr == nullptr) {
    return false;
  }
  auto &opt = *opt_ptr;
  opt = std::stoll(opt_dim_str);
  if (!StrToInt64(opt_dim_str, &opt)) {
    MS_LOG_ERROR << "Invalid opt dim value range, dim value range or format is invalid: " << opt_dim_str
                 << ". It should be a number.";
    return false;
  }
  if (opt <= 0) {
    MS_LOG(ERROR) << "Invalid opt dim value range '" << lite::kOptimizeDims << "': " << opt_dim_str;
    return false;
  }
  return true;
}

bool ProfileParser::ParseInputShape(const std::string &input_shapes_str, ProfileConfigs *profile_configs_ptr) {
  auto &profile_configs = *profile_configs_ptr;
  auto input_slices = Split(input_shapes_str, ";");
  for (auto &input_slice : input_slices) {
    auto split_pos = input_slice.rfind(':');
    if (split_pos == std::string::npos) {
      MS_LOG(ERROR) << "The input_shape should be in format of name:shape;name:shape, but got [" << input_shapes_str
                    << "]";
      return false;
    }
    std::string name = input_slice.substr(0, split_pos);
    std::string shape_str = input_slice.substr(split_pos + 1);
    if (shape_str.front() != '[' || shape_str.back() != ']') {
      MS_LOG(ERROR) << "shape format check fail.";
      return false;
    }
    constexpr size_t trim_len = 2;
    shape_str = shape_str.substr(1, shape_str.size() - trim_len);
    auto str_dims = Split(shape_str, ",");
    ProfileInputInfo info;
    info.name = name;
    ShapeVector &shape = info.input_shape;
    if (!ParseShapeStr(str_dims, &shape)) {
      MS_LOG(ERROR) << "Invalid input shape dims: " << shape_str;
      return false;
    }
    info.is_dynamic_shape = std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim < 0; });
    profile_configs.input_infos.push_back(info);
  }
  return true;
}

bool ProfileParser::ParseDynamicDims(const std::string &dynamic_dims_str, ProfileConfigs *profile_configs_ptr) {
  auto &profile_configs = *profile_configs_ptr;
  auto inputs_of_str = Split(dynamic_dims_str, ";");
  if (inputs_of_str.size() != profile_configs.input_infos.size()) {
    MS_LOG(ERROR) << "The input count " << inputs_of_str.size() << " in '" << lite::kDynamicDims
                  << "' != the input count " << profile_configs.input_infos.size() << " '"
                  << " in '" << lite::kInputShape;
    return false;
  }
  // for every input
  for (size_t input_index = 0; input_index != inputs_of_str.size(); ++input_index) {
    auto &info = profile_configs.input_infos[input_index];
    auto one_input_str = inputs_of_str[input_index];
    auto profiles_of_str = Split(one_input_str, "],[");
    if (profiles_of_str.empty()) {
      MS_LOG(ERROR) << "The profile count of " << input_index << "th input in '" << lite::kDynamicDims << "' is 0";
      return false;
    }
    if (profile_configs.profiles.empty()) {
      profile_configs.profiles.resize(profiles_of_str.size());
      for (auto &profile : profile_configs.profiles) {
        profile.inputs.resize(profile_configs.input_infos.size());
      }
    }
    if (profiles_of_str.size() != profile_configs.profiles.size()) {
      MS_LOG(ERROR) << "The profile count " << profiles_of_str.size() << " of " << input_index << "th input in '"
                    << lite::kDynamicDims << "' != profile count " << profile_configs.profiles.size() << " of "
                    << (input_index - 1) << " th input";
      return false;
    }
    // for every profile in one input, parse input range: min, max
    for (size_t profile_index = 0; profile_index != profiles_of_str.size(); ++profile_index) {
      ProfileItem &profile_item = profile_configs.profiles[profile_index];
      ProfileInputRange &input_range = profile_item.inputs[input_index];

      auto one_profile_str = profiles_of_str[profile_index];
      while (one_profile_str.front() == '[' || one_profile_str.front() == ' ') {
        one_profile_str = one_profile_str.substr(1);
      }
      while (one_profile_str.back() == ']' || one_profile_str.back() == ' ') {
        one_profile_str = one_profile_str.substr(0, one_profile_str.size() - 1);
      }
      auto dim_ranges = Split(one_profile_str, ",");

      auto &input_shape = info.input_shape;
      size_t dynamic_nbdims = std::count(input_shape.begin(), input_shape.end(), -1);
      if (dim_ranges.size() != dynamic_nbdims) {
        MS_LOG(ERROR) << "Number of dynamic dims in config '" << lite::kDynamicDims << "' " << dim_ranges.size()
                      << " != that in '" << lite::kInputShape << "' " << dynamic_nbdims << ".";
        return false;
      }
      size_t range_index = 0;
      input_range.min_dims = input_shape;
      input_range.max_dims = input_shape;
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] != -1) {
          continue;
        }
        if (!ParseRangeStr(dim_ranges[range_index++], &input_range.min_dims[i], &input_range.max_dims[i])) {
          return false;
        }
      }
      input_range.opt_dims = input_range.min_dims;  // default
    }
  }
  return true;
}

bool ProfileParser::ParseOptDims(const std::string &opt_dims_str, ProfileConfigs *profile_configs_ptr) {
  auto &profile_configs = *profile_configs_ptr;
  auto inputs_of_str = Split(opt_dims_str, ";");
  if (inputs_of_str.size() != profile_configs.input_infos.size()) {
    MS_LOG(ERROR) << "The input count " << inputs_of_str.size() << " in '" << lite::kOptimizeDims
                  << "' != the input count " << profile_configs.input_infos.size() << " '"
                  << " in '" << lite::kInputShape;
    return false;
  }
  // for every input
  for (size_t input_index = 0; input_index != inputs_of_str.size(); ++input_index) {
    auto &info = profile_configs.input_infos[input_index];
    auto one_input_str = inputs_of_str[input_index];
    auto profiles_of_str = Split(one_input_str, "],[");
    if (profiles_of_str.size() != profile_configs.profiles.size()) {
      MS_LOG(ERROR) << "The profile count " << profiles_of_str.size() << " of " << input_index << "th input in '"
                    << lite::kOptimizeDims << "' != profile count " << profile_configs.profiles.size() << " in '"
                    << lite::kDynamicDims << "'";
      return false;
    }
    // for every profile in one input, parse input range: min, max
    for (size_t profile_index = 0; profile_index != profiles_of_str.size(); ++profile_index) {
      ProfileItem &profile_item = profile_configs.profiles[profile_index];
      ProfileInputRange &input_range = profile_item.inputs[input_index];

      auto one_profile_str = profiles_of_str[profile_index];
      while (one_profile_str.front() == '[' || one_profile_str.front() == ' ') {
        one_profile_str = one_profile_str.substr(1);
      }
      while (one_profile_str.back() == ']' || one_profile_str.back() == ' ') {
        one_profile_str = one_profile_str.substr(0, one_profile_str.size() - 1);
      }
      auto opt_dims_vec = Split(one_profile_str, ",");

      auto &input_shape = info.input_shape;
      size_t dynamic_nbdims = std::count(input_shape.begin(), input_shape.end(), -1);
      if (opt_dims_vec.size() != dynamic_nbdims) {
        MS_LOG(ERROR) << "Number of dynamic dims in config '" << lite::kOptimizeDims << "' " << opt_dims_vec.size()
                      << " != that in '" << lite::kInputShape << "' " << dynamic_nbdims << ".";
        return false;
      }
      size_t dynamic_index = 0;
      input_range.opt_dims = input_shape;
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] != -1) {
          continue;
        }
        if (!ParseOptDimStr(opt_dims_vec[dynamic_index++], &input_range.opt_dims[i])) {
          return false;
        }
      }
    }
  }
  return true;
}

bool ProfileParser::Parse(const std::map<std::string, std::string> &context, bool require_opt_when_dym,
                          ProfileConfigs *profile_configs_ptr) {
  if (profile_configs_ptr == nullptr) {
    return false;
  }
  auto &profile_configs = *profile_configs_ptr;
  auto get_item = [&context](const std::string &key) -> std::string {
    auto it = context.find(key);
    return it != context.end() ? it->second : "";
  };
  auto input_shapes = get_item(lite::kInputShape);
  auto dynamic_dims = get_item(lite::kDynamicDims);
  auto opt_dims = get_item(lite::kOptimizeDims);
  if (input_shapes.empty() && dynamic_dims.empty() && opt_dims.empty()) {
    MS_LOG(INFO) << "Do not found config of input range('" << lite::kInputShape << "')";
    return true;
  }
  if (input_shapes.empty()) {
    MS_LOG(ERROR) << "Config of '" << lite::kInputShape << " cannot be empty when '" << lite::kInputShape << "' or '"
                  << lite::kOptimizeDims << "' is not empty";
    return false;
  }
  if (!ParseInputShape(input_shapes, &profile_configs)) {
    MS_LOG(ERROR) << "parse input shape failed.";
    return false;
  }
  if (dynamic_dims.empty()) {
    ProfileItem profile_item;
    for (size_t i = 0; i < profile_configs.input_infos.size(); i++) {
      if (profile_configs.input_infos[i].is_dynamic_shape) {
        MS_LOG(ERROR) << "Config of '" << lite::kDynamicDims << "' cannot be empty when " << lite::kInputShape
                      << " is dynamic";
        return false;
      }
      auto &input_shape = profile_configs.input_infos[i].input_shape;
      ProfileInputRange input_range;
      input_range.min_dims = input_shape;
      input_range.max_dims = input_shape;
      input_range.opt_dims = input_shape;
      profile_item.inputs.push_back(input_range);
    }
    profile_configs.profiles.push_back(profile_item);
    return true;
  }
  if (!ParseDynamicDims(dynamic_dims, &profile_configs)) {
    MS_LOG(ERROR) << "parse dynamic dims failed.";
    return false;
  }
  if (require_opt_when_dym) {
    if (!ParseOptDims(opt_dims, &profile_configs)) {
      MS_LOG(ERROR) << "parse optimization dims failed.";
      return false;
    }
  }
  return true;
}

bool ProfileParser::ReorderByInputNames(const std::vector<std::string> &input_names, ProfileConfigs *profile_configs) {
  if (input_names.size() != profile_configs->input_infos.size()) {
    MS_LOG_ERROR << "Input name size " << input_names.size() << " != profile config input size "
                 << profile_configs->input_infos.size();
    return false;
  }
  ProfileConfigs new_profile_configs = *profile_configs;
  auto &input_infos = profile_configs->input_infos;
  auto &profiles = profile_configs->profiles;

  for (size_t input_index = 0; input_index < input_names.size(); input_index++) {
    auto input_name = input_names[input_index];
    size_t i = 0;
    for (; i < input_infos.size(); i++) {
      if (input_infos[i].name == input_name) {
        new_profile_configs.input_infos[input_index] = input_infos[i];
        for (size_t profile_index = 0; profile_index < profiles.size(); profile_index++) {
          new_profile_configs.profiles[profile_index].inputs[input_index] = profiles[profile_index].inputs[i];
        }
        break;
      }
    }
    if (i >= input_infos.size()) {
      MS_LOG_ERROR << "Cannot find input " << input_name << " in profile '" << lite::kInputShape << "' config";
      return false;
    }
  }
  *profile_configs = new_profile_configs;
  return true;
}
}  // namespace mindspore
