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
#include "extendrt/delegate/ascend_ge/ge_dynamic_utils.h"
#include "common/utils.h"
#include "common/common.h"
#include "extendrt/delegate/ascend_ge/ge_utils.h"

namespace mindspore {
bool GeDynamicUtils::IsDynamicInputShapes(const std::vector<ShapeVector> &input_shapes) {
  return std::any_of(input_shapes.begin(), input_shapes.end(), [](const ShapeVector &shape) {
    return std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim < 0; });
  });
}

bool GeDynamicUtils::IsDynamicInputShapes(const std::vector<std::pair<std::string, ShapeVector>> &input_shapes) {
  return std::any_of(input_shapes.begin(), input_shapes.end(), [](const auto &item) {
    auto &shape = item.second;
    return std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim < 0; });
  });
}

std::vector<std::pair<std::string, ShapeVector>> GeDynamicUtils::GetGraphInputShapes(
  const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos, std::string *input_shape_ptr) {
  // get input shape from AscendDeviceInfo
  auto ascend_info = GeUtils::GetAscendDeviceInfo(context);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Cannot find AscendDeviceInfo in context";
    return {};
  }
  auto input_shape_str = ascend_info->GetInputShape();
  if (!input_shape_str.empty()) {
    MS_LOG(INFO) << "Find input shape " << input_shape_str
                 << " in AscendDeviceInfo, which may come from [ascend_context] or [acl_option_cfg_param]";
  }
  // get options from [ge_graph_options]
  auto section_it = config_infos.find(lite::kGeGraphOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("ge.inputShape");
    if (option_it != options.end()) {
      input_shape_str = option_it->second;
      MS_LOG(INFO) << "Find ge.inputShape " << input_shape_str << " in " << lite::kGeGraphOptionsSection;
    }
  }
  // get options from [aoe_tuning_options]
  section_it = config_infos.find(lite::kAoeTuningOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("input_shape");
    if (option_it != options.end()) {
      input_shape_str = option_it->second;
      MS_LOG(INFO) << "Find input_shape " << input_shape_str << " in " << lite::kAoeTuningOptionsSection;
    }
  }
  if (!input_shape_str.empty()) {
    auto input_shape_strs = lite::StrSplit(input_shape_str, ";");
    std::vector<std::pair<std::string, ShapeVector>> input_shapes;
    for (auto &shape_item_str : input_shape_strs) {
      auto split_pos = shape_item_str.rfind(":");
      if (split_pos == std::string::npos) {
        MS_LOG(ERROR) << "The input_shape should be in format of name:shape;name:shape, but got [" << shape_item_str
                      << "]";
        return {};
      }
      std::string name = shape_item_str.substr(0, split_pos);
      std::string shape_str = shape_item_str.substr(split_pos + 1);
      ShapeVector shape;
      if (!lite::ParseShapeStr(shape_str, &shape)) {
        MS_LOG(ERROR) << "Invalid input shape dims: " << shape_str << ", input_shape: " << input_shape_str;
        return {};
      }
      input_shapes.push_back(std::make_pair(name, shape));
    }
    if (input_shape_ptr != nullptr) {
      *input_shape_ptr = input_shape_str;
    }
    return input_shapes;
  }
  return {};
}

std::vector<int64_t> GeDynamicUtils::GetDynamicBatchSize(const std::shared_ptr<mindspore::Context> &context,
                                                         const ConfigInfos &config_infos) {
  // get input shape from AscendDeviceInfo
  auto ascend_info = GeUtils::GetAscendDeviceInfo(context);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Cannot find AscendDeviceInfo in context";
    return {};
  }
  auto dynamic_batch_size = ascend_info->GetDynamicBatchSize();
  // get options from [acl_option_cfg_param]
  auto section_it = config_infos.find(lite::kAclOptionParam);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("dynamic_batch_size");
    if (option_it != options.end()) {
      dynamic_batch_size = option_it->second;
      MS_LOG(INFO) << "Find dynamic_batch_size " << dynamic_batch_size << " in " << lite::kAclOptionParam;
    }
  }
  // get options from [aoe_tuning_options]
  section_it = config_infos.find(lite::kAoeTuningOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("dynamic_batch_size");
    if (option_it != options.end()) {
      dynamic_batch_size = option_it->second;
      MS_LOG(INFO) << "Find dynamic_batch_size " << dynamic_batch_size << " in " << lite::kAoeTuningOptionsSection;
    }
  }
  if (dynamic_batch_size.empty()) {
    MS_LOG(INFO) << "Not found dynamic_batch_size in AscendDeviceInfo or config file";
    return {};
  }
  // parse dynamic_batch_size
  std::vector<int64_t> dynamic_batch_size_nums;
  if (!lite::ParseShapeStr(dynamic_batch_size, &dynamic_batch_size_nums)) {
    MS_LOG(ERROR) << "Invalid dynamic_batch_size " << dynamic_batch_size;
    return {};
  }
  return {};
}

std::vector<std::vector<int64_t>> GeDynamicUtils::GetDynamicImageSize(
  const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos) {
  // get input shape from AscendDeviceInfo
  auto ascend_info = GeUtils::GetAscendDeviceInfo(context);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Cannot find AscendDeviceInfo in context";
    return {};
  }
  auto dynamic_image_size = ascend_info->GetDynamicImageSize();
  // get options from [acl_option_cfg_param]
  auto section_it = config_infos.find(lite::kAclOptionParam);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("dynamic_image_size");
    if (option_it != options.end()) {
      dynamic_image_size = option_it->second;
      MS_LOG(INFO) << "Find dynamic_image_size " << dynamic_image_size << " in " << lite::kAclOptionParam;
    }
  }
  // get options from [aoe_tuning_options]
  section_it = config_infos.find(lite::kAoeTuningOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("dynamic_image_size");
    if (option_it != options.end()) {
      dynamic_image_size = option_it->second;
      MS_LOG(INFO) << "Find dynamic_image_size " << dynamic_image_size << " in " << lite::kAoeTuningOptionsSection;
    }
  }
  if (dynamic_image_size.empty()) {
    MS_LOG(INFO) << "Not found dynamic_image_size in AscendDeviceInfo or config file";
    return {};
  }
  // parse dynamic_image_size
  auto dynamic_image_strs = lite::StrSplit(dynamic_image_size, ";");
  if (dynamic_image_strs.empty()) {
    MS_LOG(ERROR) << "Invalid dynamic_image_size " << dynamic_image_size;
    return {};
  }
  std::vector<std::vector<int64_t>> dynamic_image_size_nums;
  for (auto &item : dynamic_image_strs) {
    std::vector<int64_t> real_dims;
    if (!lite::ParseShapeStr(item, &real_dims)) {
      MS_LOG(ERROR) << "Invalid dynamic_image_size " << dynamic_image_size;
      return {};
    }
    constexpr size_t hw_dim_count = 2;
    if (real_dims.size() != hw_dim_count) {
      MS_LOG(ERROR) << "Invalid dynamic_image_size " << dynamic_image_size;
      return {};
    }
    dynamic_image_size_nums.push_back(real_dims);
  }
  return dynamic_image_size_nums;
}

std::vector<std::vector<int64_t>> GeDynamicUtils::GetDynamicDims(const std::shared_ptr<mindspore::Context> &,
                                                                 const ConfigInfos &config_infos) {
  std::string dynamic_dims;
  // get options from [acl_option_cfg_param]
  auto section_it = config_infos.find(lite::kAclOptionParam);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("dynamic_dims");
    if (option_it != options.end()) {
      dynamic_dims = option_it->second;
      MS_LOG(INFO) << "Find dynamic_dims " << dynamic_dims << " in " << lite::kAclOptionParam;
    }
  }
  // get options from [ge_graph_options]
  section_it = config_infos.find(lite::kGeGraphOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("ge.dynamicDims");
    if (option_it != options.end()) {
      dynamic_dims = option_it->second;
      MS_LOG(INFO) << "Find ge.dynamicDims " << dynamic_dims << " in " << lite::kGeGraphOptionsSection;
    }
  }
  // get options from [aoe_tuning_options]
  section_it = config_infos.find(lite::kAoeTuningOptionsSection);
  if (section_it != config_infos.end()) {
    auto &options = section_it->second;
    auto option_it = options.find("dynamic_dims");
    if (option_it != options.end()) {
      dynamic_dims = option_it->second;
      MS_LOG(INFO) << "Find dynamic_dims " << dynamic_dims << " in " << lite::kAoeTuningOptionsSection;
    }
  }
  if (dynamic_dims.empty()) {
    MS_LOG(INFO) << "Not found dynamic_dims in AscendDeviceInfo or config file";
    return {};
  }
  // parse dynamic_dims
  auto dynamic_dims_strs = lite::StrSplit(dynamic_dims, ";");
  if (dynamic_dims_strs.empty()) {
    MS_LOG(ERROR) << "Invalid dynamic_dims " << dynamic_dims;
    return {};
  }
  std::vector<std::vector<int64_t>> dynamic_dims_nums;
  for (auto &item : dynamic_dims_strs) {
    std::vector<int64_t> real_dims;
    if (!lite::ParseShapeStr(item, &real_dims)) {
      MS_LOG(ERROR) << "Invalid dynamic_dims " << dynamic_dims;
      return {};
    }
    if (!dynamic_dims_nums.empty() && dynamic_dims_nums[0].size() != real_dims.size()) {
      MS_LOG(ERROR) << "Invalid dynamic_dims " << dynamic_dims << ", dims count in all dynamic dims should be same";
      return {};
    }
    dynamic_dims_nums.push_back(real_dims);
  }
  return dynamic_dims_nums;
}

static bool CheckDynamicDims(const std::vector<int64_t> &dynamic_batch_size,
                             const std::vector<std::vector<int64_t>> &dynamic_image_size,
                             const std::vector<std::vector<int64_t>> &dynamic_dims,
                             const std::string &input_shape_str) {
  if (!dynamic_dims.empty()) {
    if (!dynamic_image_size.empty() || !dynamic_batch_size.empty()) {
      MS_LOG(ERROR) << "Option dynamic_dims, dynamic_image_size and dynamic_batch_size cannot exist simultaneously.";
      return false;
    }
  } else if (!dynamic_image_size.empty()) {
    if (!dynamic_batch_size.empty()) {
      MS_LOG(ERROR) << "Option dynamic_dims, dynamic_image_size and dynamic_batch_size cannot exist simultaneously.";
      return false;
    }
  } else if (dynamic_batch_size.empty()) {
    MS_LOG(ERROR) << "Cannot find dynamic_dims, dynamic_batch_size or dynamic_image_size in AscendDeviceInfo or "
                     "config file while there are dynamic dims in input shapes "
                  << input_shape_str;
    return false;
  }
  return true;
}

std::vector<std::pair<std::string, ShapeVector>> GeDynamicUtils::GetGraphOneRealShapes(
  const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos, std::string *input_shape_ptr) {
  std::string input_shape_str;
  auto input_shapes = GetGraphInputShapes(context, config_infos, &input_shape_str);
  if (input_shape_ptr != nullptr) {
    *input_shape_ptr = input_shape_str;
  }
  if (input_shapes.empty()) {
    MS_LOG(INFO) << "Not found input shape in AscendDeviceInfo or config file";
    return {};
  }
  if (!IsDynamicInputShapes(input_shapes)) {
    MS_LOG(INFO) << "The dims number in all of input shapes are more than 0, return input shape in AscendDeviceInfo or "
                    "config file";
    return input_shapes;
  }
  auto dynamic_batch_size = GetDynamicBatchSize(context, config_infos);
  auto dynamic_image_size = GetDynamicImageSize(context, config_infos);
  auto dynamic_dims = GetDynamicDims(context, config_infos);
  if (!CheckDynamicDims(dynamic_batch_size, dynamic_image_size, dynamic_dims, input_shape_str)) {
    return {};
  }
  if (!dynamic_dims.empty()) {
    size_t dyn_count = 0;
    for (auto &input_shape : input_shapes) {
      for (auto &dim : input_shape.second) {
        if (dim == -1) {
          if (dyn_count >= dynamic_dims[0].size()) {
            MS_LOG(ERROR) << "Invalid dynamic_dims " << dynamic_dims
                          << " while dynamic dims in input_shape is more than " << (dyn_count + 1)
                          << ", input shape: " << input_shape_str;
            return {};
          }
          dim = dynamic_dims[0][dyn_count];
          dyn_count++;
        }
      }
    }
  } else if (!dynamic_image_size.empty()) {
    for (auto &input_shape : input_shapes) {
      size_t dyn_count = 0;
      for (auto &dim : input_shape.second) {
        if (dim == -1) {
          if (dyn_count >= dynamic_image_size[0].size()) {
            MS_LOG(ERROR) << "Invalid dynamic_image_size " << dynamic_image_size
                          << " while dynamic dims in input_shape is more than " << (dyn_count + 1)
                          << ", input shape: " << input_shape_str;
            return {};
          }
          dim = dynamic_image_size[0][dyn_count];
          dyn_count++;
        }
      }
    }
  } else {  // dynamic_batch_size
    for (auto &input_shape : input_shapes) {
      size_t dyn_count = 0;
      for (auto &dim : input_shape.second) {
        if (dim == -1) {
          if (dyn_count >= 1) {
            MS_LOG(ERROR) << "Invalid dynamic_batch_size " << dynamic_batch_size
                          << " while dynamic dims in input_shape is more than " << (dyn_count + 1)
                          << ", input shape: " << input_shape_str;
            return {};
          }
          dim = dynamic_batch_size[0];
          dyn_count++;
        }
      }
    }
  }
  return input_shapes;
}
}  // namespace mindspore
