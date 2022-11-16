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

#include "src/calib_data_generator.h"
#include <experimental/filesystem>
#include <vector>
#include <set>
#include <string>
#include <numeric>
#include <algorithm>
#include "ops/tuple_get_item.h"
#include "common/anf_util.h"
#include "common/string_util.h"
#include "common/file_util.h"
#include "src/mapper_config_parser.h"
#include "src/data_preprocessor.h"
#include "adapter/utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace dpico {
namespace {
constexpr size_t kMaxSize = 1024;
}  // namespace
std::vector<std::vector<int64_t>> CalibDataGenerator::GetInputShapes(const api::AnfNodePtrList &graph_inputs) {
  std::vector<std::vector<int64_t>> input_shapes;
  for (const auto &input : graph_inputs) {
    ShapeVector shape_vector;
    if (GetShapeVectorFromParameter(input, &shape_vector) != RET_OK) {
      MS_LOG(ERROR) << "get graph input shape failed. " << input->fullname_with_scope();
      return std::vector<std::vector<int64_t>>();
    }
    if (shape_vector.empty()) {
      MS_LOG(ERROR) << input->fullname_with_scope() << " input shape is empty.";
      return std::vector<std::vector<int64_t>>();
    }
    if (shape_vector.size() == kDims4) {  // transform nchw to nhwc
      shape_vector = std::vector<int64_t>{shape_vector[kNCHW_N], shape_vector[kNCHW_H], shape_vector[kNCHW_W],
                                          shape_vector[kNCHW_C]};
    }
    for (size_t i = 0; i < shape_vector.size(); i++) {
      auto dim = shape_vector.at(i);
      if (dim < 0) {
        if (i == 0) {
          dim = 1;
        } else {
          MS_LOG(ERROR) << input->fullname_with_scope() << "'s input shape[" << i << "] is " << dim
                        << ", which is unsupported by dpico.";
          return std::vector<std::vector<int64_t>>();
        }
      }
    }
    input_shapes.emplace_back(shape_vector);
  }
  return input_shapes;
}

std::vector<std::string> CalibDataGenerator::GetInDataFileList(const api::AnfNodePtrList &graph_inputs) {
  auto preprocessed_data_dir = DataPreprocessor::GetInstance()->GetPreprocessedDataDir();
  if (preprocessed_data_dir.empty()) {
    MS_LOG(ERROR) << "preprocessed_data_dir is empty.";
    return {};
  }
  auto batch_size = DataPreprocessor::GetInstance()->GetBatchSize();
  if (batch_size == 0) {
    MS_LOG(ERROR) << "input image list batch size is 0.";
    return {};
  }
  std::vector<std::string> in_data_files_list;
  for (size_t i = 0; i < batch_size; i++) {
    std::string in_data_files;  //  such as /abs_path/op_1/1/input.bin,/abs_path/op_2/1/input.bin,...,
    for (const auto &input : graph_inputs) {
      auto op_name = input->fullname_with_scope();
      auto folder_name = ReplaceSpecifiedChar(op_name, '/', '_');
      std::string in_data_file = preprocessed_data_dir + folder_name + "/" + std::to_string(i) + "/input.bin";
      if (AccessFile(in_data_file, F_OK) == 0) {
        in_data_files += in_data_file + ',';
      } else {
        MS_LOG(ERROR) << in_data_file << " is not existed.";
        return {};
      }
    }
    (void)in_data_files_list.emplace_back(in_data_files);
  }
  return in_data_files_list;
}

int CalibDataGenerator::DumpKernelsData(const std::string &current_path,
                                        const std::vector<std::string> &in_data_file_list,
                                        const std::vector<std::string> kernel_names,
                                        const std::vector<std::vector<int64_t>> &input_shapes) {
  std::string model_file = "model.ms";
  for (auto in_data_file : in_data_file_list) {
    auto ret = lite::converter::InnerPredict(model_file, in_data_file, kernel_names, current_path, input_shapes);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Execute InnerPredict failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
STATUS CalibDataGenerator::ParseAttrFromFilename(struct OpAttr *op_attr, const std::string &file_name, bool is_input) {
  if (op_attr == nullptr) {
    MS_LOG(ERROR) << "op attr is nullptr.";
    return RET_ERROR;
  }
  // file name such as: conv1_output_1_shape_1_352_352_1_Float32_NHWC.bin;
  std::string search_str = is_input ? "_input_" : "_output_";
  std::string shape_attr = "shape";
  std::string attrs_str = file_name.substr(file_name.find(search_str) + search_str.size(),
                                           file_name.find(".bin") - file_name.find(search_str) - search_str.size());
  std::string idx = attrs_str.substr(0, attrs_str.find('_'));
  if (!IsValidUnsignedNum(idx)) {
    MS_LOG(ERROR) << idx << " is not a valid unsigned num.";
    return RET_ERROR;
  }
  op_attr->input_output_idx = std::stoi(idx);
  attrs_str = attrs_str.substr(attrs_str.find('_') + 1);
  op_attr->format = attrs_str.substr(attrs_str.rfind('_') + 1);
  attrs_str = attrs_str.substr(0, attrs_str.rfind('_'));
  op_attr->data_type = attrs_str.substr(attrs_str.rfind('_') + 1);
  attrs_str = attrs_str.substr(0, attrs_str.rfind('_'));
  auto shape_str = attrs_str.substr(attrs_str.find(shape_attr) + shape_attr.size());
  if (!shape_str.empty()) {
    auto dims = SplitString(shape_str, '_');
    for (const auto &dim : dims) {
      if (IsValidUnsignedNum(dim)) {
        op_attr->shape.push_back(std::stoi(dim));
      }
    }
  }

  return RET_OK;
}

int CalibDataGenerator::TransBinsToTxt(const std::vector<DumpOpInfo> &dump_op_infos) {
  auto batch_size = DataPreprocessor::GetInstance()->GetBatchSize();
  if (batch_size == 0) {
    MS_LOG(ERROR) << "input image list batch size is 0.";
    return RET_ERROR;
  }
  auto output_path = MapperConfigParser::GetInstance()->GetOutputPath();
  auto dump_data_dir = output_path + "dump_data/";
  auto calib_data_dir = output_path + "calib_data/";
  if (CreateDir(&calib_data_dir) != RET_OK) {
    MS_LOG(ERROR) << "create dir failed. " << calib_data_dir;
    return RET_ERROR;
  }
  for (const auto &dump_op_info : dump_op_infos) {
    std::ofstream ofs;
    std::string dump_op_txt_path =
      calib_data_dir + ReplaceSpecifiedChar(dump_op_info.origin_op_name, '/', '_') + ".txt";
    ofs.open(dump_op_txt_path, std::ios::out);
    if (!ofs.is_open()) {
      MS_LOG(ERROR) << "file open failed. " << dump_op_txt_path;
      return RET_ERROR;
    }
    (void)ofs.precision(kNumPrecision);
    bool is_input = dump_op_info.input_index >= 0;
    auto del_special_character = ReplaceSpecifiedChar(dump_op_info.dump_op_name, '/', '.');
    auto pattern = is_input
                     ? del_special_character + "_input_" + std::to_string(dump_op_info.input_index) + "_shape_"
                     : del_special_character + "_output_" + std::to_string(dump_op_info.output_index) + "_shape_";
    for (size_t i = 0; i < batch_size; i++) {
      auto cur_dump_data_dir = dump_data_dir + std::to_string(i) + '/';
      if (AccessFile(cur_dump_data_dir, F_OK) == 0) {
        std::string file_name;
        for (const auto &entry : std::experimental::filesystem::directory_iterator(cur_dump_data_dir)) {
          std::string cur_name = entry.path().filename();
          auto pos = cur_name.find(pattern);
          if (pos == 0) {
            file_name = cur_name;
            break;
          }
        }
        if (file_name.empty()) {
          MS_LOG(ERROR) << "there is no corresponding bin file of " << pattern << " in " << cur_dump_data_dir;
          return RET_ERROR;
        }
        struct OpAttr op_attr;
        if (ParseAttrFromFilename(&op_attr, file_name, is_input) != RET_OK) {
          MS_LOG(ERROR) << "parse attr from file name failed.";
          return RET_ERROR;
        }
        if (op_attr.data_type == "Float32") {
          if (ReadBinToOfstream<float>(cur_dump_data_dir + file_name, op_attr, ofs) != RET_OK) {
            MS_LOG(ERROR) << "read bin to ofstream failed.";
            ofs.close();
            return RET_ERROR;
          }
        } else if (op_attr.data_type == "Int32") {
          if (ReadBinToOfstream<int32_t>(cur_dump_data_dir + file_name, op_attr, ofs) != RET_OK) {
            ofs.close();
            MS_LOG(ERROR) << "read bin to ofstream failed.";
            return RET_ERROR;
          }
        } else {
          MS_LOG(ERROR) << "unsupported data type.";
          ofs.close();
          return RET_ERROR;
        }
        ofs << std::endl;
      } else {
        MS_LOG(ERROR) << cur_dump_data_dir << " doesn't exist.";
        ofs.close();
        return RET_ERROR;
      }
    }
    ofs.close();  // end of write data to txt
    if (MapperConfigParser::GetInstance()->AddImageList(dump_op_info.origin_op_name, dump_op_txt_path) != RET_OK) {
      MS_LOG(ERROR) << "add image list for " << dump_op_info.origin_op_name << "failed";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int CalibDataGenerator::Run(const api::AnfNodePtrList &graph_inputs, const api::AnfNodePtrList &nodes) {
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "graph inputs shouldn't be empty.";
    return RET_ERROR;
  }
  auto image_lists = MapperConfigParser::GetInstance()->GetImageLists();
  std::vector<DumpOpInfo> dump_op_infos;
  std::set<api::AnfNodePtr> has_visited;
  for (const auto &node : nodes) {
    if (has_visited.find(node) != has_visited.end()) {
      continue;
    }
    (void)has_visited.insert(node);
    DumpOpInfo dump_op_info = {node->fullname_with_scope(), node->fullname_with_scope(), -1, -1};
    if (control_flow_inputs_.find(node) != control_flow_inputs_.end()) {
      dump_op_info.dump_op_name = control_flow_inputs_[node].first->fullname_with_scope();
      dump_op_info.input_index = control_flow_inputs_[node].second;
    } else {
      dump_op_info.output_index = 0;
      if (CheckPrimitiveType(node, api::MakeShared<ops::TupleGetItem>())) {
        auto tuple_get_item_cnode = node->cast<api::CNodePtr>();
        if (tuple_get_item_cnode == nullptr || tuple_get_item_cnode->inputs().size() < kInputIndex2) {
          MS_LOG(ERROR) << "tuple_get_item_node is invalid. " << node->fullname_with_scope();
          return RET_ERROR;
        }
        dump_op_info.dump_op_name = tuple_get_item_cnode->input(1)->fullname_with_scope();
        dump_op_info.output_index = static_cast<int32_t>(GetTupleGetItemOutIndex(tuple_get_item_cnode));
      }
    }
    if (image_lists.find(dump_op_info.dump_op_name) == image_lists.end()) {
      (void)dump_op_infos.emplace_back(dump_op_info);
    }
  }
  if (dump_op_infos.empty()) {
    MS_LOG(ERROR) << "dumped ops shouldn't be empty when network is segmented";
    return RET_ERROR;
  }

  auto in_data_file_list = GetInDataFileList(graph_inputs);
  if (in_data_file_list.empty()) {
    MS_LOG(ERROR) << "get in data file for benchmark failed.";
    return RET_ERROR;
  }

  auto input_shapes = GetInputShapes(graph_inputs);
  if (input_shapes.empty()) {
    MS_LOG(ERROR) << "get input shapes for benchmark failed.";
    return RET_ERROR;
  }

  std::vector<std::string> kernel_names;
  (void)std::transform(dump_op_infos.begin(), dump_op_infos.end(), std::back_inserter(kernel_names),
                       [](DumpOpInfo const &op_info) { return op_info.dump_op_name; });
  auto output_path = MapperConfigParser::GetInstance()->GetOutputPath();
  if (DumpKernelsData(output_path, in_data_file_list, kernel_names, input_shapes) != RET_OK) {
    MS_LOG(ERROR) << "dump kernels data failed.";
    return RET_ERROR;
  }

  if (TransBinsToTxt(dump_op_infos) != RET_OK) {
    MS_LOG(ERROR) << "transform dumped files to txt failed.";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
