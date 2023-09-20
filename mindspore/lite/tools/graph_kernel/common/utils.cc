/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>

#include "tools/graph_kernel/common/utils.h"
#include "src/tensor.h"

namespace mindspore::graphkernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
std::vector<std::string> SplitString(const std::string &raw_str, char delimiter) {
  std::vector<std::string> res;
  std::string::size_type last_pos = 0;
  auto cur_pos = raw_str.find(delimiter);
  while (cur_pos != std::string::npos) {
    (void)res.emplace_back(raw_str.substr(last_pos, cur_pos - last_pos));
    cur_pos++;
    last_pos = cur_pos;
    cur_pos = raw_str.find(delimiter, cur_pos);
  }
  if (last_pos < raw_str.size()) {
    (void)res.emplace_back(raw_str.substr(last_pos, raw_str.size() - last_pos + 1));
  }
  return res;
}

int GetCustomShape(const std::string &attr, std::vector<std::vector<int>> *shapes) {
  auto split_shape_str = SplitString(attr, ',');
  for (size_t i = 0; i < split_shape_str.size(); i++) {
    size_t dim = std::stoul(split_shape_str[i]);
    if (i + dim >= split_shape_str.size()) {
      MS_LOG(ERROR) << "Shape string is invalid. The shape dim is " << dim << ", but only "
                    << split_shape_str.size() - i << " values follow.";
      return RET_ERROR;
    }
    std::vector<int> shape;
    for (size_t j = i + 1; j <= i + dim; j++) {
      shape.push_back(std::stoi(split_shape_str[j]));
    }
    i += dim;
    shapes->push_back(shape);
  }
  return RET_OK;
}

void GetCustomIndex(const std::string &dynamic_input_index, std::vector<size_t> *index) {
  auto split_index_str = SplitString(dynamic_input_index, ',');
  for (size_t i = 0; i < split_index_str.size(); i++) {
    index->push_back(std::stoul(split_index_str[i]));
  }
}

int CalculateDynamicBatchSize(const TensorC *const *inputs, size_t inputs_size,
                              const std::vector<std::vector<int>> &shapes, const std::vector<size_t> &index,
                              int *batch) {
  if (shapes.size() != inputs_size) {
    MS_LOG(ERROR) << "The saved inputs is not equal to the inputs_size: " << shapes.size() << " vs " << inputs_size;
    return RET_ERROR;
  }
  bool changed = false;
  for (auto i : index) {
    if (i >= shapes.size()) {
      MS_LOG(ERROR) << "The input num is " << shapes.size() << ", but want query index " << i;
      return RET_ERROR;
    }
    if (shapes[i].size() > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "The input shape size " << shapes[i].size() << " is greater than max size " << MAX_SHAPE_SIZE;
      return RET_ERROR;
    }
    for (size_t j = 0; j < shapes[i].size(); j++) {
      if (j == 0) {
        int bs = inputs[i]->shape_[0] / shapes[i][0];
        if (bs < 0) {
          MS_LOG(ERROR) << "AKG doesn't support batch size smaller than 1";
          return RET_ERROR;
        }
        if (bs != (*batch)) {
          if (!changed) {
            *batch = bs;
            changed = true;
          } else {
            MS_LOG(ERROR) << "AKG doesn't support inputs with different batch size";
            return RET_ERROR;
          }
        }
      } else if (inputs[i]->shape_[j] != shapes[i][j]) {
        MS_LOG(ERROR) << "AKG only support dynamic shape on axis 0";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

void SetAnfKernelInfoFormatFromAToB(const AnfNodePtr &node_a, const CNodePtr &node_b,
                                    const std::vector<std::string> &formats) {
  std::shared_ptr<device::KernelInfo> kernel_info = nullptr;
  auto kernel_info_builder = kernel::KernelBuildInfo::KernelBuildInfoBuilder();
  kernel_info_builder.SetOutputsFormat(formats);
  if (node_a->kernel_info_ptr() != nullptr) {
    kernel_info = std::make_shared<device::KernelInfo>();
  } else {
    kernel_info = std::dynamic_pointer_cast<device::KernelInfo>(node_a->kernel_info_ptr());
  }
  kernel_info->set_select_kernel_build_info(kernel_info_builder.Build());
  node_b->set_kernel_info(kernel_info);
}

void SetKernelInfoWithFormatToAnfNode(const AnfNodePtr &node, const std::vector<std::string> &format) {
  auto kernel_info_builder = kernel::KernelBuildInfo::KernelBuildInfoBuilder();
  kernel_info_builder.SetOutputsFormat(format);
  auto kernel_build_info = kernel_info_builder.Build();
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->set_select_kernel_build_info(kernel_build_info);
  node->set_kernel_info(kernel_info);
}

kernel::KernelBuildInfoPtr GetKernelInfo(const AnfNodePtr &node) {
  if (!node->has_user_data("kernel_info")) {
    return nullptr;
  }
  auto kernel_info_ptr = node->kernel_info_ptr();
  if (kernel_info_ptr == nullptr) {
    return nullptr;
  }
  auto kernel_info = std::dynamic_pointer_cast<device::KernelInfo>(kernel_info_ptr);
  if (kernel_info == nullptr) {
    MS_LOG(ERROR) << "kernel info from " << node->fullname_with_scope() << " is nullptr.";
    return nullptr;
  }
  auto kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (kernel_build_info == nullptr) {
    MS_LOG(ERROR) << "kernel build info from " << node->fullname_with_scope() << " is nullptr.";
    return nullptr;
  }
  return kernel_build_info;
}

std::string GetOutputFormatFromAnfNode(const AnfNodePtr &node, size_t output_idx) {
  auto kernel_build_info = GetKernelInfo(node);
  if (kernel_build_info == nullptr) {
    MS_LOG(EXCEPTION) << "kernel build info from " << node->fullname_with_scope() << " is empty.";
  }
  auto vec_size = kernel_build_info->GetOutputNum();
  if (output_idx >= vec_size) {
    MS_LOG(EXCEPTION) << "Index " << output_idx << " is out of the range of node output vector, output size is "
                      << kernel_build_info->GetOutputNum() << ". node is " << node->fullname_with_scope();
  }
  return kernel_build_info->GetOutputFormat(output_idx);
}
}  // namespace mindspore::graphkernel
