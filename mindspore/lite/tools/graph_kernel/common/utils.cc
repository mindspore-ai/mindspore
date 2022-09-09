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
}  // namespace mindspore::graphkernel
