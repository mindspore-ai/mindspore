/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "utils/tensorprint_utils.h"
#include <atomic>
#include <thread>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "ir/meta_tensor.h"
#include "device/convert_tensor_utils.h"
#include "./securec.h"
#ifndef NO_DLIB
#include "tdt/tsd_client.h"
#include "tdt/tdt_host_interface.h"
#include "tdt/data_common.h"
#endif

namespace mindspore {
const char kShapeSeperator[] = ",";
static std::map<std::string, TypeId> print_type_map = {
  {"int8_t", TypeId::kNumberTypeInt8},     {"uint8_t", TypeId::kNumberTypeUInt8},
  {"int16_t", TypeId::kNumberTypeInt16},   {"uint16_t", TypeId::kNumberTypeUInt16},
  {"int32_t", TypeId::kNumberTypeInt32},   {"uint32_t", TypeId::kNumberTypeUInt32},
  {"int64_t", TypeId::kNumberTypeInt64},   {"uint64_t", TypeId::kNumberTypeUInt64},
  {"float16", TypeId::kNumberTypeFloat16}, {"float", TypeId::kNumberTypeFloat32},
  {"double", TypeId::kNumberTypeFloat64}};

static std::map<std::string, size_t> type_size_map = {
  {"int8_t", sizeof(int8_t)},     {"uint8_t", sizeof(uint8_t)},   {"int16_t", sizeof(int16_t)},
  {"uint16_t", sizeof(uint16_t)}, {"int32_t", sizeof(int32_t)},   {"uint32_t", sizeof(uint32_t)},
  {"int64_t", sizeof(int64_t)},   {"uint64_t", sizeof(uint64_t)}, {"float16", sizeof(float) / 2},
  {"float", sizeof(float)},       {"double", sizeof(double)}};

bool ParseTensorShape(const std::string &input_shape_str, std::vector<int> *const tensor_shape, size_t *dims) {
  if (tensor_shape == nullptr) {
    return false;
  }
  std::string shape_str = input_shape_str;
  if (shape_str.size() <= 2) {
    return false;
  }
  (void)shape_str.erase(shape_str.begin());
  shape_str.pop_back();
  shape_str += kShapeSeperator;
  string::size_type pos_begin = 0;
  string::size_type pos_end = shape_str.find(kShapeSeperator);
  while (pos_end != std::string::npos) {
    string dim_str = shape_str.substr(pos_begin, pos_end - pos_begin);
    tensor_shape->emplace_back(std::stoi(dim_str));
    (*dims) = (*dims) * std::stoul(dim_str);
    pos_begin = pos_end + sizeof(kShapeSeperator) - 1;
    pos_end = shape_str.find(kShapeSeperator, pos_begin);
  }
  return true;
}

bool PrintTensorToString(const char *str_data_ptr, mindspore::tensor::Tensor *const print_tensor,
                         const size_t &memory_size) {
  auto *tensor_data_ptr = static_cast<uint8_t *>(print_tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(tensor_data_ptr);
  auto cp_ret =
    memcpy_s(tensor_data_ptr, static_cast<size_t>(print_tensor->data().nbytes()), str_data_ptr, memory_size);
  if (cp_ret != EOK) {
    MS_LOG(ERROR) << "Print op Failed to copy the memory to py::tensor " << cp_ret;
    return false;
  }
  return true;
}

#ifndef NO_DLIB
bool ConvertDataItem2Tensor(const std::vector<tdt::DataItem> &items) {
  //  Acquire Python GIL
  py::gil_scoped_acquire gil_acquire;
  std::ostringstream buf;
  bool ret_end_sequence = false;
  for (auto &item : items) {
    if (item.dataType_ == tdt::TDT_END_OF_SEQUENCE) {
      ret_end_sequence = true;
      break;
    }
    std::vector<int> tensor_shape;
    size_t totaldims = 1;
    if (!ParseTensorShape(item.tensorShape_, &tensor_shape, &totaldims)) {
      MS_LOG(ERROR) << "Tensor print can not parse tensor shape, receive info" << item.tensorShape_;
      continue;
    }
    std::shared_ptr<std::string> str_data_ptr = std::static_pointer_cast<std::string>(item.dataPtr_);
    MS_EXCEPTION_IF_NULL(str_data_ptr);

    if (item.tensorType_ == "string") {
      std::string data(reinterpret_cast<const char *>(str_data_ptr->c_str()), item.dataLen_);
      buf << data << std::endl;
    } else {
      auto type_iter = print_type_map.find(item.tensorType_);
      if (type_iter == print_type_map.end()) {
        MS_LOG(ERROR) << "type of tensor need to print is not soupport" << item.tensorType_;
        continue;
      }
      auto type_id = type_iter->second;
      mindspore::tensor::Tensor print_tensor(type_id, tensor_shape);
      auto memory_size = totaldims * type_size_map[item.tensorType_];
      if (PrintTensorToString(str_data_ptr->data(), &print_tensor, memory_size)) {
        buf << print_tensor.ToStringRepr() << std::endl;
      }
    }
  }
  std::cout << buf.str() << std::endl;
  return ret_end_sequence;
}

void TensorPrint::operator()() {
  while (true) {
    std::vector<tdt::DataItem> bundle;
    if (tdt::TdtHostPopData("_npu_log", bundle) != 0) {
      break;
    }
    if (ConvertDataItem2Tensor(bundle)) {
      break;
    }
  }
}
#endif
}  // namespace mindspore
