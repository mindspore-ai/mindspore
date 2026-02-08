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
#include "tools/data_dump/utils.h"

#include <string>

#include "ir/tensor_new.h"
#include "tools/data_dump/npy_header.h"
#include "utils/distributed_meta.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {

bool CheckStoul(size_t *const output_digit, const std::string &input_str) {
  try {
    *output_digit = std::stoul(input_str);
  } catch (const std::out_of_range &oor) {
    MS_LOG(ERROR) << "Out of Range error: " << oor.what() << " when parse " << input_str;
    return false;
  } catch (const std::invalid_argument &ia) {
    MS_LOG(ERROR) << "Invalid argument: " << ia.what() << " when parse " << input_str;
    return false;
  }
  return true;
}

string ShapeToString(const ShapeVector &shape) {
  std::ostringstream sstr;
  sstr << "\"[";
  for (size_t i = 0; i < shape.size(); i++) {
    sstr << (i > 0 ? "," : "") << shape[i];
  }
  sstr << "]\"";
  return string{sstr.str()};
}

namespace datadump {

std::uint32_t GetRankID() {
  std::uint32_t rank_id = 0;
  if (mindspore::DistributedMeta::GetInstance()->initialized()) {
    rank_id = mindspore::DistributedMeta::GetInstance()->global_rank_id();
  }
  return rank_id;
}

bool StartsWith(const std::string &s, const std::string &prefix) {
  if (s.length() < prefix.length()) {
    return false;
  }
  return s.find(prefix) == 0 ? true : false;
}

bool EndsWith(const std::string &s, const std::string &sub) {
  if (s.length() < sub.length()) {
    return false;
  }
  return s.rfind(sub) == (s.length() - sub.length()) ? true : false;
}

void SaveTensor2NPY(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr) {
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  if (tensor_ptr->data_type_c() == TypeId::kNumberTypeBFloat16) {
    auto new_tensor = tensor::from_spec(TypeId::kNumberTypeFloat32, tensor_ptr->shape(), device::DeviceType::kCPU);
    auto input_addr = static_cast<bfloat16 *>(tensor_ptr->device_address()->GetMutablePtr());
    auto output_addr = static_cast<float *>(new_tensor->device_address()->GetMutablePtr());
    size_t size = SizeOf(tensor_ptr->shape());
    tensor::TransDataType<float, bfloat16>(input_addr, output_addr, size);
    tensor_ptr = new_tensor;
  }
  std::string npy_header = GenerateNpyHeader(tensor_ptr->shape(), tensor_ptr->data_type());
  if (!npy_header.empty()) {
    ChangeFileMode(file_name, S_IWUSR);
    std::fstream output{file_name, std::ios::out | std::ios::trunc | std::ios::binary};
    if (!output.is_open()) {
      MS_LOG(ERROR) << "For 'TensorDump' ops, open " << file_name << " file failed, the args of 'file' is invalid.";
      return;
    }
    output << npy_header;
    (void)output.write(reinterpret_cast<const char *>(tensor_ptr->data_c()), SizeToLong(tensor_ptr->Size()));
    if (output.bad()) {
      output.close();
      MS_LOG(ERROR) << "For 'TensorDump' ops, write mem to " << file_name << " failed.";
      return;
    }
    output.close();
    ChangeFileMode(file_name, S_IRUSR);
  } else {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the type of " << TypeIdToType(tensor_ptr->data_type())->ToString()
                  << " not support dump.";
  }
}
}  // namespace datadump

}  // namespace mindspore
