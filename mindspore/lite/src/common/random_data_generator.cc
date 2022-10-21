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

#include "src/common/random_data_generator.h"
#include "include/errorcode.h"
#include "mindapi/base/type_id.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
constexpr float kInputDataFloatMin = 0.1f;
constexpr float kInputDataFloatMax = 1.0f;
constexpr double kInputDataDoubleMin = 0.1;
constexpr double kInputDataDoubleMax = 1.0;
constexpr int64_t kInputDataInt64Min = 0;
constexpr int64_t kInputDataInt64Max = 1;
constexpr int32_t kInputDataInt32Min = 0;
constexpr int32_t kInputDataInt32Max = 1;
constexpr int16_t kInputDataInt16Min = 0;
constexpr int16_t kInputDataInt16Max = 1;
constexpr int16_t kInputDataInt8Min = -127;
constexpr int16_t kInputDataInt8Max = 127;
constexpr int16_t kInputDataUint8Min = 0;
constexpr int16_t kInputDataUint8Max = 254;
}  // namespace
int GenRandomData(size_t size, void *data, int data_type) {
  MS_ASSERT(data != nullptr);
  switch (data_type) {
    case kNumberTypeFloat32:
    case kNumberTypeFloat:
      FillRandomData<float>(size, data, std::uniform_real_distribution<float>(kInputDataFloatMin, kInputDataFloatMax));
      break;
    case kNumberTypeFloat64:
      FillRandomData<double>(size, data,
                             std::uniform_real_distribution<double>(kInputDataDoubleMin, kInputDataDoubleMax));
      break;
    case kNumberTypeInt64:
      FillRandomData<int64_t>(size, data,
                              std::uniform_int_distribution<int64_t>(kInputDataInt64Min, kInputDataInt64Max));
      break;
    case kNumberTypeInt:
    case kNumberTypeInt32:
      FillRandomData<int32_t>(size, data,
                              std::uniform_int_distribution<int32_t>(kInputDataInt32Min, kInputDataInt32Max));
      break;
    case kNumberTypeInt16:
      FillRandomData<int16_t>(size, data,
                              std::uniform_int_distribution<int16_t>(kInputDataInt16Min, kInputDataInt16Max));
      break;
    case kNumberTypeInt8:
      FillRandomData<int8_t>(size, data, std::uniform_int_distribution<int16_t>(kInputDataInt8Min, kInputDataInt8Max));
      break;
    case kNumberTypeUInt8:
      FillRandomData<uint8_t>(size, data,
                              std::uniform_int_distribution<uint16_t>(kInputDataUint8Min, kInputDataUint8Max));
      break;
    default:
      char *casted_data = static_cast<char *>(data);
      for (size_t i = 0; i < size; i++) {
        casted_data[i] = static_cast<char>(i);
      }
  }
  return RET_OK;
}

int GenRandomData(mindspore::MSTensor *tensor) {
  CHECK_NULL_RETURN(tensor);
  auto input_data = tensor->MutableData();
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "MallocData for inTensor failed";
    return RET_ERROR;
  }
  int status = RET_ERROR;
  if (static_cast<TypeId>(tensor->DataType()) == kObjectTypeString) {
    MSTensor *input = MSTensor::StringsToTensor(tensor->Name(), {"you're the best."});
    if (input == nullptr) {
      std::cerr << "StringsToTensor failed" << std::endl;
      MS_LOG(ERROR) << "StringsToTensor failed";
      return RET_ERROR;
    }
    *tensor = *input;
    delete input;
  } else {
    status = GenRandomData(tensor->DataSize(), input_data, static_cast<int>(tensor->DataType()));
  }
  if (status != RET_OK) {
    std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
    MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
    return status;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
