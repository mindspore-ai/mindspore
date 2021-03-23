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

#include "coder/generator/component/const_blocks/benchmark.h"

namespace mindspore::lite::micro {

const char *benchmark_source = R"RAW(
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

#include <iostream>
#include <string>
#include <cstring>

#include "include/lite_session.h"
#include "include/ms_tensor.h"
#include "include/errorcode.h"

#include "load_input.h"

using namespace mindspore;

void usage() {
  printf(
    "-- mindspore benchmark params usage:\n"
    "args[0]: executable file\n"
    "args[1]: inputs binary file\n"
    "args[2]: model weight binary file\n"
    "args[3]: loop count for performance test\n"
    "args[4]: runtime thread num\n"
    "args[5]: runtime thread bind mode\n\n");
}

#define LOG_INFO(content, args...) \
  { printf("[INFO] %s|%d: " #content "\r\n", __func__, __LINE__, ##args); }

template <typename T>
void PrintData(void *data, size_t data_number) {
  if (data == nullptr) {
    return;
  }
  auto casted_data = static_cast<T *>(data);
  for (size_t i = 0; i < 10 && i < data_number; i++) {
    printf("%s,", std::to_string(casted_data[i]).c_str());
  }
  printf("\n");
}

void TensorToString(tensor::MSTensor *tensor) {
  printf("name: %s, ", tensor->tensor_name().c_str());
  printf(", DataType: %d", tensor->data_type());
  printf(", Size: %lu", tensor->Size());
  printf(", Shape: ");
  for (auto &dim : tensor->shape()) {
    printf("%d ", dim);
  }
  printf(", Data: \n");
  switch (tensor->data_type()) {
    case kNumberTypeFloat32: {
      PrintData<float>(tensor->MutableData(), tensor->ElementsNum());
    } break;
    case kNumberTypeFloat16: {
      PrintData<int16_t>(tensor->MutableData(), tensor->ElementsNum());
    } break;
    case kNumberTypeInt32: {
      PrintData<int32_t>(tensor->MutableData(), tensor->ElementsNum());
    } break;
    case kNumberTypeInt16: {
      PrintData<int16_t>(tensor->MutableData(), tensor->ElementsNum());
    } break;
    case kNumberTypeInt8: {
      PrintData<int8_t>(tensor->MutableData(), tensor->ElementsNum());
    } break;
    case kNumberTypeUInt8: {
      PrintData<uint8_t>(tensor->MutableData(), tensor->ElementsNum());
    } break;
    default:
      std::cout << "Unsupported data type to print" << std::endl;
      break;
  }
}

int main(int argc, const char **argv) {
  if (argc < 2) {
    printf("input command is invalid\n");
    usage();
    return lite::RET_ERROR;
  }
  printf("=======run benchmark======\n");

  const char *model_buffer = nullptr;
  int model_size = 0;
  // read .bin file by ReadBinaryFile;
  if (argc >= 3) {
    model_buffer = static_cast<const char *>(ReadInputData(argv[2], &model_size));
  }
  session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, model_size, nullptr);
  if (session == nullptr) {
    std::cerr << "create lite session failed" << std::endl;
    return lite::RET_ERROR;
  }

  // set model inputs tensor data
  Vector<tensor::MSTensor *> inputs = session->GetInputs();
  size_t inputs_num = inputs.size();
  void *inputs_binbuf[inputs_num];
  int inputs_size[inputs_num];
  for (size_t i = 0; i < inputs_num; ++i) {
    inputs_size[i] = inputs[i]->Size();
  }
  int ret = ReadInputsFile(const_cast<char *>(argv[1]), inputs_binbuf, inputs_size, inputs_num);
  if (ret != lite::RET_OK) {
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < inputs_num; ++i) {
    void *input_data = inputs[i]->MutableData();
    memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
  }

  ret = session->RunGraph();
  if (ret != lite::RET_OK) {
    return lite::RET_ERROR;
  }

  Vector<String> outputs_name = session->GetOutputTensorNames();
  for (const auto &name : outputs_name) {
    auto output = session->GetOutputByTensorName(name);
    TensorToString(output);
  }
  printf("========run success=======\n");
  delete session;
  for (size_t i = 0; i < inputs_num; ++i) {
    free(inputs_binbuf[i]);
  }
  return lite::RET_OK;
}
)RAW";

}  // namespace mindspore::lite::micro
