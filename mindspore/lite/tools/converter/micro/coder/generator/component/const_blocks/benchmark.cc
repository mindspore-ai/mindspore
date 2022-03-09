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
const char benchmark_source[] = R"RAW(
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

#include "load_input.h"
#include "calib_output.h"
#include "c_api/types_c.h"
#include "c_api/model_c.h"
#include "c_api/context_c.h"
#include "src/tensor.h"
#include <time.h>
#include <iostream>

using namespace mindspore;

void usage() {
  printf(
    "-- mindspore benchmark params usage:\n"
    "args[0]: executable file\n"
    "args[1]: inputs binary file\n"
    "args[2]: model weight binary file\n"
    "args[3]: loop count for performance test\n"
    "args[4]: calibration file\n"
    "args[5]: runtime thread num\n"
    "args[6]: runtime thread bind mode\n\n");
}

uint64_t GetTimeUs() {
  const int USEC = 1000000;
  const int MSEC = 1000;
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  auto retval = (uint64_t)((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return retval;
}

template <typename T>
void PrintData(void *data, size_t data_number) {
  if (data == nullptr) {
    return;
  }
  auto casted_data = static_cast<T *>(data);
  for (size_t i = 0; i < 10 && i < data_number; i++) {
    printf("%s, ", std::to_string(casted_data[i]).c_str());
  }
  printf("\n");
}

void PrintTensor(MSTensorHandle tensor) {
  printf("name: %s, ", MSTensorGetName(tensor));
  MSDataType data_type = MSTensorGetDataType(tensor);
  printf("DataType: %d, ", data_type);
  int element_num = static_cast<int>(MSTensorGetElementNum(tensor));
  printf("Elements: %d, ", element_num);
  printf("Shape: [");
  size_t shape_num = 0;
  const int64_t *dims = MSTensorGetShape(tensor, &shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    printf("%d ", (int)dims[i]);
  }
  printf("], Data: \n");
  void *data = MSTensorGetMutableData(tensor);
  switch (data_type) {
    case kMSDataTypeNumberTypeFloat32: {
      PrintData<float>(data, element_num);
    } break;
    case kMSDataTypeNumberTypeFloat16: {
      PrintData<int16_t>(data, element_num);
    } break;
    case kMSDataTypeNumberTypeInt32: {
      PrintData<int32_t>(data, element_num);
    } break;
    case kMSDataTypeNumberTypeInt16: {
      PrintData<int16_t>(data, element_num);
    } break;
    case kMSDataTypeNumberTypeInt8: {
      PrintData<int8_t>(data, element_num);
    } break;
    case kMSDataTypeNumberTypeUInt8: {
      PrintData<uint8_t>(data, element_num);
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
    return -1;
  }
  printf("=======run benchmark======\n");

  MSContextHandle ms_context_handle = NULL;
  if (argc >= 7) {
    int thread_num = atoi(argv[5]);
    if (thread_num < 1) {
      printf("Thread number error! It should be greater than 0\n");
      return -1;
    }
    int bind_mode = atoi(argv[6]);
    if (bind_mode < 0 || bind_mode > 2) {
      printf("Thread bind mode error! 0: No bind, 1: Bind hign cpu, 2: Bind mid cpu.\n");
      return -1;
    }
    ms_context_handle = MSContextCreate();
    if(ms_context_handle) {
      MSContextSetThreadNum(ms_context_handle, thread_num);
      MSContextSetThreadAffinityMode(ms_context_handle, bind_mode);
    }
    printf("context: ThreadNum: %d, BindMode: %d\n", thread_num,bind_mode);
  }

  void *model_buffer = NULL;
  int model_size = 0;
  // read .bin file by ReadBinaryFile;
  if (argc >= 3) {
    model_buffer = ReadInputData(argv[2], &model_size);
  }
  MSModelHandle model_handle = MSModelCreate();
  int ret = MSModelBuild(model_handle, model_buffer, model_size, kMSModelTypeMindIR, ms_context_handle);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuildFromFile failed, ret: %d\n", ret);
    free(model_buffer);
    model_buffer = NULL;
    return ret;
  }
  if (model_buffer) {
    free(model_buffer);
    model_buffer = NULL;
  }
  // set model inputs tensor data
  MSTensorHandleArray inputs_handle = MSModelGetInputs(model_handle);
  if (inputs_handle.handle_list == nullptr) {
    printf("MSModelGetInputs failed, ret: %d", ret);
    return ret;
  }
  size_t inputs_num = inputs_handle.handle_num;
  void *inputs_binbuf[inputs_num];
  int inputs_size[inputs_num];
  for (size_t i = 0; i < inputs_num; ++i) {
    MSTensorHandle tensor = inputs_handle.handle_list[i];
    inputs_size[i] = (int)MSTensorGetDataSize(tensor);
  }
  ret = ReadInputsFile((char *)(argv[1]), inputs_binbuf, inputs_size, (int)inputs_num);
  if (ret != 0) {
    MSModelDestroy(&model_handle);
    return ret;
  }
  for (size_t i = 0; i < inputs_num; ++i) {
    void *input_data = MSTensorGetMutableData(inputs_handle.handle_list[i]);
    memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
    free(inputs_binbuf[i]);
    inputs_binbuf[i] = nullptr;
  }

  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  if (!outputs_handle.handle_list) {
    printf("MSModelGetOutputs failed, ret: %d", ret);
    return ret;
  }
  for (size_t i = 0; i < outputs_handle.handle_num; ++i) {
    MSTensorHandle tensor = outputs_handle.handle_list[i];
    inputs_size[i] = (int)MSTensorGetDataSize(tensor);
  }
  if (argc >= 4) {
    int loop_count = atoi(argv[3]);
    printf("\nloop count: %d\n", loop_count);
    uint64_t start_time = GetTimeUs();
    for (int i = 0; i < loop_count; ++i) {
      ret = MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
      if (ret != kMSStatusSuccess) {
        MSModelDestroy(&model_handle);
        printf("MSModelPredict failed, ret: %d", kMSStatusSuccess);
        return ret;
      }
    }
    uint64_t end_time = GetTimeUs();
    float total_time = (float)(end_time - start_time) / 1000.0f;
    printf("total time: %.5fms, per time: %.5fms\n", total_time, total_time / loop_count);
  }
  ret = MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }
  printf("========run success=======\n");
  printf("\noutputs: \n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensor(output);
  }
  if (argc >= 5) {
    auto *calibrator = new (std::nothrow) lite::Calibrator();
    if (calibrator == nullptr) {
      MSModelDestroy(&model_handle);
      return lite::RET_NULL_PTR;
    }
    ret = calibrator->ReadCalibData(argv[4]);
    if (ret != lite::RET_OK) {
      MSModelDestroy(&model_handle);
      delete calibrator;
      return lite::RET_ERROR;
    }
    ret = calibrator->CompareOutputs(outputs_handle);
    if (ret != lite::RET_OK) {
      MSModelDestroy(&model_handle);
      delete calibrator;
      return lite::RET_ERROR;
    }
    delete calibrator;
  }
  printf("========run success=======\n");
  MSModelDestroy(&model_handle);
  return lite::RET_OK;
}

)RAW";
}  // namespace mindspore::lite::micro
