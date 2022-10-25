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

#include "coder/generator/component/const_blocks/benchmark.h"

namespace mindspore::lite::micro {
const char benchmark_source[] = R"RAW(/**
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
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define kMaxThreadNum 4
#define kBindDefault 1

void usage() {
  printf(
    "-- mindspore benchmark params usage:\n"
    "args[0]: executable file\n"
    "args[1]: inputs binary file\n"
    "args[2]: model weight binary file\n"
    "args[3]: loop count for performance test\n"
    "args[4]: calibration file\n"
    "args[5]: runtime thread num, default is 1\n"
    "args[6]: runtime thread bind mode, 0: No bind, 1: Bind high cpu, 2: Bind mid cpu, default is 1\n"
    "args[7]: warm up loop count, default is 3\n"
    "args[8]: cosine distance threshold, default is 0.9999\n\n");
}

uint64_t GetTimeUs() {
  const int USEC = 1000000;
  const int MSEC = 1000;
  struct timespec ts = {0, 0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0;
  }
  uint64_t retval = (uint64_t)((ts.tv_sec * USEC) + (ts.tv_nsec / MSEC));
  return retval;
}

void PrintTensorHandle(MSTensorHandle tensor) {
  printf("name: %s, ", MSTensorGetName(tensor));
  MSDataType data_type = MSTensorGetDataType(tensor);
  printf("DataType: %d, ", data_type);
  size_t element_num = (size_t)(MSTensorGetElementNum(tensor));
  printf("Elements: %zu, ", element_num);
  printf("Shape: [");
  size_t shape_num = 0;
  const int64_t *dims = MSTensorGetShape(tensor, &shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    printf("%d ", (int)dims[i]);
  }
  printf("], Data: \n");
  void *data = MSTensorGetMutableData(tensor);
  element_num = element_num > 10 ? 10 : element_num;
  switch (data_type) {
    case kMSDataTypeNumberTypeFloat32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%.6f, ", ((float *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeFloat16:
    case kMSDataTypeNumberTypeInt16: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRId16, ((int16_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRId32, ((int32_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRIi8, ((int8_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeUInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%u", ((uint8_t *)data)[i]);
      }
      printf("\n");
    } break;
    default:
      printf("Unsupported data type to print");
      break;
  }
}

int main(int argc, const char **argv) {
  if (argc < 2) {
    printf("input command is invalid\n");
    usage();
    return kMSStatusLiteError;
  }
  printf("=======run benchmark======\n");

  MSContextHandle ms_context_handle = MSContextCreate();
  if (argc >= 6) {
    int thread_num = atoi(argv[5]);
    if (thread_num < 1 || thread_num > kMaxThreadNum) {
      printf("Thread number error! It should be greater than 0 and less than 5\n");
      return kMSStatusLiteParamInvalid;
    }
    MSContextSetThreadNum(ms_context_handle, thread_num);
  }
  printf("ThreadNum: %d.\n", MSContextGetThreadNum(ms_context_handle));

  int bind_mode = kBindDefault;
  if (argc >= 7) {
    bind_mode = atoi(argv[6]);
    if (bind_mode < 0 || bind_mode > 2) {
      printf("Thread bind mode error! 0: No bind, 1: Bind hign cpu, 2: Bind mid cpu.\n");
      return kMSStatusLiteParamInvalid;
    }
  }
  MSContextSetThreadAffinityMode(ms_context_handle, bind_mode);
  printf("BindMode: %d.\n", MSContextGetThreadAffinityMode(ms_context_handle));

  void *model_buffer = NULL;
  int model_size = 0;
  // read .bin file by ReadBinaryFile;
  if (argc >= 3) {
    model_buffer = ReadInputData(argv[2], &model_size);
    if (model_buffer == NULL) {
      printf("Read model file failed.");
      return kMSStatusLiteParamInvalid;
    }
  }
  MSModelHandle model_handle = MSModelCreate();
  int ret = MSModelBuild(model_handle, model_buffer, model_size, kMSModelTypeMindIR, ms_context_handle);
  MSContextDestroy(&ms_context_handle);
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
  if (inputs_handle.handle_list == NULL) {
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
    inputs_binbuf[i] = NULL;
  }

  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  if (!outputs_handle.handle_list) {
    printf("MSModelGetOutputs failed, ret: %d", ret);
    return ret;
  }

  int warm_up_loop_count = 3;
  if (argc >= 8) {
      warm_up_loop_count = atoi(argv[7]);
      if (warm_up_loop_count < 0) {
        printf("The warm up loop count error! Cannot be less than 0.\n");
        return kMSStatusLiteParamInvalid;
      }
  }
  printf("Running warm up loops...");
  for (int i = 0; i < warm_up_loop_count; ++i) {
    ret = MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      printf("MSModelPredict failed, ret: %d", ret);
      return ret;
    }
  }

  if (argc >= 4) {
    int loop_count = atoi(argv[3]);
    printf("\nloop count: %d\n", loop_count);
    uint64_t start_time = GetTimeUs();
    for (int i = 0; i < loop_count; ++i) {
      ret = MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
      if (ret != kMSStatusSuccess) {
        MSModelDestroy(&model_handle);
        printf("MSModelPredict failed, ret: %d", ret);
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
    PrintTensorHandle(output);
  }
  if (argc >= 5) {
    CalibTensor *calib_tensors;
    int calib_num = 0;
    ret = ReadCalibData(argv[4], &calib_tensors, &calib_num);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      return ret;
    }
    float cosine_distance_threshold = 0.9999;
    if (argc >= 9) {
      cosine_distance_threshold = atof(argv[8]);
    }
    ret = CompareOutputs(outputs_handle, &calib_tensors, calib_num, cosine_distance_threshold);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      return ret;
    }
    FreeCalibTensors(&calib_tensors, calib_num);
  }
  printf("========run success=======\n");
  MSModelDestroy(&model_handle);
  return kMSStatusSuccess;
}
)RAW";

const char benchmark_source_cortex[] = R"RAW(/**
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

#include "benchmark.h"
#include "calib_output.h"
#include "load_input.h"
#include "data.h"
#include "c_api/types_c.h"
#include "c_api/model_c.h"
#include "c_api/context_c.h"
#include "src/tensor.h"
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t g_WorkSpace[WORK_SPACE_SIZE];

// Print data in tensor
void PrintTensorHandle(MSTensorHandle tensor) {
  printf("name: %s, ", MSTensorGetName(tensor));
  MSDataType data_type = MSTensorGetDataType(tensor);
  printf("DataType: %d, ", data_type);
  size_t element_num = (size_t)(MSTensorGetElementNum(tensor));
  printf("Elements: %zu, ", element_num);
  printf("Shape: [");
  size_t shape_num = 0;
  const int64_t *dims = MSTensorGetShape(tensor, &shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    printf("%d ", (int)dims[i]);
  }
  printf("], Data: \n");
  void *data = MSTensorGetMutableData(tensor);
  element_num = element_num > 10 ? 10 : element_num;
  switch (data_type) {
    case kMSDataTypeNumberTypeFloat32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%.6f, ", ((float *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%ld", ((int32_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%c", ((int8_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeUInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%u", ((uint8_t *)data)[i]);
      }
      printf("\n");
    } break;
    default:
      printf("Unsupported data type to print");
      break;
  }
}

int benchmark() {
  int ret;
  printf("========run benchmark======\n");
  printf("========Model build========\n");
  MSModelHandle model_handle = MSModelCreate();
  if (model_handle == NULL) {
    printf("MSModelCreate failed.\n");
    return kMSStatusLiteNullptr;
  }
  size_t workspace_size = MSModelCalcWorkspaceSize(model_handle);
  if (workspace_size > WORK_SPACE_SIZE) {
    printf("This Model inference requires %ul bytes of memory.\n", workspace_size);
    return kMSStatusLiteError;
  }
  MSModelSetWorkspace(model_handle, g_WorkSpace, WORK_SPACE_SIZE);
  ret = MSModelBuild(model_handle, NULL, 0, kMSModelTypeMindIR, NULL);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuildFromFile failed, ret : %d.\n", ret);
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Load inputs=======\n");
  MSTensorHandleArray inputs_handle = MSModelGetInputs(model_handle);
  if (inputs_handle.handle_list == NULL) {
    printf("MSModelGetInputs failed.");
    MSModelDestroy(&model_handle);
    return kMSStatusLiteError;
  }
  ret = SetDataToMSTensor(&inputs_handle, &g_inputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }
  ret = LoadCalibInputs(&inputs_handle, &g_calib_inputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Set outputs data pointer=======\n");
  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  if (outputs_handle.handle_list == NULL) {
    printf("MSModelGetOutputs failed.");
    MSModelDestroy(&model_handle);
    return kMSStatusLiteError;
  }
  ret = SetDataToMSTensor(&outputs_handle, &g_outputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Infer start=======\n");
  ret = MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Compare outputs=======\n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }

  ret = CompareOutputs(&outputs_handle, &g_calib_outputs);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("========Calib success=======\n");
  MSModelDestroy(&model_handle);
  return kMSStatusSuccess;
}

)RAW";
}  // namespace mindspore::lite::micro
