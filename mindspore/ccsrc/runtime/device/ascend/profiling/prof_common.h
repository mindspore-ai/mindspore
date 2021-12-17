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
#ifndef MSPROFILER_PROF_COMMON_H_
#define MSPROFILER_PROF_COMMON_H_

#include <stdint.h>

#define MSPROF_DATA_HEAD_MAGIC_NUM 0x5a5a
#define MSPROF_DATA_HASH_MIN_LEN_128 128
#define MSPROF_GE_MODELLOAD_DATA_BYTES 104
#define MSPROF_DATA_HASH_MIN_LEN_64 64

#define MSPROF_MIX_DATA_RESERVE_BYTES 7
#define MSPROF_MIX_DATA_STRING_LEN 120

enum MsprofMixDataType {
  MSPROF_MIX_DATA_HASH_ID = 0,
  MSPROF_MIX_DATA_STRING = 1,
};

enum MsprofDataTag {
  MSPROF_GE_DATA_TAG_MODEL_LOAD = 20,
  MSPROF_GE_DATA_TAG_FUSION = 21,
  MSPROF_GE_DATA_TAG_INFER = 22,
  MSPROF_GE_DATA_TAG_TASK = 23,
  MSPROF_GE_DATA_TAG_TENSOR = 24,
  MSPROF_GE_DATA_TAG_STEP = 25,
  MSPROF_GE_DATA_TAG_ID_MAP = 26,
};

struct MsprofMixData {
  uint8_t type;
  uint8_t rsv[MSPROF_MIX_DATA_RESERVE_BYTES];
  union {
    uint64_t hashId;
    char dataStr[MSPROF_MIX_DATA_STRING_LEN];
  } data;
};
using MixData = struct MsprofMixData;

struct MsprofGeProfModelLoadData {
  uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
  uint16_t dataTag = MSPROF_GE_DATA_TAG_MODEL_LOAD;
  uint32_t modelId;
  MixData modelName;
  uint64_t startTime;
  uint64_t endTime;
  uint8_t reserve[MSPROF_GE_MODELLOAD_DATA_BYTES];
};

#define MSPROF_FUSION_DATA_RESERVE_BYTES 14
#define MSPROF_GE_FUSION_OP_NUM 8
struct MsprofGeProfFusionData {
  uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
  uint16_t dataTag = MSPROF_GE_DATA_TAG_FUSION;
  uint32_t modelId;
  MixData fusionName;
  uint64_t inputMemSize;
  uint64_t outputMemSize;
  uint64_t weightMemSize;
  uint64_t workspaceMemSize;
  uint64_t totalMemSize;
  uint16_t fusionOpNum;
  uint64_t fusionOp[MSPROF_GE_FUSION_OP_NUM];
  uint8_t reserve[MSPROF_FUSION_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_INFER_DATA_RESERVE_BYTES 64
struct MsprofGeProfInferData {
  uint16_t magicNumber;
  uint16_t dataTag;
  uint32_t modelId;
  MixData modelName;
  uint32_t requestId;
  uint32_t threadId;
  uint64_t inputDataStartTime;
  uint64_t inputDataEndTime;
  uint64_t inferStartTime;
  uint64_t inferEndTime;
  uint64_t outputDataStartTime;
  uint64_t outputDataEndTime;
  uint8_t reserve[MSPROF_GE_INFER_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_TASK_DATA_RESERVE_BYTES 16
#define MSPROF_GE_OP_TYPE_LEN 56
enum MsprofGeTaskType { MSPROF_GE_TASK_TYPE_AI_CORE = 0, MSPROF_GE_TASK_TYPE_AI_CPU, MSPROF_GE_TASK_TYPE_AIV };
enum MsprofGeShapeType { MSPROF_GE_SHAPE_TYPE_STATIC = 0, MSPROF_GE_SHAPE_TYPE_DYNAMIC };

struct MsprofGeOpType {
  uint8_t type;
  uint8_t rsv[MSPROF_MIX_DATA_RESERVE_BYTES];
  union {
    uint64_t hashId;
    char dataStr[MSPROF_GE_OP_TYPE_LEN];
  } data;
};
using GeOpType = struct MsprofGeOpType;

struct MsprofGeProfTaskData {
  uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
  uint16_t dataTag = MSPROF_GE_DATA_TAG_TASK;
  uint32_t taskType;
  MixData opName;
  GeOpType opType;
  uint64_t curIterNum;
  uint64_t timeStamp;
  uint32_t shapeType;
  uint32_t blockDims;
  uint32_t modelId;
  uint32_t streamId;
  uint32_t taskId;
  uint32_t threadId;
  uint8_t reserve[MSPROF_GE_TASK_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_TENSOR_DATA_RESERVE_BYTES 4
#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8
#define MSPROF_GE_TENSOR_DATA_NUM 5
enum MsprofGeTensorType { MSPROF_GE_TENSOR_TYPE_INPUT = 0, MSPROF_GE_TENSOR_TYPE_OUTPUT };
struct MsprofGeTensorData {
  uint32_t tensorType;
  uint32_t format;
  uint32_t dataType;
  uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};
using GeTensorData = struct MsprofGeTensorData;

struct MsprofGeProfTensorData {
  uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
  uint16_t dataTag = MSPROF_GE_DATA_TAG_TENSOR;
  uint32_t modelId;
  uint64_t curIterNum;
  uint32_t streamId;
  uint32_t taskId;
  uint8_t tensorNum;
  GeTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
  uint8_t reserve[MSPROF_GE_TENSOR_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_STEP_DATA_RESERVE_BYTES 27
struct MsprofGeProfStepData {
  uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
  uint16_t dataTag = MSPROF_GE_DATA_TAG_STEP;
  uint32_t modelId;
  uint32_t streamId;
  uint32_t taskId;
  uint64_t timeStamp;
  uint64_t curIterNum;
  uint32_t threadId;
  uint8_t tag;
  uint8_t reserve[MSPROF_GE_STEP_DATA_RESERVE_BYTES];
};

#define MSRPOF_GE_ID_MAP_DATA_RESERVE_BYTES 6
struct MsprofGeProfIdMapData {
  uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
  uint16_t dataTag = MSPROF_GE_DATA_TAG_ID_MAP;
  uint32_t graphId;
  uint32_t modelId;
  uint32_t sessionId;
  uint64_t timeStamp;
  uint16_t mode;
  uint8_t reserve[MSRPOF_GE_ID_MAP_DATA_RESERVE_BYTES];
};
#endif
