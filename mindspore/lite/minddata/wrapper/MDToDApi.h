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
#ifndef DATASET_MDTODAPI_H_
#define DATASET_MDTODAPI_H_

#include <stdint.h>
#include <sys/types.h>

class MDToDApi;

typedef struct MDToDBuff {
  void *Buff;
  size_t DataSize;
  size_t TensorSize[4];
  size_t MaxBuffSize;
} MDToDBuff_t;

typedef struct MDToDConf {
  const char *pFolderPath;
  const char *pSchemFile;
  const char *pStoragePath;
  MDToDBuff_t columnsToReadBuff;
  float MEAN[3];
  float STD[3];
  int ResizeSizeWH[2];
  int fixOrientation;
  int CropSizeWH[2];
  int64_t fileid;  // -1 All files, otherwise get a single specific file
} MDToDConf_t;

typedef struct MDToDResult {
  int64_t fileid;
  int32_t isForTrain;
  int32_t noOfFaces;
  int32_t orientation;
  MDToDBuff_t fileNameBuff;
  MDToDBuff_t labelBuff;
  MDToDBuff_t imageBuff;
  MDToDBuff_t embeddingBuff;
  MDToDBuff_t boundingBoxesBuff;
  MDToDBuff_t confidencesBuff;
  MDToDBuff_t landmarksBuff;
  MDToDBuff_t faceFileNamesBuff;
  MDToDBuff_t imageQualitiesBuff;
  MDToDBuff_t faceEmbeddingsBuff;
} MDToDResult_t;

typedef int (*MDToDApi_pathTest_t)(const char *path);
typedef int (*MDToDApi_testAlbum_t)();
typedef MDToDApi *(*MDToDApi_createPipeLine_t)(MDToDConf_t MDConf);
typedef int (*MDToDApi_GetNext_t)(MDToDApi *pMDToDApi, MDToDResult_t *results);
typedef int (*MDToDApi_UpdateEmbeding_t)(MDToDApi *pMDToDApi, const char *column, float *emmbeddings,
                                         size_t emmbeddingsSize);
typedef int (*MDToDApi_UpdateStringArray_t)(MDToDApi *pMDToDApi, const char *column, MDToDBuff_t MDbuff);
typedef int (*MDToDApi_UpdateFloatArray_t)(MDToDApi *pMDToDApi, const char *column, MDToDBuff_t MDbuff);
typedef int (*MDToDApi_UpdateIsForTrain_t)(MDToDApi *pMDToDApi, uint8_t isForTrain);
typedef int (*MDToDApi_UpdateNoOfFaces_t)(MDToDApi *pMDToDApi, int32_t noOfFaces);
typedef int (*MDToDApi_Stop_t)(MDToDApi *pMDToDApi);
typedef int (*MDToDApi_Destroy_t)(MDToDApi *pMDToDApi);

#endif
