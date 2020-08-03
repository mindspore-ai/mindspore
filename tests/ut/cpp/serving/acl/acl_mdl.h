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

#ifndef ACL_STUB_INC_ACL_MDL
#define ACL_STUB_INC_ACL_MDL
#include "acl_base.h"

#define ACL_MAX_DIM_CNT 128
#define ACL_MAX_TENSOR_NAME_LEN 128
#define ACL_MAX_BATCH_NUM 128
#define ACL_MAX_HW_NUM 128
#define ACL_MAX_SHAPE_COUNT 128

typedef struct aclmdlDataset aclmdlDataset;
typedef struct aclmdlDesc aclmdlDesc;

typedef struct aclmdlIODims {
  char name[ACL_MAX_TENSOR_NAME_LEN];
  size_t dimCount;
  int64_t dims[ACL_MAX_DIM_CNT];
} aclmdlIODims;

aclmdlDesc *aclmdlCreateDesc();
aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc);
aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId);

size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc);
size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc);
size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index);
size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index);

aclmdlDataset *aclmdlCreateDataset();
aclError aclmdlDestroyDataset(const aclmdlDataset *dataSet);
aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataSet, aclDataBuffer *dataBuffer);
size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataSet);
aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataSet, size_t index);

aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId);
aclError aclmdlLoadFromMem(const void *model, size_t modelSize, uint32_t *modelId);
aclError aclmdlLoadFromFileWithMem(const char *modelPath, uint32_t *modelId, void *workPtr, size_t workSize,
                                   void *weightPtr, size_t weightSize);
aclError aclmdlLoadFromMemWithMem(const void *model, size_t modelSize, uint32_t *modelId, void *workPtr,
                                  size_t workSize, void *weightPtr, size_t weightSize);

aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output);
aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream);
aclError aclmdlUnload(uint32_t modelId);

aclError aclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize);
aclError aclmdlQuerySizeFromMem(const void *model, size_t modelSize, size_t *workSize, size_t *weightSize);

aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);
aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);
aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index);
aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index);

aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index);
aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index);

#endif