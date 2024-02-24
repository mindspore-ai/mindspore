/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_TDT_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_TDT_SYMBOL_H_
#include <string>
#include "acl/acl_tdt.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace transform {

ORIGIN_METHOD(acltdtAddDataItem, aclError, acltdtDataset *, acltdtDataItem *)
ORIGIN_METHOD(acltdtCreateChannel, acltdtChannelHandle *, uint32_t, const char *)
ORIGIN_METHOD(acltdtCreateChannelWithCapacity, acltdtChannelHandle *, uint32_t, const char *, size_t)
ORIGIN_METHOD(acltdtCreateDataItem, acltdtDataItem *, acltdtTensorType, const int64_t *, size_t, aclDataType, void *,
              size_t)
ORIGIN_METHOD(acltdtCreateDataset, acltdtDataset *)
ORIGIN_METHOD(acltdtDestroyChannel, aclError, acltdtChannelHandle *)
ORIGIN_METHOD(acltdtDestroyDataItem, aclError, acltdtDataItem *)
ORIGIN_METHOD(acltdtDestroyDataset, aclError, acltdtDataset *)
ORIGIN_METHOD(acltdtGetDataAddrFromItem, void *, const acltdtDataItem *)
ORIGIN_METHOD(acltdtGetDataItem, acltdtDataItem *, const acltdtDataset *, size_t)
ORIGIN_METHOD(acltdtGetDatasetName, const char *, const acltdtDataset *)
ORIGIN_METHOD(acltdtGetDatasetSize, size_t, const acltdtDataset *)
ORIGIN_METHOD(acltdtGetDataSizeFromItem, size_t, const acltdtDataItem *)
ORIGIN_METHOD(acltdtGetDataTypeFromItem, aclDataType, const acltdtDataItem *)
ORIGIN_METHOD(acltdtGetDimNumFromItem, size_t, const acltdtDataItem *)
ORIGIN_METHOD(acltdtGetDimsFromItem, aclError, const acltdtDataItem *, int64_t *, size_t)
ORIGIN_METHOD(acltdtGetTensorTypeFromItem, acltdtTensorType, const acltdtDataItem *)
ORIGIN_METHOD(acltdtQueryChannelSize, aclError, const acltdtChannelHandle *, size_t *)
ORIGIN_METHOD(acltdtReceiveTensor, aclError, const acltdtChannelHandle *, acltdtDataset *, int32_t)
ORIGIN_METHOD(acltdtSendTensor, aclError, const acltdtChannelHandle *, const acltdtDataset *, int32_t)
ORIGIN_METHOD(acltdtStopChannel, aclError, acltdtChannelHandle *)

extern acltdtAddDataItemFunObj acltdtAddDataItem_;
extern acltdtCreateChannelFunObj acltdtCreateChannel_;
extern acltdtCreateChannelWithCapacityFunObj acltdtCreateChannelWithCapacity_;
extern acltdtCreateDataItemFunObj acltdtCreateDataItem_;
extern acltdtCreateDatasetFunObj acltdtCreateDataset_;
extern acltdtDestroyChannelFunObj acltdtDestroyChannel_;
extern acltdtDestroyDataItemFunObj acltdtDestroyDataItem_;
extern acltdtDestroyDatasetFunObj acltdtDestroyDataset_;
extern acltdtGetDataAddrFromItemFunObj acltdtGetDataAddrFromItem_;
extern acltdtGetDataItemFunObj acltdtGetDataItem_;
extern acltdtGetDatasetNameFunObj acltdtGetDatasetName_;
extern acltdtGetDatasetSizeFunObj acltdtGetDatasetSize_;
extern acltdtGetDataSizeFromItemFunObj acltdtGetDataSizeFromItem_;
extern acltdtGetDataTypeFromItemFunObj acltdtGetDataTypeFromItem_;
extern acltdtGetDimNumFromItemFunObj acltdtGetDimNumFromItem_;
extern acltdtGetDimsFromItemFunObj acltdtGetDimsFromItem_;
extern acltdtGetTensorTypeFromItemFunObj acltdtGetTensorTypeFromItem_;
extern acltdtQueryChannelSizeFunObj acltdtQueryChannelSize_;
extern acltdtReceiveTensorFunObj acltdtReceiveTensor_;
extern acltdtSendTensorFunObj acltdtSendTensor_;
extern acltdtStopChannelFunObj acltdtStopChannel_;

void LoadAcltdtApiSymbol(const std::string &ascend_path);
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_TDT_SYMBOL_H_
