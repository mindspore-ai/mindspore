/**
* Copyright 2022 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_ACL_PLUGIN_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_ACL_PLUGIN_H

#include <memory>
#include "minddata/dataset/core/device_tensor.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/image/dvpp/utils/resouce_info.h"
#include "utils/dlopen_macro.h"

class DvppCommon;
struct DvppDataInfo;

PLUGIN_METHOD(CreateDvppVideo, void *, void *, uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
              const std::string &);
PLUGIN_METHOD(InitDvppVideo, int, void *);
PLUGIN_METHOD(CloseDvppVideo, int, void *);
PLUGIN_METHOD(DvppVideoDumpFrame, int, void *);

PLUGIN_METHOD(InitResource, int, ResourceInfo &);
PLUGIN_METHOD(GetContext, void *, int);
PLUGIN_METHOD(Release, void);

PLUGIN_METHOD(CreateAclProcessWithResize, void *, uint32_t, uint32_t, uint32_t, uint32_t, void *, bool, void *,
              const std::shared_ptr<DvppCommon> &);
PLUGIN_METHOD(CreateAclProcessWithPara, void *, uint32_t, uint32_t, void *, bool, void *,
              const std::shared_ptr<DvppCommon> &);
PLUGIN_METHOD(CreateAclProcess, void *, void *, bool, void *, const std::shared_ptr<DvppCommon> &);
PLUGIN_METHOD(DestroyAclProcess, void, void *);
PLUGIN_METHOD(ReleaseAclProcess, int, void *);
PLUGIN_METHOD(InitAclProcess, int, void *);
PLUGIN_METHOD(GetContextFromAclProcess, void *, void *);
PLUGIN_METHOD(GetStreamFromAclProcess, void *, void *);
PLUGIN_METHOD(JPEG_DRC_WITH_DATA, int, void *, const RawData &);
PLUGIN_METHOD(JPEG_DR_WITH_DATA, int, void *, const RawData &);
PLUGIN_METHOD(JPEG_D_WITH_DATA, int, void *, const RawData &);
PLUGIN_METHOD(JPEG_R_WITH_DATA, int, void *, const DvppDataInfo &);
PLUGIN_METHOD(JPEG_C_WITH_DATA, int, void *, const DvppDataInfo &);
PLUGIN_METHOD(PNG_D_WITH_DATA, int, void *, const RawData &);
PLUGIN_METHOD(JPEG_DRC, int, void *);
PLUGIN_METHOD(JPEG_DR, int, void *);
PLUGIN_METHOD(JPEG_D, int, void *);
PLUGIN_METHOD(JPEG_R, int, void *, const std::string &);
PLUGIN_METHOD(JPEG_C, int, void *, const std::string &);
PLUGIN_METHOD(PNG_D, int, void *);
PLUGIN_METHOD(GetMemoryData, void *, void *);
PLUGIN_METHOD(GetCropedDeviceData, DvppDataInfo *, void *);
PLUGIN_METHOD(GetResizedDeviceData, DvppDataInfo *, void *);
PLUGIN_METHOD(GetDecodeDeviceData, DvppDataInfo *, void *);
PLUGIN_METHOD(H2D_Sink, int, void *, const std::shared_ptr<mindspore::dataset::Tensor> &,
              std::shared_ptr<mindspore::dataset::DeviceTensor> &);
PLUGIN_METHOD(D2H_Pop, int, void *, const std::shared_ptr<mindspore::dataset::DeviceTensor> &,
              std::shared_ptr<mindspore::dataset::Tensor> &);
PLUGIN_METHOD(DeviceMemoryRelease, int, void *);
PLUGIN_METHOD(SetResizeParas, int, void *, uint32_t, uint32_t);
PLUGIN_METHOD(SetCropParas, int, void *, uint32_t, uint32_t);

ORIGIN_METHOD(aclrtMallocHost, int, void **, size_t);
PLUGIN_METHOD(aclrtMemcpy, int, void *, size_t, const void *, size_t, int);
ORIGIN_METHOD(aclrtFreeHost, int, void *);

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_UTILS_ACL_PLUGIN_H
