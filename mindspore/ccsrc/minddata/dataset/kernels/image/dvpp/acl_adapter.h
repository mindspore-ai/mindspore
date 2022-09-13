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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ACL_ADAPTER_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ACL_ADAPTER_H

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "minddata/dataset/kernels/image/dvpp/utils/acl_plugin.h"
#include "minddata/dataset/kernels/image/dvpp/utils/AclLiteError.h"
#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"

namespace mindspore::dataset {
class AclAdapter {
 public:
  static AclAdapter &GetInstance();

  bool HasAclPlugin() const { return plugin_handle_ != nullptr; }

  void *CreateDvppVideo(void *context, uint8_t *data, uint32_t size, uint32_t width, uint32_t height, uint32_t type,
                        uint32_t out_format, const std::string &output) const;
  AclLiteError InitDvppVideo(void *dvpp_video) const;
  AclLiteError CloseDvppVideo(void *dvpp_video) const;
  AclLiteError DvppVideoDumpFrame(void *dvpp_video) const;
  APP_ERROR InitResource(ResourceInfo *resource_info) const;
  void *GetContext(int device_id) const;
  void Release() const;
  void *CreateAclProcessWithResize(uint32_t resize_width, uint32_t resize_height, uint32_t crop_width,
                                   uint32_t crop_height, void *context, bool is_crop, void *stream,
                                   const std::shared_ptr<DvppCommon> &dvpp_common) const;
  void *CreateAclProcessWithPara(uint32_t para_width, uint32_t para_height, void *context, bool is_crop, void *stream,
                                 const std::shared_ptr<DvppCommon> &dvpp_common) const;
  void *CreateAclProcess(void *context, bool is_crop, void *stream,
                         const std::shared_ptr<DvppCommon> &dvpp_common) const;
  void DestroyAclProcess(void *acl_process) const;
  APP_ERROR ReleaseAclProcess(void *acl_process) const;
  APP_ERROR InitAclProcess(void *acl_process) const;
  void *GetContextFromAclProcess(void *acl_process) const;
  void *GetStreamFromAclProcess(void *acl_process) const;
  APP_ERROR JPEG_DRC_WITH_DATA(void *acl_process, const RawData &data) const;
  APP_ERROR JPEG_DR_WITH_DATA(void *acl_process, const RawData &data) const;
  APP_ERROR JPEG_D_WITH_DATA(void *acl_process, const RawData &data) const;
  APP_ERROR JPEG_R_WITH_DATA(void *acl_process, const DvppDataInfo &data) const;
  APP_ERROR JPEG_C_WITH_DATA(void *acl_process, const DvppDataInfo &data) const;
  APP_ERROR PNG_D_WITH_DATA(void *acl_process, const RawData &data) const;
  APP_ERROR JPEG_DRC(void *acl_process) const;
  APP_ERROR JPEG_DR(void *acl_process) const;
  APP_ERROR JPEG_D(void *acl_process) const;
  APP_ERROR JPEG_R(void *acl_process, const std::string &last_step) const;
  APP_ERROR JPEG_C(void *acl_process, const std::string &last_step) const;
  APP_ERROR PNG_D(void *acl_process) const;
  void *GetMemoryData(void *acl_process) const;
  DvppDataInfo *GetCropedDeviceData(void *acl_process) const;
  DvppDataInfo *GetResizedDeviceData(void *acl_process) const;
  DvppDataInfo *GetDecodeDeviceData(void *acl_process) const;
  APP_ERROR H2D_Sink(void *acl_process, const std::shared_ptr<mindspore::dataset::Tensor> &input,
                     std::shared_ptr<mindspore::dataset::DeviceTensor> *device_input) const;
  APP_ERROR D2H_Pop(void *acl_process, const std::shared_ptr<mindspore::dataset::DeviceTensor> &device_output,
                    std::shared_ptr<mindspore::dataset::Tensor> *output) const;
  APP_ERROR DeviceMemoryRelease(void *acl_process) const;
  APP_ERROR SetResizeParas(void *acl_process, uint32_t width, uint32_t height) const;
  APP_ERROR SetCropParas(void *acl_process, uint32_t width, uint32_t height) const;
  int Memcpy(void *dst, size_t dest_max, const void *src, size_t count, int kind) const;
  int MallocHost(void **host_ptr, size_t size) const;
  int FreeHost(void *host_ptr) const;

 private:
  AclAdapter() = default;
  ~AclAdapter() { FinalizePlugin(); }
  void InitPlugin();
  void FinalizePlugin();

  void *plugin_handle_ = nullptr;

  CreateDvppVideoFunObj create_dvpp_video_fun_obj_;
  InitDvppVideoFunObj init_dvpp_video_fun_obj_;
  CloseDvppVideoFunObj close_dvpp_video_fun_obj_;
  DvppVideoDumpFrameFunObj dvpp_video_dump_frame_fun_obj_;
  InitResourceFunObj init_resource_fun_obj_;
  GetContextFunObj get_context_fun_obj_;
  ReleaseFunObj release_fun_obj_;
  CreateAclProcessWithResizeFunObj create_acl_process_with_resize_fun_obj_;
  CreateAclProcessWithParaFunObj create_acl_process_with_para_fun_obj_;
  CreateAclProcessFunObj create_acl_process_fun_obj_;
  DestroyAclProcessFunObj destroy_acl_process_fun_obj_;
  ReleaseAclProcessFunObj release_acl_process_fun_obj_;
  InitAclProcessFunObj init_acl_process_fun_obj_;
  GetContextFromAclProcessFunObj get_context_from_acl_process_fun_obj_;
  GetStreamFromAclProcessFunObj get_stream_from_acl_process_fun_obj_;
  JPEG_DRC_WITH_DATAFunObj jpeg_drc_with_data_fun_obj_;
  JPEG_DR_WITH_DATAFunObj jpeg_dr_with_data_fun_obj_;
  JPEG_D_WITH_DATAFunObj jpeg_d_with_data_fun_obj_;
  JPEG_R_WITH_DATAFunObj jpeg_r_with_data_fun_obj_;
  JPEG_C_WITH_DATAFunObj jpeg_c_with_data_fun_obj_;
  PNG_D_WITH_DATAFunObj png_d_with_data_fun_obj_;
  JPEG_DRCFunObj jpeg_drc_fun_obj_;
  JPEG_DRFunObj jpeg_dr_fun_obj_;
  JPEG_DFunObj jpeg_d_fun_obj_;
  JPEG_RFunObj jpeg_r_fun_obj_;
  JPEG_CFunObj jpeg_c_fun_obj_;
  PNG_DFunObj png_d_fun_obj_;
  GetMemoryDataFunObj get_memory_data_fun_obj_;
  GetCropedDeviceDataFunObj get_croped_device_data_fun_obj_;
  GetResizedDeviceDataFunObj get_resized_device_data_fun_obj_;
  GetDecodeDeviceDataFunObj get_decode_device_data_fun_obj_;
  H2D_SinkFunObj h_2_d_sink_fun_obj_;
  D2H_PopFunObj d_2_h_pop_fun_obj_;
  DeviceMemoryReleaseFunObj device_memory_release_fun_obj_;
  SetResizeParasFunObj set_resize_paras_fun_obj_;
  SetCropParasFunObj set_crop_paras_fun_obj_;
  aclrtMallocHostFunObj aclrt_malloc_host_fun_obj_;
  aclrtFreeHostFunObj aclrt_free_host_fun_obj_;
  aclrtMemcpyFunObj aclrt_memcpy_fun_obj_;
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_DVPP_ACL_ADAPTER_H
