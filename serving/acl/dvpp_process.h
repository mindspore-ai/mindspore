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

#ifndef INC_DVPP_PROCESS_ACL
#define INC_DVPP_PROCESS_ACL
#include <vector>
#include <string>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl/ops/acl_dvpp.h"
#include "include/inference.h"

namespace mindspore::inference {

struct DvppDecodePara {
  acldvppPixelFormat pixel_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
};

struct DvppResizePara {
  uint32_t output_width = 0;
  uint32_t output_height = 0;
};

enum DvppCropType {
  // crop left,top,right,bottom is given in config
  kDvppCropTypeOffset = 0,
  // crop left,top,right,bottom is calculated by image width/height and output crop width/height
  kDvppCropTypeCentre = 1,
};

struct DvppRoiArea {
  uint32_t left = 0;
  uint32_t top = 0;
  uint32_t right = 0;
  uint32_t bottom = 0;
};

struct DvppCropInfo {
  DvppCropType crop_type = kDvppCropTypeOffset;
  DvppRoiArea crop_area;     // when kDvppCropTypeOffset
  uint32_t crop_width = 0;   // when kDvppCropTypeCentre
  uint32_t crop_height = 0;  // when kDvppCropTypeCentre
};

struct DvppCropPara {
  DvppCropInfo crop_info;
  uint32_t output_width = 0;
  uint32_t output_height = 0;
};

struct DvppCropAndPastePara {
  DvppCropInfo crop_info;
  DvppRoiArea paste_area;
  uint32_t output_width = 0;
  uint32_t output_height = 0;
};

class DvppProcess {
 public:
  DvppProcess();
  ~DvppProcess();

  Status InitResource(aclrtStream stream);
  void Finalize();
  Status InitJpegDecodePara(const DvppDecodePara &decode_para);                  // jpeg decode + (resize | crop)
  Status InitResizePara(const DvppResizePara &resize_para);                      // jpeg decode + resize
  Status InitCropPara(const DvppCropPara &crop_para);                            // jpeg decode + crop
  Status InitCropAndPastePara(const DvppCropAndPastePara &crop_and_paste_para);  // jpeg decode + crop&paste

  Status InitWithJsonConfig(const std::string &json_config);

  // output device buffer will be destroy by DvppProcess itself.
  Status Process(const void *pic_buffer, size_t pic_buffer_size, void *&output_device_buffer, size_t &output_size);
  Status Process(const std::vector<const void *> &pic_buffer_list, const std::vector<size_t> &pic_buffer_size_list,
                 void *&output_device_buffer, size_t &output_size);

 private:
  uint32_t pic_width_ = 0;
  uint32_t pic_height_ = 0;

  DvppDecodePara decode_para_;
  DvppResizePara resize_para_;
  DvppCropPara crop_para_;
  DvppCropAndPastePara crop_and_paste_para_;
  // only one of the resize or crop flag can be true
  bool to_resize_flag_ = false;
  bool to_crop_flag_ = false;
  bool to_crop_and_paste_flag_ = false;

  void *input_pic_dev_buffer_ = nullptr;
  uint32_t input_pic_buffer_size_ = 0;

  uint32_t decode_output_buffer_size_ = 0;
  void *decode_output_buffer_dev_ = nullptr;
  acldvppPicDesc *decode_output_desc_ = nullptr;

  acldvppResizeConfig *resize_config_ = nullptr;
  acldvppRoiConfig *crop_area_ = nullptr;
  acldvppRoiConfig *paste_area_ = nullptr;

  acldvppPicDesc *vpc_output_desc_ = nullptr;
  void *vpc_output_buffer_dev_ = nullptr;  // vpc_output_buffer_size_ length
  uint32_t vpc_output_buffer_size_ = 0;

  void *batch_vpc_output_buffer_dev_ = nullptr;  // batch_size_ * vpc_output_buffer_size_ length
  uint32_t batch_size_ = 0;

  aclrtStream stream_ = nullptr;
  acldvppChannelDesc *dvpp_channel_desc_ = nullptr;

  uint32_t AlignmentHelper(uint32_t org_size, uint32_t alignment) const;
  uint32_t GetImageBufferSize(uint32_t stride_width, uint32_t stride_height, acldvppPixelFormat pixel_format) const;
  Status GetPicDescStride(uint32_t width, uint32_t height, uint32_t &stride_width, uint32_t &stride_height);
  Status GetPicDescStrideDecode(uint32_t width, uint32_t height, uint32_t &stride_width, uint32_t &stride_height);
  Status InputInputBuffer(const void *pic_buffer, size_t pic_buffer_size);
  Status InitDecodeOutputDesc(uint32_t image_width,
                              uint32_t image_height);  // decode_output_desc_, decode_output_buffer_dev_
  Status CheckRoiAreaWidthHeight(uint32_t width, uint32_t height);
  Status CheckAndAdjustRoiArea(DvppRoiArea &area);
  Status UpdateCropArea(uint32_t image_width, uint32_t image_height);
  Status CheckResizeImageInfo(uint32_t image_width, uint32_t image_height) const;
  void DestroyDecodeDesc();

  Status InitVpcOutputDesc(uint32_t output_width, uint32_t output_height,
                           acldvppPixelFormat pixel_format);  // vpc_output_desc_, vpc_output_buffer_dev_batch_
  Status InitRoiAreaConfig(acldvppRoiConfig *&roi_area, const DvppRoiArea &init_para);
  Status InitCommonCropPara(DvppCropInfo &crop_info, uint32_t out_width, uint32_t out_height);
  Status InitResizeOutputDesc();        // vpc_output_desc_, vpc_output_buffer_dev_, resize_config
  Status InitCropOutputDesc();          // vpc_output_desc_, vpc_output_buffer_dev_, crop_area_
  Status InitCropAndPasteOutputDesc();  // vpc_output_desc_, vpc_output_buffer_dev_, crop_area_, paste_area_
  void DestroyVpcOutputDesc();

  Status ProcessDecode();
  Status ProcessResize();
  Status ProcessCrop();
  Status ProcessCropAndPaste();
  void DestroyResource();

  Status GetJpegWidthHeight(const void *pic_buffer, size_t pic_buffer_size, uint32_t &image_width,
                            uint32_t &image_height);
};

}  // namespace mindspore::inference

#endif  // INC_DVPP_PROCESS_ACL
