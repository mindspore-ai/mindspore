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

#include "serving/acl/dvpp_process.h"
#include <jpeglib.h>
#include <fstream>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "include/infer_log.h"

namespace mindspore {
namespace inference {

DvppProcess::DvppProcess() {}

DvppProcess::~DvppProcess() {}

static uint32_t ToEven(uint32_t num) { return (num + 1) / 2 * 2; }

static uint32_t ToOdd(uint32_t num) {
  if (num == 0) {
    return 1;
  }
  return (num + 1) / 2 * 2 - 1;
}

class DvppJsonConfigParser {
 public:
  DvppJsonConfigParser() = default;
  ~DvppJsonConfigParser() = default;

  Status InitWithJsonConfig(const std::string &json_config);
  DvppDecodePara GetDecodePara() const { return decode_para_; }
  DvppResizePara GetResizePara() const { return resize_para_; }
  DvppCropPara GetCropPara() const { return crop_para_; }
  DvppCropAndPastePara GetCropAndPastePara() const { return crop_and_paste_para_; }
  bool HasResizeConfig() const { return resize_flag_; }
  bool HasCropConfig() const { return crop_flag_; }
  bool HasCropAndPasteConfig() const { return crop_and_paste_flag_; }

 private:
  DvppDecodePara decode_para_;
  DvppResizePara resize_para_;
  DvppCropPara crop_para_;
  DvppCropAndPastePara crop_and_paste_para_;
  bool resize_flag_ = false;
  bool crop_flag_ = false;
  bool crop_and_paste_flag_ = false;

  Status GetStringValue(const nlohmann::json &json_item, const std::string &key, std::string &val);
  Status GetIntValue(const nlohmann::json &json_item, const std::string &key, uint32_t &val);
  Status ParseInputPara(const nlohmann::json &preprocess_item);
  Status ParseDecodePara(const nlohmann::json &preprocess_item);
  Status ParseResizePara(const nlohmann::json &json_item);
  Status ParseCropPara(const nlohmann::json &json_item);
  Status ParseCropAndPastePara(const nlohmann::json &json_item);
  Status InitWithJsonConfigImp(const std::string &json_config);
};

Status DvppProcess::InitResource(aclrtStream stream) {
  stream_ = stream;
  aclError acl_ret;
  dvpp_channel_desc_ = acldvppCreateChannelDesc();
  if (dvpp_channel_desc_ == nullptr) {
    MSI_LOG_ERROR << "acldvppCreateChannelDesc failed";
    return FAILED;
  }
  acl_ret = acldvppCreateChannel(dvpp_channel_desc_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acldvppCreateChannel failed, acl return " << acl_ret;
    return FAILED;
  }
  MSI_LOG_INFO << "End init dvpp process resource";
  return SUCCESS;
}

void DvppProcess::DestroyResource() {
  if (dvpp_channel_desc_ != nullptr) {
    auto acl_ret = acldvppDestroyChannel(dvpp_channel_desc_);
    if (acl_ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "acldvppDestroyChannel failed, acl return " << acl_ret;
    }
    acl_ret = acldvppDestroyChannelDesc(dvpp_channel_desc_);
    if (acl_ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "acldvppDestroyChannelDesc failed, acl return " << acl_ret;
    }
    dvpp_channel_desc_ = nullptr;
  }
}

void DvppProcess::Finalize() {
  DestroyDecodeDesc();
  DestroyVpcOutputDesc();
  DestroyResource();
  if (resize_config_ != nullptr) {
    acldvppDestroyResizeConfig(resize_config_);
    resize_config_ = nullptr;
  }
  if (crop_area_ != nullptr) {
    acldvppDestroyRoiConfig(crop_area_);
    crop_area_ = nullptr;
  }
  if (paste_area_ != nullptr) {
    acldvppDestroyRoiConfig(paste_area_);
    paste_area_ = nullptr;
  }
  if (input_pic_dev_buffer_ != nullptr) {
    acldvppFree(input_pic_dev_buffer_);
  }
  input_pic_buffer_size_ = 0;
  MSI_LOG_INFO << "End dvpp process finalize";
}

Status DvppProcess::InitJpegDecodePara(const DvppDecodePara &decode_para) {
  decode_para_ = decode_para;
  MSI_LOG_INFO << "Init decode para, pixel_format " << decode_para_.pixel_format;
  return SUCCESS;
}

Status DvppProcess::InitResizePara(const DvppResizePara &resize_para) {
  resize_para_ = resize_para;
  MSI_LOG_INFO << "Init resize para, "
               << "output_width " << resize_para_.output_width << ", output_height " << resize_para_.output_height;
  to_resize_flag_ = true;
  to_crop_flag_ = false;
  to_crop_and_paste_flag_ = false;
  Status ret = InitResizeOutputDesc();
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "InitResizeOutputDesc failed";
  }
  return ret;
}

Status DvppProcess::InitCommonCropPara(DvppCropInfo &crop_info, uint32_t output_width, uint32_t output_height) {
  if (crop_info.crop_type == kDvppCropTypeOffset) {
    if (CheckAndAdjustRoiArea(crop_info.crop_area) != SUCCESS) {
      MSI_LOG_ERROR << "Check and adjust crop area failed";
      return FAILED;
    }
    MSI_LOG_INFO << "Init common crop para, crop type offset "
                 << ", left " << crop_info.crop_area.left << ", right " << crop_info.crop_area.right << ", top "
                 << crop_info.crop_area.top << ", bottom " << crop_info.crop_area.bottom << ", output_width "
                 << output_width << ", output_height " << output_height;
  } else {
    crop_info.crop_width = ToEven(crop_info.crop_width);
    crop_info.crop_height = ToEven(crop_info.crop_height);
    if (CheckRoiAreaWidthHeight(crop_info.crop_width, crop_info.crop_height) != SUCCESS) {
      MSI_LOG_ERROR << "Check crop area width and height failed, actually width " << crop_info.crop_width << " height "
                    << crop_info.crop_height;
      return FAILED;
    }
    MSI_LOG_INFO << "Init common crop para, crop type centre "
                 << ", crop_width " << crop_info.crop_width << ", crop_height " << crop_info.crop_height
                 << ", output_width " << output_width << ", output_height " << output_height;
  }
  return SUCCESS;
}

Status DvppProcess::InitCropPara(const DvppCropPara &crop_para) {
  crop_para_ = crop_para;
  if (InitCommonCropPara(crop_para_.crop_info, crop_para_.output_width, crop_para_.output_height) != SUCCESS) {
    MSI_LOG_ERROR << "Init common crop para failed in InitCropPara";
    return FAILED;
  }
  to_crop_flag_ = true;
  to_resize_flag_ = false;
  to_crop_and_paste_flag_ = false;
  Status ret = InitCropOutputDesc();
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "InitCropOutputDesc failed";
  }
  return ret;
}

Status DvppProcess::InitCropAndPastePara(const DvppCropAndPastePara &crop_and_paste_para) {
  crop_and_paste_para_ = crop_and_paste_para;
  if (InitCommonCropPara(crop_and_paste_para_.crop_info, crop_and_paste_para_.output_width,
                         crop_and_paste_para_.output_height) != SUCCESS) {
    MSI_LOG_ERROR << "Init common crop para failed in InitCropAndPastePara";
    return FAILED;
  }
  auto &paste_area = crop_and_paste_para_.paste_area;
  if (CheckAndAdjustRoiArea(paste_area) != SUCCESS) {
    MSI_LOG_ERROR << "Check and adjust paste area failed";
    return FAILED;
  }
  MSI_LOG_INFO << "Init crop and paste para, paste info: "
               << ", left " << paste_area.left << ", right " << paste_area.right << ", top " << paste_area.top
               << ", bottom " << paste_area.bottom;

  to_crop_and_paste_flag_ = true;
  to_crop_flag_ = false;
  to_resize_flag_ = false;
  Status ret = InitCropAndPasteOutputDesc();
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "InitCropAndPasteOutputDesc failed";
  }
  return ret;
}

Status DvppProcess::InputInputBuffer(const void *pic_buffer, size_t pic_buffer_size) {
  aclError acl_ret;
  if (pic_buffer_size != input_pic_buffer_size_) {
    acldvppFree(input_pic_dev_buffer_);
    input_pic_buffer_size_ = 0;
    acl_ret = acldvppMalloc(&input_pic_dev_buffer_, pic_buffer_size);
    if (acl_ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "acldvppMalloc input picture buffer on device failed, buffer size " << pic_buffer_size;
      return FAILED;
    }
    input_pic_buffer_size_ = pic_buffer_size;
  }
  acl_ret =
    aclrtMemcpy(input_pic_dev_buffer_, input_pic_buffer_size_, pic_buffer, pic_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "aclrtMemcpy input picture buffer to device, buffer size " << pic_buffer_size;
    return FAILED;
  }
  return SUCCESS;
}

static void JpegErrorExitCustom(j_common_ptr cinfo) {
  char jpeg_last_error_msg[JMSG_LENGTH_MAX];
  if (cinfo != nullptr && cinfo->err != nullptr && cinfo->err->format_message != nullptr) {
    (*(cinfo->err->format_message))(cinfo, jpeg_last_error_msg);
  }
  throw std::runtime_error(jpeg_last_error_msg);
}

Status DvppProcess::GetJpegWidthHeight(const void *pic_buffer, size_t pic_buffer_size, uint32_t &image_width,
                                       uint32_t &image_height) {
  struct jpeg_decompress_struct jpeg_header;
  struct jpeg_error_mgr jpeg_error;
  jpeg_header.err = jpeg_std_error(&jpeg_error);
  jpeg_error.error_exit = JpegErrorExitCustom;
  try {
    jpeg_create_decompress(&jpeg_header);
    jpeg_mem_src(&jpeg_header, reinterpret_cast<const unsigned char *>(pic_buffer), pic_buffer_size);
    (void)jpeg_read_header(&jpeg_header, TRUE);
  } catch (std::runtime_error &e) {
    jpeg_destroy_decompress(&jpeg_header);
    MSI_LOG_ERROR << "jpeg images read failed, " << e.what();
    return INFER_STATUS(INVALID_INPUTS) << "jpeg images decode failed";
  }
  image_width = jpeg_header.image_width;
  image_height = jpeg_header.image_height;

  if (jpeg_header.jpeg_color_space != JCS_YCbCr) {
    MSI_LOG_ERROR << "Expect color space YUV(YCbCr), current " << jpeg_header.jpeg_color_space;
    jpeg_destroy_decompress(&jpeg_header);
    return INFER_STATUS(INVALID_INPUTS) << "Expect color space YUV(YCbCr), current " << jpeg_header.jpeg_color_space;
  }
  if (jpeg_header.dc_huff_tbl_ptrs[0] == nullptr) {
    MSI_LOG_ERROR << "Only support Huffman code";
    jpeg_destroy_decompress(&jpeg_header);
    return INFER_STATUS(INVALID_INPUTS) << "Only support Huffman code";
  }
  jpeg_destroy_decompress(&jpeg_header);

  const uint32_t min_width = 32;
  const uint32_t max_width = 8192;
  const uint32_t min_height = 32;
  const uint32_t max_height = 8192;
  if (image_width < min_width || image_width > max_width) {
    MSI_LOG_ERROR << "expect image width [" << min_width << ", " << max_width << "], the real image width is "
                  << image_width;
    return INFER_STATUS(INVALID_INPUTS) << "expect image width [" << min_width << ", " << max_width
                                        << "], the real image width is " << image_width;
  }
  if (image_height < min_height || image_height > max_height) {
    MSI_LOG_ERROR << "expect image height [" << min_height << ", " << max_height << "], the real image height is "
                  << image_height;
    return INFER_STATUS(INVALID_INPUTS) << "expect image height [" << min_height << ", " << max_height
                                        << "], the real image height is " << image_height;
  }
  return SUCCESS;
}

Status DvppProcess::Process(const void *pic_buffer, size_t pic_buffer_size, void *&output_device_buffer,
                            size_t &output_size) {
  if (dvpp_channel_desc_ == nullptr) {
    MSI_LOG_ERROR << "Process failed, dvpp not inited";
    return FAILED;
  }
  uint32_t image_width = 0;
  uint32_t image_height = 0;
  Status ret = GetJpegWidthHeight(pic_buffer, pic_buffer_size, image_width, image_height);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "Get jpeg image height and width failed";
    return ret;
  }
  MSI_LOG_INFO << "Get jpeg width " << image_width << ", height " << image_height;
  ret = InitDecodeOutputDesc(image_width, image_height);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "InitDecodeOutputDesc failed";
    return FAILED;
  }
  ret = UpdateCropArea(image_width, image_height);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "Update crop area failed";
    return ret;
  }
  ret = CheckResizeImageInfo(image_width, image_height);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "Check resize para failed";
    return ret;
  }
  if (InputInputBuffer(pic_buffer, pic_buffer_size) != SUCCESS) {
    MSI_LOG_ERROR << "InputInputBuffer failed";
    return FAILED;
  }
  if (ProcessDecode() != SUCCESS) {
    MSI_LOG_ERROR << "Process Decode failed";
    return INFER_STATUS(INVALID_INPUTS) << "Decode image failed";
  }
  MSI_LOG_INFO << "Process Decode success";
  if (to_resize_flag_) {
    if (ProcessResize() != SUCCESS) {
      MSI_LOG_ERROR << "Process Resize failed";
      return INFER_STATUS(FAILED) << "Resize image failed";
    }
    MSI_LOG_INFO << "Process Resize success";
  } else if (to_crop_flag_) {
    if (ProcessCrop() != SUCCESS) {
      MSI_LOG_ERROR << "Process Crop failed";
      return INFER_STATUS(FAILED) << "Crop image failed";
    }
    MSI_LOG_INFO << "Process Crop success";
  } else if (to_crop_and_paste_flag_) {
    if (ProcessCropAndPaste() != SUCCESS) {
      MSI_LOG_ERROR << "Process Crop And Paste failed";
      return INFER_STATUS(FAILED) << "Crop And Paste image failed";
    }
    MSI_LOG_INFO << "Process Crop And Paste success";
  }
  if (vpc_output_buffer_dev_ == nullptr) {
    output_device_buffer = decode_output_buffer_dev_;
    output_size = decode_output_buffer_size_;
  } else {
    output_device_buffer = vpc_output_buffer_dev_;
    output_size = vpc_output_buffer_size_;
  }
  MSI_LOG_INFO << "Process dvpp success";
  return SUCCESS;
}

Status DvppProcess::Process(const std::vector<const void *> &pic_buffer_list,
                            const std::vector<size_t> &pic_buffer_size_list, void *&output_device_buffer,
                            size_t &output_size) {
  auto batch_size = pic_buffer_list.size();
  if (batch_size == 0 || batch_size != pic_buffer_size_list.size()) {
    MSI_LOG_ERROR << "invalid batch size " << batch_size << ", pic size count" << pic_buffer_size_list.size();
    return FAILED;
  }
  MSI_LOG_INFO << "Begin dvpp process, batch size " << batch_size;
  if (batch_size == 1) {
    return Process(pic_buffer_list[0], pic_buffer_size_list[0], output_device_buffer, output_size);
  }
  size_t total_buffer_size = vpc_output_buffer_size_ * batch_size;
  if (batch_size_ != batch_size) {
    if (batch_vpc_output_buffer_dev_ != nullptr) {
      acldvppFree(batch_vpc_output_buffer_dev_);
      batch_vpc_output_buffer_dev_ = nullptr;
    }
    batch_size_ = batch_size;
    auto acl_rt = acldvppMalloc(&batch_vpc_output_buffer_dev_, total_buffer_size);
    if (acl_rt != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "acldvppMalloc failed, buffer size " << total_buffer_size;
      return FAILED;
    }
  }
  for (size_t i = 0; i < batch_size; i++) {
    const void *pic_buffer = pic_buffer_list[i];
    uint32_t pic_size = pic_buffer_size_list[i];
    if (pic_buffer == nullptr || pic_size == 0) {
      MSI_LOG_ERROR << "Get " << 0 << "th images failed";
      return FAILED;
    }
    void *output_dev_buffer_tmp = nullptr;
    size_t output_buffer_size_tmp = 0;
    Status ret = Process(pic_buffer, pic_size, output_dev_buffer_tmp, output_buffer_size_tmp);
    if (ret != SUCCESS) {
      MSI_LOG_ERROR << "dvpp process failed";
      return ret;
    }
    aclrtMemcpy(static_cast<uint8_t *>(batch_vpc_output_buffer_dev_) + vpc_output_buffer_size_ * i,
                total_buffer_size - vpc_output_buffer_size_ * i, output_dev_buffer_tmp, vpc_output_buffer_size_,
                ACL_MEMCPY_DEVICE_TO_DEVICE);

    MSI_LOG_INFO << "Dvpp process " << i << " th images success, input pic size " << pic_size << " output buffer size "
                 << output_buffer_size_tmp;
  }
  output_device_buffer = batch_vpc_output_buffer_dev_;
  output_size = total_buffer_size;
  MSI_LOG_INFO << "End dvpp process, batch size " << batch_size << ", output size " << output_size;
  return SUCCESS;
}

uint32_t DvppProcess::AlignmentHelper(uint32_t org_size, uint32_t alignment) const {
  if (alignment == 0) {
    return 0;
  }
  return (org_size + alignment - 1) / alignment * alignment;
}

uint32_t DvppProcess::GetImageBufferSize(uint32_t stride_width, uint32_t stride_height,
                                         acldvppPixelFormat pixel_format) const {
  if (stride_height == 0 || stride_width == 0) {
    MSI_LOG_ERROR << "invalid stride height or width, stride_width " << stride_width << " stride_height "
                  << stride_height;
    return 0;
  }
  if (UINT32_MAX / 3 < stride_height || UINT32_MAX / (3 * stride_height) < stride_width) {
    MSI_LOG_ERROR << "invalid stride height or width, stride_width " << stride_width << " stride_height "
                  << stride_height;
    return 0;
  }
  if (pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_420 || pixel_format == PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
    return stride_width * stride_height * 3 / 2;  // 420
  } else if (pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_422 || pixel_format == PIXEL_FORMAT_YVU_SEMIPLANAR_422) {
    return stride_width * stride_height * 2;  // 422
  } else if (pixel_format == PIXEL_FORMAT_YUV_SEMIPLANAR_444 || pixel_format == PIXEL_FORMAT_YVU_SEMIPLANAR_444) {
    return stride_width * stride_height * 3;  // 444
  }
  MSI_LOG_ERROR << "Not support pixel format " << pixel_format;
  return 0;
}

Status DvppProcess::GetPicDescStride(uint32_t width, uint32_t height, uint32_t &stride_width, uint32_t &stride_height) {
  const uint32_t width_alignment = 16;
  const uint32_t height_alignment = 2;
  const uint32_t stride_width_minimum = 32;
  const uint32_t stride_width_maximum = 4096;
  const uint32_t stride_height_minimum = 6;
  const uint32_t stride_height_maximum = 4096;

  stride_width = AlignmentHelper(width, width_alignment);
  stride_height = AlignmentHelper(height, height_alignment);
  if (stride_width == 0 || stride_height == 0) {
    MSI_LOG_ERROR << "Init VPC output desc failed, get stride width or height failed";
    return FAILED;
  }
  if (stride_width < stride_width_minimum || stride_width > stride_width_maximum) {
    MSI_LOG_ERROR << "Expect stride width [" << stride_width_minimum << ", " << stride_width_maximum
                  << "], current stride width " << stride_width << " given width " << width;
    return FAILED;
  }
  if (stride_height < stride_height_minimum || stride_height > stride_height_maximum) {
    MSI_LOG_ERROR << "Expect stride height [" << stride_height_minimum << ", " << stride_height_maximum
                  << "], current stride height " << stride_height << " given height " << height;
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::GetPicDescStrideDecode(uint32_t width, uint32_t height, uint32_t &stride_width,
                                           uint32_t &stride_height) {
  const uint32_t width_alignment = 128;
  const uint32_t height_alignment = 16;
  const uint32_t width_minimum = 32;
  const uint32_t width_maximum = 4096;  // decode support 8192, dvpp(resize/crop/crop&paste) support 4096
  const uint32_t height_minimum = 32;
  const uint32_t height_maximum = 4096;  // decode support 8192, dvpp(resize/crop/crop&paste) support 4096
  if (width < width_minimum || width > width_maximum) {
    MSI_LOG_ERROR << "Expect width [" << width_minimum << ", " << width_maximum << "], current width " << width;
    return INFER_STATUS(INVALID_INPUTS) << "Expect width [" << width_minimum << ", " << width_maximum
                                        << "], current width " << width;
  }
  if (height < height_minimum || height > height_maximum) {
    MSI_LOG_ERROR << "Expect height [" << height_minimum << ", " << height_maximum << "], current height " << height;
    return INFER_STATUS(INVALID_INPUTS) << "Expect height [" << height_minimum << ", " << height_maximum
                                        << "], current height " << height;
  }
  stride_width = AlignmentHelper(width, width_alignment);
  stride_height = AlignmentHelper(height, height_alignment);
  if (stride_width == 0 || stride_height == 0) {
    MSI_LOG_ERROR << "Init decode output desc failed, get stride width or height failed";
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::InitVpcOutputDesc(uint32_t output_width, uint32_t output_height, acldvppPixelFormat pixel_format) {
  DestroyVpcOutputDesc();
  uint32_t vpc_stride_width = 0;
  uint32_t vpc_stride_height = 0;
  if (GetPicDescStride(output_width, output_height, vpc_stride_width, vpc_stride_height) != SUCCESS) {
    MSI_LOG_ERROR << "Init VPC output desc failed, get VPC output stride width/height failed";
    return FAILED;
  }
  vpc_output_buffer_size_ = GetImageBufferSize(vpc_stride_width, vpc_stride_height, pixel_format);
  if (vpc_output_buffer_size_ == 0) {
    MSI_LOG_ERROR << "Init VPC output desc failed, get image buffer size failed";
    return FAILED;
  }
  auto acl_ret = acldvppMalloc(&vpc_output_buffer_dev_, vpc_output_buffer_size_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Init VPC output desc failed, malloc dvpp memory failed";
    return FAILED;
  }
  vpc_output_desc_ = acldvppCreatePicDesc();
  if (vpc_output_desc_ == nullptr) {
    MSI_LOG_ERROR << "Init VPC output desc failed, create pic desc failed";
    return FAILED;
  }
  acldvppSetPicDescData(vpc_output_desc_, vpc_output_buffer_dev_);
  acldvppSetPicDescSize(vpc_output_desc_, vpc_output_buffer_size_);
  acldvppSetPicDescFormat(vpc_output_desc_, pixel_format);
  acldvppSetPicDescWidth(vpc_output_desc_, output_width);
  acldvppSetPicDescHeight(vpc_output_desc_, output_height);
  acldvppSetPicDescWidthStride(vpc_output_desc_, vpc_stride_width);
  acldvppSetPicDescHeightStride(vpc_output_desc_, vpc_stride_height);
  MSI_LOG_INFO << "Init VPC output desc success";
  return SUCCESS;
}

void DvppProcess::DestroyVpcOutputDesc() {
  if (vpc_output_desc_ != nullptr) {
    acldvppDestroyPicDesc(vpc_output_desc_);
    vpc_output_desc_ = nullptr;
  }
  if (vpc_output_buffer_dev_ != nullptr) {
    acldvppFree(vpc_output_buffer_dev_);
    vpc_output_buffer_dev_ = nullptr;
  }
  if (batch_vpc_output_buffer_dev_ != nullptr) {
    acldvppFree(batch_vpc_output_buffer_dev_);
    batch_vpc_output_buffer_dev_ = nullptr;
  }
  vpc_output_buffer_size_ = 0;
  MSI_LOG_INFO << "End destroy vpc desc";
}

Status DvppProcess::InitDecodeOutputDesc(uint32_t image_width, uint32_t image_height) {
  if (decode_output_buffer_dev_ != nullptr && image_width == pic_width_ && image_height == pic_height_) {
    return SUCCESS;
  }
  DestroyDecodeDesc();

  pic_width_ = image_width;
  pic_height_ = image_height;

  uint32_t stride_width = 0;
  uint32_t stride_height = 0;
  Status ret = GetPicDescStrideDecode(pic_width_, pic_height_, stride_width, stride_height);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "Init VPC output desc failed, get VPC output stride width/height failed";
    return ret;
  }

  decode_output_buffer_size_ = GetImageBufferSize(stride_width, stride_height, decode_para_.pixel_format);
  if (decode_output_buffer_size_ == 0) {
    MSI_LOG_ERROR << "Init decode output desc failed, get image buffer size failed";
    return FAILED;
  }
  auto acl_ret = acldvppMalloc(&decode_output_buffer_dev_, decode_output_buffer_size_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Init decode output desc failed, malloc dvpp memory failed";
    return FAILED;
  }
  decode_output_desc_ = acldvppCreatePicDesc();
  if (decode_output_desc_ == nullptr) {
    MSI_LOG_ERROR << "Init decode output desc failed, create pic desc failed";
    return FAILED;
  }
  acldvppSetPicDescData(decode_output_desc_, decode_output_buffer_dev_);
  acldvppSetPicDescSize(decode_output_desc_, decode_output_buffer_size_);
  acldvppSetPicDescFormat(decode_output_desc_, decode_para_.pixel_format);
  acldvppSetPicDescWidth(decode_output_desc_, pic_width_);
  acldvppSetPicDescHeight(decode_output_desc_, pic_height_);
  acldvppSetPicDescWidthStride(decode_output_desc_, stride_width);
  acldvppSetPicDescHeightStride(decode_output_desc_, stride_height);
  MSI_LOG_INFO << "Init decode output desc success";
  return SUCCESS;
}

Status DvppProcess::CheckRoiAreaWidthHeight(uint32_t width, uint32_t height) {
  const uint32_t min_crop_width = 10;
  const uint32_t max_crop_width = 4096;
  const uint32_t min_crop_height = 6;
  const uint32_t max_crop_height = 4096;

  if (width < min_crop_width || width > max_crop_width) {
    MSI_LOG_ERROR << "Expect roi area width in [" << min_crop_width << ", " << max_crop_width << "], actually "
                  << width;
    return FAILED;
  }
  if (height < min_crop_height || height > max_crop_height) {
    MSI_LOG_ERROR << "Expect roi area height in [" << min_crop_height << ", " << max_crop_height << "], actually "
                  << height;
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::CheckAndAdjustRoiArea(DvppRoiArea &area) {
  if (area.right < area.left) {
    MSI_LOG_ERROR << "check roi area failed, left " << area.left << ", right " << area.right;
    return FAILED;
  }
  if (area.bottom < area.top) {
    MSI_LOG_ERROR << "check roi area failed, top " << area.top << ", bottom " << area.bottom;
    return FAILED;
  }

  area.left = ToEven(area.left);
  area.top = ToEven(area.top);
  area.right = ToOdd(area.right);
  area.bottom = ToOdd(area.bottom);

  auto width = area.right - area.left + 1;
  auto height = area.bottom - area.top + 1;
  if (CheckRoiAreaWidthHeight(width, height) != SUCCESS) {
    MSI_LOG_ERROR << "Check roi area width and height failed,"
                  << " actually width " << width << " left " << area.left << ", right " << area.right
                  << " actually height " << height << " top " << area.top << ", bottom " << area.bottom;
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::UpdateCropArea(uint32_t image_width, uint32_t image_height) {
  DvppCropInfo *crop_info = nullptr;
  if (to_crop_flag_) {
    crop_info = &crop_para_.crop_info;
  } else if (to_crop_and_paste_flag_) {
    crop_info = &crop_and_paste_para_.crop_info;
  } else {
    return SUCCESS;
  }
  if (crop_info->crop_type != kDvppCropTypeCentre) {
    return SUCCESS;
  }
  if (image_width < crop_info->crop_width) {
    MSI_LOG_ERROR << "Image width " << image_width << "smaller than crop width " << crop_info->crop_width;
    return INFER_STATUS(INVALID_INPUTS) << "Image width " << image_width << "smaller than crop width "
                                        << crop_info->crop_width;
  }
  if (image_height < crop_info->crop_height) {
    MSI_LOG_ERROR << "Image height " << image_height << "smaller than crop height " << crop_info->crop_height;
    return INFER_STATUS(INVALID_INPUTS) << "Image width " << image_width << "smaller than crop width "
                                        << crop_info->crop_width;
  }
  uint32_t left = ToEven((image_width - crop_info->crop_width) / 2);
  uint32_t top = ToEven((image_height - crop_info->crop_height) / 2);
  uint32_t right = ToOdd(left + crop_info->crop_width);
  uint32_t bottom = ToOdd(top + crop_info->crop_height);

  auto acl_ret = acldvppSetRoiConfig(crop_area_, left, right, top, bottom);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Update Crop Area failed";
    return FAILED;
  }
  MSI_LOG_INFO << "Update crop area, crop type centre, crop info: "
               << ", left " << left << ", right " << right << ", top " << top << ", bottom " << bottom;
  return SUCCESS;
}

Status DvppProcess::CheckResizeImageInfo(uint32_t image_width, uint32_t image_height) const {
  if (!to_resize_flag_) {
    return SUCCESS;
  }
  // resize ratio required [1/32, 16]
  auto check_resize_ratio = [](uint32_t before_resize, uint32_t after_resize) {
    if (before_resize == 0 || after_resize == 0) {
      return false;
    }
    if (before_resize / after_resize > 32) {
      return false;
    }
    if (after_resize / before_resize > 16) {
      return false;
    }
    return true;
  };
  if (!check_resize_ratio(image_width, resize_para_.output_width)) {
    MSI_LOG_ERROR << "Resize ratio required [1/32, 16], current width resize from " << image_width << " to "
                  << resize_para_.output_width;
    return INFER_STATUS(INVALID_INPUTS) << "Resize ratio required [1/32, 16], current width resize from " << image_width
                                        << " to " << resize_para_.output_width;
  }
  if (!check_resize_ratio(image_height, resize_para_.output_height)) {
    MSI_LOG_ERROR << "Resize ratio required [1/32, 16], current height resize from " << image_height << " to "
                  << resize_para_.output_height;
    return INFER_STATUS(INVALID_INPUTS) << "Resize ratio required [1/32, 16], current height resize from "
                                        << image_height << " to " << resize_para_.output_height;
  }
  return SUCCESS;
}

void DvppProcess::DestroyDecodeDesc() {
  if (decode_output_desc_ != nullptr) {
    acldvppDestroyPicDesc(decode_output_desc_);
    decode_output_desc_ = nullptr;
  }
  if (decode_output_buffer_dev_ != nullptr) {
    acldvppFree(decode_output_buffer_dev_);
    decode_output_buffer_dev_ = nullptr;
  }
  decode_output_buffer_size_ = 0;
  MSI_LOG_INFO << "End destroy decode desc";
}

Status DvppProcess::InitResizeOutputDesc() {
  if (InitVpcOutputDesc(resize_para_.output_width, resize_para_.output_height, decode_para_.pixel_format) != SUCCESS) {
    MSI_LOG_ERROR << "Init VPC output desc failed";
    return FAILED;
  }
  if (resize_config_ == nullptr) {
    resize_config_ = acldvppCreateResizeConfig();
    if (resize_config_ == nullptr) {
      MSI_LOG_ERROR << "Create Resize config failed";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status DvppProcess::InitRoiAreaConfig(acldvppRoiConfig *&roi_area, const DvppRoiArea &init_para) {
  if (roi_area == nullptr) {
    roi_area = acldvppCreateRoiConfig(init_para.left, init_para.right, init_para.top, init_para.bottom);
    if (roi_area == nullptr) {
      MSI_LOG_ERROR << "Create Roi config failed";
      return FAILED;
    }
  } else {
    auto acl_ret = acldvppSetRoiConfig(roi_area, init_para.left, init_para.right, init_para.top, init_para.bottom);
    if (acl_ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "Set Roi config failed";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status DvppProcess::InitCropOutputDesc() {
  if (InitVpcOutputDesc(crop_para_.output_width, crop_para_.output_height, decode_para_.pixel_format) != SUCCESS) {
    MSI_LOG_ERROR << "Init VPC output desc failed";
    return FAILED;
  }
  if (InitRoiAreaConfig(crop_area_, crop_para_.crop_info.crop_area) != SUCCESS) {
    MSI_LOG_ERROR << "Init crop area failed";
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::InitCropAndPasteOutputDesc() {
  if (InitVpcOutputDesc(crop_and_paste_para_.output_width, crop_and_paste_para_.output_height,
                        decode_para_.pixel_format) != SUCCESS) {
    MSI_LOG_ERROR << "Init VPC output desc failed";
    return FAILED;
  }
  if (InitRoiAreaConfig(crop_area_, crop_and_paste_para_.crop_info.crop_area) != SUCCESS) {
    MSI_LOG_ERROR << "Init crop area failed";
    return FAILED;
  }
  if (InitRoiAreaConfig(paste_area_, crop_and_paste_para_.paste_area) != SUCCESS) {
    MSI_LOG_ERROR << "Init paste area failed";
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::ProcessDecode() {
  aclError acl_ret;
  acl_ret = acldvppJpegDecodeAsync(dvpp_channel_desc_, input_pic_dev_buffer_, input_pic_buffer_size_,
                                   decode_output_desc_, stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acldvppJpegDecodeAsync failed, acl return " << acl_ret;
    return FAILED;
  }
  acl_ret = aclrtSynchronizeStream(stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "aclrtSynchronizeStream failed, acl return " << acl_ret;
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::ProcessResize() {
  aclError acl_ret;
  acl_ret = acldvppVpcResizeAsync(dvpp_channel_desc_, decode_output_desc_, vpc_output_desc_, resize_config_, stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acldvppVpcResizeAsync failed, acl return " << acl_ret;
    return FAILED;
  }
  acl_ret = aclrtSynchronizeStream(stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "aclrtSynchronizeStream failed, acl return " << acl_ret;
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::ProcessCrop() {
  aclError acl_ret;
  acl_ret = acldvppVpcCropAsync(dvpp_channel_desc_, decode_output_desc_, vpc_output_desc_, crop_area_, stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acldvppVpcCropAsync failed, acl return " << acl_ret;
    return FAILED;
  }
  acl_ret = aclrtSynchronizeStream(stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "aclrtSynchronizeStream failed, acl return " << acl_ret;
    return FAILED;
  }
  return SUCCESS;
}

Status DvppProcess::ProcessCropAndPaste() {
  aclError acl_ret;
  acl_ret = acldvppVpcCropAndPasteAsync(dvpp_channel_desc_, decode_output_desc_, vpc_output_desc_, crop_area_,
                                        paste_area_, stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acldvppVpcCropAndPasteAsync failed, acl return " << acl_ret;
    return FAILED;
  }
  acl_ret = aclrtSynchronizeStream(stream_);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "aclrtSynchronizeStream failed, acl return " << acl_ret;
    return FAILED;
  }
  return SUCCESS;
}

Status DvppJsonConfigParser::GetStringValue(const nlohmann::json &json_item, const std::string &key, std::string &val) {
  auto it = json_item.find(key);
  if (it == json_item.end()) {
    MSI_LOG_ERROR << "get string item " << key << " failed";
    return FAILED;
  }
  if (!it->is_string()) {
    MSI_LOG_ERROR << "item " << key << " value is not string type";
    return FAILED;
  }
  val = it->get<std::string>();
  return SUCCESS;
}

Status DvppJsonConfigParser::GetIntValue(const nlohmann::json &json_item, const std::string &key, uint32_t &val) {
  auto it = json_item.find(key);
  if (it == json_item.end()) {
    MSI_LOG_ERROR << "get string item " << key << " failed";
    return FAILED;
  }
  if (!it->is_number_integer()) {
    MSI_LOG_ERROR << "item " << key << " value is not integer type";
    return FAILED;
  }
  val = it->get<uint32_t>();
  return SUCCESS;
}

Status DvppJsonConfigParser::ParseInputPara(const nlohmann::json &preprocess_item) {
  auto input = preprocess_item.find("input");
  if (input == preprocess_item.end()) {
    MSI_LOG_ERROR << "get input failed";
    return FAILED;
  }
  if (!input->is_object()) {
    MSI_LOG_ERROR << "input is not object";
    return FAILED;
  }
  return SUCCESS;
}

Status DvppJsonConfigParser::ParseDecodePara(const nlohmann::json &preprocess_item) {
  auto decode_para = preprocess_item.find("decode_para");
  if (decode_para == preprocess_item.end()) {
    MSI_LOG_ERROR << "get input failed";
    return FAILED;
  }
  if (!decode_para->is_object()) {
    MSI_LOG_ERROR << "input is not object";
    return FAILED;
  }
  const std::unordered_map<std::string, acldvppPixelFormat> pixel_format_map = {
    {"YUV420SP", PIXEL_FORMAT_YUV_SEMIPLANAR_420}, {"YVU420SP", PIXEL_FORMAT_YVU_SEMIPLANAR_420},
    {"YUV422SP", PIXEL_FORMAT_YUV_SEMIPLANAR_422}, {"YVU422SP", PIXEL_FORMAT_YVU_SEMIPLANAR_422},
    {"YUV444SP", PIXEL_FORMAT_YUV_SEMIPLANAR_444}, {"YVU444SP", PIXEL_FORMAT_YVU_SEMIPLANAR_444},
  };
  std::string pixel_format;
  if (GetStringValue(*decode_para, "out_pixel_format", pixel_format) != SUCCESS) {
    MSI_LOG_ERROR << "get op out_pixel_format failed";
    return FAILED;
  }
  auto format = pixel_format_map.find(pixel_format);
  if (format == pixel_format_map.end()) {
    MSI_LOG_ERROR << "unsupported out_pixel_format " << pixel_format;
    return FAILED;
  }
  decode_para_.pixel_format = format->second;
  return SUCCESS;
}

Status DvppJsonConfigParser::ParseResizePara(const nlohmann::json &json_item) {
  if (GetIntValue(json_item, "out_width", resize_para_.output_width) != SUCCESS) {
    return FAILED;
  }
  if (GetIntValue(json_item, "out_height", resize_para_.output_height) != SUCCESS) {
    return FAILED;
  }
  resize_flag_ = true;
  return SUCCESS;
}

Status DvppJsonConfigParser::ParseCropPara(const nlohmann::json &json_item) {
  if (GetIntValue(json_item, "out_width", crop_para_.output_width) != SUCCESS) {
    return FAILED;
  }
  if (GetIntValue(json_item, "out_height", crop_para_.output_height) != SUCCESS) {
    return FAILED;
  }
  auto &crop_info = crop_para_.crop_info;
  std::string crop_type = "crop_type";
  if (GetStringValue(json_item, "crop_type", crop_type) != SUCCESS) {
    return FAILED;
  }
  if (crop_type == "offset") {
    MSI_LOG_INFO << "Crop type is 'offset'";
    crop_info.crop_type = kDvppCropTypeOffset;
    auto &crop_area = crop_info.crop_area;
    if (GetIntValue(json_item, "crop_left", crop_area.left) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_top", crop_area.top) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_right", crop_area.right) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_bottom", crop_area.bottom) != SUCCESS) {
      return FAILED;
    }
  } else if (crop_type == "centre") {
    MSI_LOG_INFO << "Crop type is 'centre'";
    if (GetIntValue(json_item, "crop_width", crop_info.crop_width) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_height", crop_info.crop_height) != SUCCESS) {
      return FAILED;
    }
    crop_info.crop_type = kDvppCropTypeCentre;
  } else {
    MSI_LOG_ERROR << "Invalid crop type " << crop_type << ", expect offset or centre";
    return FAILED;
  }
  crop_flag_ = true;
  return SUCCESS;
}

Status DvppJsonConfigParser::ParseCropAndPastePara(const nlohmann::json &json_item) {
  // crop info
  if (GetIntValue(json_item, "out_width", crop_and_paste_para_.output_width) != SUCCESS) {
    return FAILED;
  }
  if (GetIntValue(json_item, "out_height", crop_and_paste_para_.output_height) != SUCCESS) {
    return FAILED;
  }
  auto &crop_info = crop_and_paste_para_.crop_info;
  std::string crop_type = "crop_type";
  if (GetStringValue(json_item, "crop_type", crop_type) != SUCCESS) {
    return FAILED;
  }
  if (crop_type == "offset") {
    MSI_LOG_INFO << "Crop type is 'offset'";
    crop_info.crop_type = kDvppCropTypeOffset;
    auto &crop_area = crop_info.crop_area;
    if (GetIntValue(json_item, "crop_left", crop_area.left) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_top", crop_area.top) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_right", crop_area.right) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_bottom", crop_area.bottom) != SUCCESS) {
      return FAILED;
    }
  } else if (crop_type == "centre") {
    MSI_LOG_INFO << "Crop type is 'centre'";
    if (GetIntValue(json_item, "crop_width", crop_info.crop_width) != SUCCESS) {
      return FAILED;
    }
    if (GetIntValue(json_item, "crop_height", crop_info.crop_height) != SUCCESS) {
      return FAILED;
    }
    crop_info.crop_type = kDvppCropTypeCentre;
  } else {
    MSI_LOG_ERROR << "Invalid crop type " << crop_type << ", expect offset or centre";
    return FAILED;
  }
  // paste info
  auto &paste_area = crop_and_paste_para_.paste_area;
  if (GetIntValue(json_item, "paste_left", paste_area.left) != SUCCESS) {
    return FAILED;
  }
  if (GetIntValue(json_item, "paste_top", paste_area.top) != SUCCESS) {
    return FAILED;
  }
  if (GetIntValue(json_item, "paste_right", paste_area.right) != SUCCESS) {
    return FAILED;
  }
  if (GetIntValue(json_item, "paste_bottom", paste_area.bottom) != SUCCESS) {
    return FAILED;
  }
  crop_and_paste_flag_ = true;
  return SUCCESS;
}

Status DvppJsonConfigParser::InitWithJsonConfigImp(const std::string &json_config) {
  std::ifstream fp(json_config);
  if (!fp.is_open()) {
    MSI_LOG_ERROR << "read json config file failed";
    return FAILED;
  }
  const auto &model_info = nlohmann::json::parse(fp);
  auto preprocess_list = model_info.find("preprocess");
  if (preprocess_list == model_info.end()) {
    MSI_LOG_ERROR << "get preprocess failed";
    return FAILED;
  }
  if (!preprocess_list->is_array()) {
    MSI_LOG_ERROR << "preprocess is not array";
    return FAILED;
  }
  if (preprocess_list->empty()) {
    MSI_LOG_ERROR << "preprocess size is 0";
    return FAILED;
  }
  auto &preprocess = preprocess_list->at(0);
  // input
  if (ParseInputPara(preprocess) != SUCCESS) {
    MSI_LOG_ERROR << "parse input failed";
    return FAILED;
  }
  // decode para
  if (ParseDecodePara(preprocess) != SUCCESS) {
    MSI_LOG_ERROR << "parse decode failed";
    return FAILED;
  }
  // ops
  auto dvpp_process = preprocess.find("dvpp_process");
  if (dvpp_process == preprocess.end()) {
    MSI_LOG_ERROR << "get dvpp_process failed";
    return FAILED;
  }
  if (!dvpp_process->is_object()) {
    MSI_LOG_ERROR << "dvpp_process is not array";
    return FAILED;
  }
  const auto &item = *dvpp_process;
  std::string op_name;
  if (GetStringValue(item, "op_name", op_name) != SUCCESS) {
    return FAILED;
  }
  if (op_name == "resize") {
    if (ParseResizePara(item) != SUCCESS) {
      MSI_LOG_ERROR << "Parse resize para failed";
      return FAILED;
    }
  } else if (op_name == "crop") {
    if (ParseCropPara(item) != SUCCESS) {
      MSI_LOG_ERROR << "Parse crop para failed";
      return FAILED;
    }
  } else if (op_name == "crop_and_paste") {
    if (ParseCropAndPastePara(item) != SUCCESS) {
      MSI_LOG_ERROR << "Parse decode para failed";
      return FAILED;
    }
  } else {
    MSI_LOG_ERROR << "Unsupport op name " << op_name << ", expect resize, crop or crop_and_paste";
    return FAILED;
  }
  return SUCCESS;
}

Status DvppJsonConfigParser::InitWithJsonConfig(const std::string &json_config) {
  try {
    auto ret = InitWithJsonConfigImp(json_config);
    if (ret != SUCCESS) {
      MSI_LOG_ERROR << "init dvpp with json config failed, json config " << json_config;
      return FAILED;
    }
  } catch (nlohmann::json::exception &e) {
    MSI_LOG_ERROR << "init dvpp with json config failed, json config " << json_config << ", error: " << e.what();
    return FAILED;
  }
  MSI_LOG_INFO << "Init with json config " << json_config << " success";
  return SUCCESS;
}

Status DvppProcess::InitWithJsonConfig(const std::string &json_config) {
  DvppJsonConfigParser parser;
  if (parser.InitWithJsonConfig(json_config) != SUCCESS) {
    MSI_LOG_ERROR << "init json config failed";
    return FAILED;
  }
  if (InitJpegDecodePara(parser.GetDecodePara()) != SUCCESS) {
    MSI_LOG_ERROR << "init decode para failed";
    return FAILED;
  }
  if (parser.HasResizeConfig()) {
    if (InitResizePara(parser.GetResizePara()) != SUCCESS) {
      MSI_LOG_ERROR << "init resize para failed";
      return FAILED;
    }
  } else if (parser.HasCropConfig()) {
    if (InitCropPara(parser.GetCropPara()) != SUCCESS) {
      MSI_LOG_ERROR << "init crop para failed";
      return FAILED;
    }
  } else if (parser.HasCropAndPasteConfig()) {
    if (InitCropAndPastePara(parser.GetCropAndPastePara()) != SUCCESS) {
      MSI_LOG_ERROR << "init crop and paste para failed";
      return FAILED;
    }
  }
  return SUCCESS;
}

}  // namespace inference
}  // namespace mindspore
