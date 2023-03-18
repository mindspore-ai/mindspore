/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/image_utils.h"
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/image/affine_op.h"
#include "minddata/dataset/kernels/image/auto_contrast_op.h"
#include "minddata/dataset/kernels/image/invert_op.h"
#include "minddata/dataset/kernels/image/math_utils.h"
#include "minddata/dataset/kernels/image/posterize_op.h"
#include "minddata/dataset/kernels/image/resize_cubic_op.h"
#include "minddata/dataset/kernels/image/sharpness_op.h"
#include "minddata/dataset/kernels/image/solarize_op.h"
#include "minddata/dataset/kernels/data/data_utils.h"

const int32_t MAX_INT_PRECISION = 16777216;  // float int precision is 16777216
const int32_t DOUBLING_FACTOR = 2;           // used as multiplier with MAX_INT_PRECISION
const int32_t DEFAULT_NUM_HEIGHT = 1;
const int32_t DEFAULT_NUM_WIDTH = 1;

namespace mindspore {
namespace dataset {
int GetCVInterpolationMode(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::kLinear:
      return static_cast<int>(cv::InterpolationFlags::INTER_LINEAR);
    case InterpolationMode::kCubic:
      return static_cast<int>(cv::InterpolationFlags::INTER_CUBIC);
    case InterpolationMode::kArea:
      return static_cast<int>(cv::InterpolationFlags::INTER_AREA);
    case InterpolationMode::kNearestNeighbour:
      return static_cast<int>(cv::InterpolationFlags::INTER_NEAREST);
    default:
      return static_cast<int>(cv::InterpolationFlags::INTER_LINEAR);
  }
}

int GetCVBorderType(BorderType type) {
  switch (type) {
    case BorderType::kConstant:
      return static_cast<int>(cv::BorderTypes::BORDER_CONSTANT);
    case BorderType::kEdge:
      return static_cast<int>(cv::BorderTypes::BORDER_REPLICATE);
    case BorderType::kReflect:
      return static_cast<int>(cv::BorderTypes::BORDER_REFLECT101);
    case BorderType::kSymmetric:
      return static_cast<int>(cv::BorderTypes::BORDER_REFLECT);
    default:
      return static_cast<int>(cv::BorderTypes::BORDER_CONSTANT);
  }
}

Status GetConvertShape(ConvertMode convert_mode, const std::shared_ptr<CVTensor> &input_cv,
                       std::vector<dsize_t> *node) {
  std::vector<ConvertMode> one_channels = {ConvertMode::COLOR_BGR2GRAY, ConvertMode::COLOR_RGB2GRAY,
                                           ConvertMode::COLOR_BGRA2GRAY, ConvertMode::COLOR_RGBA2GRAY};
  std::vector<ConvertMode> three_channels = {
    ConvertMode::COLOR_BGRA2BGR, ConvertMode::COLOR_RGBA2RGB, ConvertMode::COLOR_RGBA2BGR, ConvertMode::COLOR_BGRA2RGB,
    ConvertMode::COLOR_BGR2RGB,  ConvertMode::COLOR_RGB2BGR,  ConvertMode::COLOR_GRAY2BGR, ConvertMode::COLOR_GRAY2RGB};
  std::vector<ConvertMode> four_channels = {ConvertMode::COLOR_BGR2BGRA,  ConvertMode::COLOR_RGB2RGBA,
                                            ConvertMode::COLOR_BGR2RGBA,  ConvertMode::COLOR_RGB2BGRA,
                                            ConvertMode::COLOR_BGRA2RGBA, ConvertMode::COLOR_RGBA2BGRA,
                                            ConvertMode::COLOR_GRAY2BGRA, ConvertMode::COLOR_GRAY2RGBA};
  if (std::find(three_channels.begin(), three_channels.end(), convert_mode) != three_channels.end()) {
    *node = {input_cv->shape()[0], input_cv->shape()[1], 3};
  } else if (std::find(four_channels.begin(), four_channels.end(), convert_mode) != four_channels.end()) {
    *node = {input_cv->shape()[0], input_cv->shape()[1], 4};
  } else if (std::find(one_channels.begin(), one_channels.end(), convert_mode) != one_channels.end()) {
    *node = {input_cv->shape()[0], input_cv->shape()[1]};
  } else {
    RETURN_STATUS_UNEXPECTED(
      "The mode of image channel conversion must be in ConvertMode, which mainly includes "
      "conversion between RGB, BGR, GRAY, RGBA etc.");
  }
  return Status::OK();
}

Status ImageNumChannels(const std::shared_ptr<Tensor> &image, dsize_t *channels) {
  RETURN_UNEXPECTED_IF_NULL(channels);
  if (image->Rank() < kMinImageRank) {
    RETURN_STATUS_UNEXPECTED(
      "GetImageNumChannels: invalid parameter, image should have at least two dimensions, but got: " +
      std::to_string(image->Rank()));
  } else if (image->Rank() == kMinImageRank) {
    *channels = 1;
  } else {
    *channels = image->shape()[-1];
  }
  return Status::OK();
}

Status ImageSize(const std::shared_ptr<Tensor> &image, std::vector<dsize_t> *size) {
  RETURN_UNEXPECTED_IF_NULL(size);
  *size = std::vector<dsize_t>(kMinImageRank);
  if (image->Rank() < kMinImageRank) {
    RETURN_STATUS_UNEXPECTED("GetImageSize: invalid parameter, image should have at least two dimensions, but got: " +
                             std::to_string(image->Rank()));
  } else if (image->Rank() == kMinImageRank) {
    (*size)[0] = image->shape()[0];
    (*size)[1] = image->shape()[1];
  } else {
    const int32_t kHeightIndex = -3;
    const int32_t kWidthIndex = -2;
    (*size)[0] = image->shape()[kHeightIndex];
    (*size)[1] = image->shape()[kWidthIndex];
  }
  return Status::OK();
}

Status ValidateImage(const std::shared_ptr<Tensor> &image, const std::string &op_name,
                     const std::set<uint8_t> &valid_dtype, const std::set<dsize_t> &valid_rank,
                     const std::set<dsize_t> &valid_channel) {
  // Validate image dtype
  if (!valid_dtype.empty()) {
    auto dtype = image->type();
    if (valid_dtype.find(dtype.value()) == valid_dtype.end()) {
      std::string err_msg = op_name + ": the data type of image tensor does not match the requirement of operator.";
      err_msg += " Expecting tensor in type of " + DataTypeSetToString(valid_dtype);
      err_msg += ". But got type " + dtype.ToString() + ".";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  // Validate image rank
  auto rank = image->Rank();
  if (!valid_rank.empty()) {
    if (valid_rank.find(rank) == valid_rank.end()) {
      std::string err_msg = op_name + ": the dimension of image tensor does not match the requirement of operator.";
      err_msg += " Expecting tensor in dimension of " + NumberSetToString(valid_rank);
      if (valid_rank == std::set<dsize_t>({kMinImageRank, kDefaultImageRank})) {
        err_msg += ", in shape of <H, W> or <H, W, C>";
      } else if (valid_rank == std::set<dsize_t>({kMinImageRank})) {
        err_msg += ", in shape of <H, W>";
      } else if (valid_rank == std::set<dsize_t>({kDefaultImageRank})) {
        err_msg += ", in shape of <H, W, C>";
      }
      err_msg += ". But got dimension " + std::to_string(rank) + ".";
      if (rank == 1) {
        err_msg += " You may need to perform Decode first.";
      }
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  } else {
    if (rank < kMinImageRank) {
      std::string err_msg =
        op_name + ": the image tensor should have at least two dimensions. You may need to perform Decode first.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  // Validate image channel
  if (!valid_channel.empty()) {
    dsize_t channel = 1;
    RETURN_IF_NOT_OK(ImageNumChannels(image, &channel));
    if (valid_channel.find(channel) == valid_channel.end()) {
      std::string err_msg = op_name + ": the channel of image tensor does not match the requirement of operator.";
      err_msg += " Expecting tensor in channel of " + NumberSetToString(valid_channel);
      err_msg += ". But got channel " + std::to_string(channel) + ".";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  return Status::OK();
}

Status ValidateImageDtype(const std::string &op_name, DataType dtype) {
  uint8_t type = dtype.AsCVType();
  if (type == kCVInvalidType) {
    std::string type_name = "unknown";
    if (dtype.value() < DataType::NUM_OF_TYPES) {
      type_name = std::string(DataType::kTypeInfo[dtype.value()].name_);
    }
    std::string err_msg = op_name + ": Cannot convert [" + type_name + "] to OpenCV type." +
                          " Currently unsupported data type: [uint32, int64, uint64, string]";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status ValidateImageRank(const std::string &op_name, int32_t rank) {
  if (rank != kMinImageRank && rank != kDefaultImageRank) {
    std::string err_msg =
      op_name + ": input tensor is not in shape of <H,W> or <H,W,C>, but got rank: " + std::to_string(rank);
    if (rank == 1) {
      err_msg = err_msg + ". You may need to perform Decode first.";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

bool CheckTensorShape(const std::shared_ptr<Tensor> &tensor, const int &channel) {
  if (tensor == nullptr) {
    return false;
  }
  bool rc = false;
  if (tensor->shape().Size() <= channel) {
    return false;
  }
  if (tensor->Rank() != kDefaultImageRank ||
      (tensor->shape()[channel] != 1 && tensor->shape()[channel] != kDefaultImageChannel)) {
    rc = true;
  }
  return rc;
}

Status Flip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, int flip_code) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Flip: load image failed.");
  }

  std::shared_ptr<CVTensor> output_cv;
  RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));

  try {
    cv::flip(input_cv->mat(), output_cv->mat(), flip_code);
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Flip: " + std::string(e.what()));
  }
  return Status::OK();
}

Status HorizontalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  return Flip(std::move(input), output, 1);
}

Status VerticalFlip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  return Flip(std::move(input), output, 0);
}

Status Resize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t output_height,
              int32_t output_width, double fx, double fy, InterpolationMode mode) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Resize: load image failed.");
  }
  RETURN_IF_NOT_OK(ValidateImageRank("Resize", input_cv->Rank()));

  cv::Mat in_image = input_cv->mat();
  const uint32_t kResizeShapeLimits = 1000;
  // resize image too large or too small, 1000 is arbitrarily chosen here to prevent open cv from segmentation fault
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kResizeShapeLimits) > in_image.rows,
                               "Resize: in_image rows out of bounds.");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kResizeShapeLimits) > in_image.cols,
                               "Resize: in_image cols out of bounds.");
  if (output_height > in_image.rows * kResizeShapeLimits || output_width > in_image.cols * kResizeShapeLimits) {
    std::string err_msg =
      "Resize: the resizing width or height is too big, it's 1000 times bigger than the original image, got output "
      "height: " +
      std::to_string(output_height) + ", width: " + std::to_string(output_width) +
      ", and original image size:" + std::to_string(in_image.rows) + ", " + std::to_string(in_image.cols);
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  if (output_height == 0 || output_width == 0) {
    std::string err_msg = "Resize: the input value of 'resize' is invalid, width or height is zero.";
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }

  if (mode == InterpolationMode::kCubicPil) {
    if (input_cv->shape().Size() != kDefaultImageChannel ||
        input_cv->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("Resize: Interpolation mode PILCUBIC only supports image with 3 channels, but got: " +
                               input_cv->shape().ToString());
    }

    LiteMat imIn, imOut;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({output_height, output_width, 3});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input_cv->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    imOut.Init(output_width, output_height, static_cast<int>(input_cv->shape()[kChannelIndexHWC]),
               reinterpret_cast<void *>(buffer), LDataType::UINT8);
    imIn.Init(static_cast<int>(input_cv->shape()[1]), static_cast<int>(input_cv->shape()[0]),
              static_cast<int>(input_cv->shape()[kChannelIndexHWC]), input_cv->mat().data, LDataType::UINT8);
    if (ResizeCubic(imIn, imOut, output_width, output_height) == false) {
      RETURN_STATUS_UNEXPECTED("Resize: failed to do resize, please check the error msg.");
    }
    *output = output_tensor;
    return Status::OK();
  }
  try {
    TensorShape shape{output_height, output_width};
    if (input_cv->Rank() == kDefaultImageRank) {
      int num_channels = static_cast<int>(input_cv->shape()[kChannelIndexHWC]);
      shape = shape.AppendDim(num_channels);
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(shape, input_cv->type(), &output_cv));

    auto cv_mode = GetCVInterpolationMode(mode);
    cv::resize(in_image, output_cv->mat(), cv::Size(output_width, output_height), fx, fy, cv_mode);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Resize: " + std::string(e.what()));
  }
}

const unsigned char kJpegMagic[] = "\xFF\xD8\xFF";
constexpr dsize_t kJpegMagicLen = 3;
const unsigned char kPngMagic[] = "\x89\x50\x4E\x47";
constexpr dsize_t kPngMagicLen = 4;

bool IsNonEmptyJPEG(const std::shared_ptr<Tensor> &input) {
  return input->SizeInBytes() > kJpegMagicLen && memcmp(input->GetBuffer(), kJpegMagic, kJpegMagicLen) == 0;
}

bool IsNonEmptyPNG(const std::shared_ptr<Tensor> &input) {
  return input->SizeInBytes() > kPngMagicLen && memcmp(input->GetBuffer(), kPngMagic, kPngMagicLen) == 0;
}

Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  RETURN_IF_NOT_OK(CheckUnsupportedImage(input));

  Status ret;
  if (IsNonEmptyJPEG(input)) {
    ret = JpegCropAndDecode(input, output);
  } else {
    ret = DecodeCv(input, output);
  }

  // decode failed and dump it
  if (ret != Status::OK()) {
    return DumpImageAndAppendStatus(input, ret);
  }
  return ret;
}

Status DecodeCv(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Decode: load image failed.");
  }
  try {
    cv::Mat img_mat = cv::imdecode(input_cv->mat(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (img_mat.data == nullptr) {
      std::string err = "Decode: image decode failed.";
      RETURN_STATUS_UNEXPECTED(err);
    }
    cv::cvtColor(img_mat, img_mat, static_cast<int>(cv::COLOR_BGR2RGB));
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(img_mat, 3, &output_cv));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Decode: " + std::string(e.what()));
  }
}

static void JpegInitSource(j_decompress_ptr cinfo) {}

static boolean JpegFillInputBuffer(j_decompress_ptr cinfo) {
  if (cinfo->src->bytes_in_buffer == 0) {
    // Under ARM platform raise runtime_error may cause core problem,
    // so we catch runtime_error and just return FALSE.
    try {
      ERREXIT(cinfo, JERR_INPUT_EMPTY);
    } catch (std::runtime_error &e) {
      return FALSE;
    }
    return FALSE;
  }
  return TRUE;
}

static void JpegTermSource(j_decompress_ptr cinfo) {}

static void JpegSkipInputData(j_decompress_ptr cinfo, int64_t jump) {
  if (jump < 0) {
    return;
  }
  if (static_cast<size_t>(jump) > cinfo->src->bytes_in_buffer) {
    cinfo->src->bytes_in_buffer = 0;
    return;
  } else {
    cinfo->src->bytes_in_buffer -= jump;
    cinfo->src->next_input_byte += jump;
  }
}

void JpegSetSource(j_decompress_ptr cinfo, const void *data, int64_t datasize) {
  cinfo->src = static_cast<struct jpeg_source_mgr *>(
    (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo), JPOOL_PERMANENT, sizeof(struct jpeg_source_mgr)));
  cinfo->src->init_source = JpegInitSource;
  cinfo->src->fill_input_buffer = JpegFillInputBuffer;
#if defined(_WIN32) || defined(_WIN64) || defined(ENABLE_ARM32) || defined(__APPLE__)
  cinfo->src->skip_input_data = reinterpret_cast<void (*)(j_decompress_ptr, long)>(JpegSkipInputData);
#else
  cinfo->src->skip_input_data = JpegSkipInputData;
#endif
  cinfo->src->resync_to_restart = jpeg_resync_to_restart;
  cinfo->src->term_source = JpegTermSource;
  cinfo->src->bytes_in_buffer = datasize;
  cinfo->src->next_input_byte = static_cast<const JOCTET *>(data);
}

thread_local std::vector<Status> jpeg_status;

Status CheckJpegExit(jpeg_decompress_struct *cinfo) {
  if (!jpeg_status.empty()) {
    jpeg_destroy_decompress(cinfo);
    Status s = jpeg_status[0];
    jpeg_status.clear();
    return s;
  }
  return Status::OK();
}

static Status JpegReadScanlines(jpeg_decompress_struct *const cinfo, int max_scanlines_to_read, JSAMPLE *buffer,
                                int buffer_size, int crop_w, int crop_w_aligned, int offset, int stride) {
  // scanlines will be read to this buffer first, must have the number
  // of components equal to the number of components in the image
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int64_t>::max() / cinfo->output_components) > crop_w_aligned,
                               "JpegReadScanlines: multiplication out of bounds.");
  int64_t scanline_size = crop_w_aligned * cinfo->output_components;
  std::vector<JSAMPLE> scanline(scanline_size);
  JSAMPLE *scanline_ptr = &scanline[0];
  while (cinfo->output_scanline < static_cast<unsigned int>(max_scanlines_to_read)) {
    int num_lines_read = 0;
    try {
      num_lines_read = jpeg_read_scanlines(cinfo, &scanline_ptr, 1);
      RETURN_IF_NOT_OK(CheckJpegExit(cinfo));
    } catch (std::runtime_error &e) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Decode: image decode failed.");
    }
    if (cinfo->out_color_space == JCS_CMYK && num_lines_read > 0) {
      for (int i = 0; i < crop_w; ++i) {
        const int cmyk_pixel = 4 * i + offset;
        const int c = scanline_ptr[cmyk_pixel];
        const int m = scanline_ptr[cmyk_pixel + 1];
        const int y = scanline_ptr[cmyk_pixel + 2];
        const int k = scanline_ptr[cmyk_pixel + 3];
        int r, g, b;
        if (cinfo->saw_Adobe_marker) {
          r = (k * c) / kMaxBitValue;
          g = (k * m) / kMaxBitValue;
          b = (k * y) / kMaxBitValue;
        } else {
          r = (kMaxBitValue - c) * (kMaxBitValue - k) / kMaxBitValue;
          g = (kMaxBitValue - m) * (kMaxBitValue - k) / kMaxBitValue;
          b = (kMaxBitValue - y) * (kMaxBitValue - k) / kMaxBitValue;
        }
        buffer[kDefaultImageChannel * i + kRIndex] = r;
        buffer[kDefaultImageChannel * i + kGIndex] = g;
        buffer[kDefaultImageChannel * i + kBIndex] = b;
      }
    } else if (num_lines_read > 0) {
      int copy_status = memcpy_s(buffer, buffer_size, scanline_ptr + offset, stride);
      if (copy_status != 0) {
        jpeg_destroy_decompress(cinfo);
        RETURN_STATUS_UNEXPECTED("[Internal ERROR] Decode: memcpy failed.");
      }
    } else {
      jpeg_destroy_decompress(cinfo);
      std::string err_msg = "[Internal ERROR] Decode: image decode failed.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    buffer += stride;
    buffer_size = buffer_size - stride;
  }
  return Status::OK();
}

static Status JpegSetColorSpace(jpeg_decompress_struct *cinfo) {
  switch (cinfo->num_components) {
    case 1:
      // we want to output 3 components if it's grayscale
      cinfo->out_color_space = JCS_RGB;
      return Status::OK();
    case 3:
      cinfo->out_color_space = JCS_RGB;
      return Status::OK();
    case 4:
      // Need to manually convert to RGB
      cinfo->out_color_space = JCS_CMYK;
      return Status::OK();
    default:
      jpeg_destroy_decompress(cinfo);
      std::string err_msg = "[Internal ERROR] Decode: image decode failed.";
      RETURN_STATUS_UNEXPECTED(err_msg);
  }
}

void JpegErrorExitCustom(j_common_ptr cinfo) {
  char jpeg_error_msg[JMSG_LENGTH_MAX];
  (*(cinfo->err->format_message))(cinfo, jpeg_error_msg);
  // we encounter core dump when execute jpeg_start_decompress at arm platform,
  // so we collect Status instead of throwing exception.
  jpeg_status.push_back(
    STATUS_ERROR(StatusCode::kMDUnexpectedError, "Error raised by libjpeg: " + std::string(jpeg_error_msg)));
}

Status JpegCropAndDecode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int crop_x, int crop_y,
                         int crop_w, int crop_h) {
  struct jpeg_decompress_struct cinfo;
  auto DestroyDecompressAndReturnError = [&cinfo](const std::string &err) {
    jpeg_destroy_decompress(&cinfo);
    RETURN_STATUS_UNEXPECTED(err);
  };
  struct JpegErrorManagerCustom jerr;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = JpegErrorExitCustom;
  try {
    jpeg_create_decompress(&cinfo);
    JpegSetSource(&cinfo, input->GetBuffer(), input->SizeInBytes());
    (void)jpeg_read_header(&cinfo, TRUE);
    RETURN_IF_NOT_OK(JpegSetColorSpace(&cinfo));
    jpeg_calc_output_dimensions(&cinfo);
    RETURN_IF_NOT_OK(CheckJpegExit(&cinfo));
  } catch (std::runtime_error &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - crop_w) > crop_x,
                               "JpegCropAndDecode: addition(crop x and crop width) out of bounds, got crop x:" +
                                 std::to_string(crop_x) + ", and crop width:" + std::to_string(crop_w));
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - crop_h) > crop_y,
                               "JpegCropAndDecode: addition(crop y and crop height) out of bounds, got crop y:" +
                                 std::to_string(crop_y) + ", and crop height:" + std::to_string(crop_h));
  if (crop_x == 0 && crop_y == 0 && crop_w == 0 && crop_h == 0) {
    crop_w = cinfo.output_width;
    crop_h = cinfo.output_height;
  } else if (crop_w == 0 || static_cast<unsigned int>(crop_w + crop_x) > cinfo.output_width || crop_h == 0 ||
             static_cast<unsigned int>(crop_h + crop_y) > cinfo.output_height) {
    return DestroyDecompressAndReturnError(
      "Crop: invalid crop size, corresponding crop value equal to 0 or too big, got crop width: " +
      std::to_string(crop_w) + ", crop height:" + std::to_string(crop_h) +
      ", and crop x coordinate:" + std::to_string(crop_x) + ", crop y coordinate:" + std::to_string(crop_y));
  }
  const int mcu_size = cinfo.min_DCT_scaled_size;
  CHECK_FAIL_RETURN_UNEXPECTED(mcu_size != 0, "JpegCropAndDecode: divisor mcu_size is zero.");
  unsigned int crop_x_aligned = (crop_x / mcu_size) * mcu_size;
  unsigned int crop_w_aligned = crop_w + crop_x - crop_x_aligned;
  try {
    bool status = jpeg_start_decompress(&cinfo);
    CHECK_FAIL_RETURN_UNEXPECTED(status, "JpegCropAndDecode: fail to decode, jpeg maybe a multi-scan file or broken.");
    RETURN_IF_NOT_OK(CheckJpegExit(&cinfo));
    jpeg_crop_scanline(&cinfo, &crop_x_aligned, &crop_w_aligned);
    RETURN_IF_NOT_OK(CheckJpegExit(&cinfo));
  } catch (std::runtime_error &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  JDIMENSION skipped_scanlines = jpeg_skip_scanlines(&cinfo, crop_y);
  // three number of output components, always convert to RGB and output
  constexpr int kOutNumComponents = 3;
  TensorShape ts = TensorShape({crop_h, crop_w, kOutNumComponents});
  std::shared_ptr<Tensor> output_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(ts, DataType(DataType::DE_UINT8), &output_tensor));
  const int buffer_size = output_tensor->SizeInBytes();
  JSAMPLE *buffer = reinterpret_cast<JSAMPLE *>(&(*output_tensor->begin<uint8_t>()));
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() - skipped_scanlines) > crop_h,
                               "JpegCropAndDecode: addition out of bounds.");
  const int max_scanlines_to_read = skipped_scanlines + crop_h;
  // stride refers to output tensor, which has 3 components at most
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() / crop_w) > kOutNumComponents,
                               "JpegCropAndDecode: multiplication out of bounds.");
  const int stride = crop_w * kOutNumComponents;
  // offset is calculated for scanlines read from the image, therefore
  // has the same number of components as the image
  int minius_value = crop_x - crop_x_aligned;
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / minius_value) > cinfo.output_components,
                               "JpegCropAndDecode: multiplication out of bounds.");
  const int offset = minius_value * cinfo.output_components;
  RETURN_IF_NOT_OK(
    JpegReadScanlines(&cinfo, max_scanlines_to_read, buffer, buffer_size, crop_w, crop_w_aligned, offset, stride));
  *output = output_tensor;
  jpeg_destroy_decompress(&cinfo);
  return Status::OK();
}

Status Rescale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rescale, float shift) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Rescale: load image failed.");
  }
  cv::Mat input_image = input_cv->mat();
  std::shared_ptr<CVTensor> output_cv;
  RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), DataType(DataType::DE_FLOAT32), &output_cv));
  try {
    input_image.convertTo(output_cv->mat(), CV_32F, rescale, shift);
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Rescale: " + std::string(e.what()));
  }
  return Status::OK();
}

Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Crop: load image failed.");
  }
  RETURN_IF_NOT_OK(ValidateImageRank("Crop", input_cv->Rank()));
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - y) > h,
                               "Crop: addition(x and height) out of bounds, got height:" + std::to_string(h) +
                                 ", and coordinate y:" + std::to_string(y));
  // account for integer overflow
  if (y < 0 || (y + h) > input_cv->shape()[0] || (y + h) < 0) {
    RETURN_STATUS_UNEXPECTED(
      "Crop: invalid y coordinate value for crop, y coordinate value exceeds the boundary of the image, got y: " +
      std::to_string(y));
  }
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - x) > w, "Crop: addition out of bounds.");
  // account for integer overflow
  if (x < 0 || (x + w) > input_cv->shape()[1] || (x + w) < 0) {
    RETURN_STATUS_UNEXPECTED(
      "Crop: invalid x coordinate value for crop, "
      "x coordinate value exceeds the boundary of the image, got x: " +
      std::to_string(x));
  }
  try {
    TensorShape shape{h, w};
    if (input_cv->Rank() == kDefaultImageRank) {
      int num_channels = static_cast<int>(input_cv->shape()[kChannelIndexHWC]);
      shape = shape.AppendDim(num_channels);
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(shape, input_cv->type(), &output_cv));
    cv::Rect roi(x, y, w, h);
    (input_cv->mat())(roi).copyTo(output_cv->mat());
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Crop: " + std::string(e.what()));
  }
}

Status ConvertColor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, ConvertMode convert_mode) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    RETURN_IF_NOT_OK(ValidateImageRank("ConvertColor", input_cv->Rank()));
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] ConvertColor: load image failed.");
    }
    if (input_cv->Rank() == kDefaultImageRank) {
      int num_channels = static_cast<int>(input_cv->shape()[kChannelIndexHWC]);
      if (num_channels != kMinImageChannel && num_channels != kDefaultImageChannel &&
          num_channels != kMaxImageChannel) {
        RETURN_STATUS_UNEXPECTED("ConvertColor: number of channels of image should be 1, 3, 4, but got:" +
                                 std::to_string(num_channels));
      }
    }
    std::vector<dsize_t> node;
    RETURN_IF_NOT_OK(GetConvertShape(convert_mode, input_cv, &node));
    if (node.empty()) {
      RETURN_STATUS_UNEXPECTED(
        "ConvertColor: convert mode must be in ConvertMode, which mainly includes conversion "
        "between RGB, BGR, GRAY, RGBA etc.");
    }
    TensorShape out_shape = TensorShape(node);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(convert_mode));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("ConvertColor: " + std::string(e.what()));
  }
}

Status HwcToChw(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  try {
    if (input->Rank() == kMinImageRank) {
      // If input tensor is 2D, we assume we have hw dimensions
      *output = input;
      return Status::OK();
    }
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] HWC2CHW: load image failed.");
    }
    if (input_cv->Rank() != kDefaultImageRank) {
      RETURN_STATUS_UNEXPECTED("HWC2CHW: image shape should be <H,W> or <H,W,C>, but got rank: " +
                               std::to_string(input_cv->Rank()));
    }
    cv::Mat output_img;

    int height = static_cast<int>(input_cv->shape()[0]);
    int width = static_cast<int>(input_cv->shape()[1]);
    int num_channels = static_cast<int>(input_cv->shape()[kChannelIndexHWC]);

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(TensorShape{num_channels, height, width}, input_cv->type(), &output_cv));

    for (int i = 0; i < num_channels; ++i) {
      cv::Mat mat;
      RETURN_IF_NOT_OK(output_cv->MatAtIndex({i}, &mat));
      cv::extractChannel(input_cv->mat(), mat, i);
    }
    *output = std::move(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("HWC2CHW: " + std::string(e.what()));
  }
}

Status MaskWithTensor(const std::shared_ptr<Tensor> &sub_mat, std::shared_ptr<Tensor> *input, int x, int y,
                      int crop_width, int crop_height, ImageFormat image_format) {
  constexpr int64_t input_shape = 2;
  if (image_format == ImageFormat::HWC) {
    if (CheckTensorShape(*input, input_shape)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "input shape doesn't match <H,W,C> format, got shape:" +
        (*input)->shape().ToString());
    }
    if (CheckTensorShape(sub_mat, input_shape)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "sub_mat shape doesn't match <H,W,C> format, got shape:" +
        (*input)->shape().ToString());
    }
    int number_of_channels = (*input)->shape()[kChannelIndexHWC];
    for (int i = 0; i < crop_width; i++) {
      for (int j = 0; j < crop_height; j++) {
        for (int c = 0; c < number_of_channels; c++) {
          RETURN_IF_NOT_OK(CopyTensorValue(sub_mat, input, {j, i, c}, {y + j, x + i, c}));
        }
      }
    }
  } else if (image_format == ImageFormat::CHW) {
    if (CheckTensorShape(*input, 0)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "input shape doesn't match <C,H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    if (CheckTensorShape(sub_mat, 0)) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "sub_mat shape doesn't match <C,H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    int number_of_channels = (*input)->shape()[0];
    for (int i = 0; i < crop_width; i++) {
      for (int j = 0; j < crop_height; j++) {
        for (int c = 0; c < number_of_channels; c++) {
          RETURN_IF_NOT_OK(CopyTensorValue(sub_mat, input, {c, j, i}, {c, y + j, x + i}));
        }
      }
    }
  } else if (image_format == ImageFormat::HW) {
    if ((*input)->Rank() != kMinImageRank) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "input shape doesn't match <H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    if (sub_mat->Rank() != kMinImageRank) {
      RETURN_STATUS_UNEXPECTED(
        "CutMixBatch: MaskWithTensor failed: "
        "sub_mat shape doesn't match <H,W> format, got shape:" +
        (*input)->shape().ToString());
    }
    for (int i = 0; i < crop_width; i++) {
      for (int j = 0; j < crop_height; j++) {
        RETURN_IF_NOT_OK(CopyTensorValue(sub_mat, input, {j, i}, {y + j, x + i}));
      }
    }
  } else {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: MaskWithTensor failed: "
      "image format must be <C,H,W>, <H,W,C>, or <H,W>, got shape:" +
      (*input)->shape().ToString());
  }
  return Status::OK();
}

Status CopyTensorValue(const std::shared_ptr<Tensor> &source_tensor, std::shared_ptr<Tensor> *dest_tensor,
                       const std::vector<int64_t> &source_indx, const std::vector<int64_t> &dest_indx) {
  if (source_tensor->type() != (*dest_tensor)->type()) {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: CopyTensorValue failed: "
      "source and destination tensor must have the same type.");
  }
  if (source_tensor->type() == DataType::DE_UINT8) {
    uint8_t pixel_value = 0;
    RETURN_IF_NOT_OK(source_tensor->GetItemAt(&pixel_value, source_indx));
    RETURN_IF_NOT_OK((*dest_tensor)->SetItemAt(dest_indx, pixel_value));
  } else if (source_tensor->type() == DataType::DE_FLOAT32) {
    float pixel_value = 0;
    RETURN_IF_NOT_OK(source_tensor->GetItemAt(&pixel_value, source_indx));
    RETURN_IF_NOT_OK((*dest_tensor)->SetItemAt(dest_indx, pixel_value));
  } else {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: CopyTensorValue failed: "
      "Tensor type is not supported. Tensor type must be float32 or uint8.");
  }
  return Status::OK();
}

Status SwapRedAndBlue(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "SwapRedBlue", {3, 5, 11}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    CHECK_FAIL_RETURN_UNEXPECTED(
      input_cv->shape().Size() > kChannelIndexHWC,
      "SwapRedAndBlue: rank of input data should be greater than:" + std::to_string(kChannelIndexHWC) +
        ", but got:" + std::to_string(input_cv->shape().Size()));
    int num_channels = static_cast<int>(input_cv->shape()[kChannelIndexHWC]);
    if (input_cv->shape().Size() != kDefaultImageRank || num_channels != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("SwapRedBlue: image shape should be in <H,W,C> format, but got:" +
                               input_cv->shape().ToString());
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));

    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_BGR2RGB));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("SwapRedBlue: " + std::string(e.what()));
  }
}

Status CropAndResize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y,
                     int crop_height, int crop_width, int target_height, int target_width, InterpolationMode mode) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] CropAndResize: load image failed.");
    }
    RETURN_IF_NOT_OK(ValidateImageRank("CropAndResize", input_cv->Rank()));
    // image too large or too small, 1000 is arbitrary here to prevent opencv from segmentation fault
    const uint32_t kCropShapeLimits = 1000;
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kCropShapeLimits) > crop_height,
                                 "CropAndResize: crop_height out of bounds.");
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int>::max() / kCropShapeLimits) > crop_width,
                                 "CropAndResize: crop_width out of bounds.");
    if (crop_height == 0 || crop_width == 0 || target_height == 0 || target_height > crop_height * kCropShapeLimits ||
        target_width == 0 || target_width > crop_width * kCropShapeLimits) {
      std::string err_msg =
        "CropAndResize: the resizing width or height 1) is too big, it's up to " + std::to_string(kCropShapeLimits) +
        " times the original image; 2) can not be 0. Detail info is: crop_height: " + std::to_string(crop_height) +
        ", crop_width: " + std::to_string(crop_width) + ", target_height: " + std::to_string(target_height) +
        ", target_width: " + std::to_string(target_width);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    cv::Rect roi(x, y, crop_width, crop_height);
    auto cv_mode = GetCVInterpolationMode(mode);
    cv::Mat cv_in = input_cv->mat();

    if (mode == InterpolationMode::kCubicPil) {
      if (input_cv->shape().Size() != kDefaultImageChannel ||
          input_cv->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
        RETURN_STATUS_UNEXPECTED(
          "CropAndResize: Interpolation mode PILCUBIC only supports image with 3 channels, but got: " +
          input_cv->shape().ToString());
      }

      cv::Mat input_roi = cv_in(roi);
      std::shared_ptr<CVTensor> input_image;
      RETURN_IF_NOT_OK(CVTensor::CreateFromMat(input_roi, input_cv->Rank(), &input_image));
      LiteMat imIn, imOut;
      std::shared_ptr<Tensor> output_tensor;
      TensorShape new_shape = TensorShape({target_height, target_width, 3});
      RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input_cv->type(), &output_tensor));
      uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
      imOut.Init(target_width, target_height, input_cv->shape()[kChannelIndexHWC], reinterpret_cast<void *>(buffer),
                 LDataType::UINT8);
      imIn.Init(input_image->shape()[1], input_image->shape()[0], input_image->shape()[kChannelIndexHWC],
                input_image->mat().data, LDataType::UINT8);
      if (ResizeCubic(imIn, imOut, target_width, target_height) == false) {
        RETURN_STATUS_UNEXPECTED("Resize: failed to do resize, please check the error msg.");
      }
      *output = output_tensor;
      return Status::OK();
    }

    TensorShape shape{target_height, target_width};
    if (input_cv->Rank() == kDefaultImageRank) {
      int num_channels = static_cast<int>(input_cv->shape()[kChannelIndexHWC]);
      shape = shape.AppendDim(num_channels);
    }
    std::shared_ptr<CVTensor> cvt_out;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(shape, input_cv->type(), &cvt_out));
    cv::resize(cv_in(roi), cvt_out->mat(), cv::Size(target_width, target_height), 0, 0, cv_mode);
    *output = std::static_pointer_cast<Tensor>(cvt_out);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("CropAndResize: " + std::string(e.what()));
  }
}

Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> center,
              float degree, InterpolationMode interpolation, bool expand, uint8_t fill_r, uint8_t fill_g,
              uint8_t fill_b) {
  try {
    RETURN_IF_NOT_OK(ValidateImageRank("Rotate", input->Rank()));
    dsize_t channel = 1;
    RETURN_IF_NOT_OK(ImageNumChannels(input, &channel));
    CHECK_FAIL_RETURN_UNEXPECTED(channel <= kMaxImageChannel || interpolation != InterpolationMode::kCubic,
                                 "Rotate: interpolation can not be CUBIC when image channel is greater than 4.");
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Rotate: load image failed.");
    }

    cv::Mat input_img = input_cv->mat();
    if (input_img.cols > (MAX_INT_PRECISION * DOUBLING_FACTOR) ||
        input_img.rows > (MAX_INT_PRECISION * DOUBLING_FACTOR)) {
      RETURN_STATUS_UNEXPECTED("Rotate: image is too large and center is not precise, got image width:" +
                               std::to_string(input_img.cols) + ", and image height:" + std::to_string(input_img.rows) +
                               ", both should be small than:" + std::to_string(MAX_INT_PRECISION * DOUBLING_FACTOR));
    }
    float fx = 0, fy = 0;
    if (center.empty()) {
      // default to center of image
      fx = (input_img.cols - 1) * kHalf;
      fy = (input_img.rows - 1) * kHalf;
    } else {
      fx = center[0];
      fy = center[1];
    }
    cv::Mat output_img;
    cv::Scalar fill_color = cv::Scalar(fill_b, fill_g, fill_r);
    // maybe don't use uint32 for image dimension here
    cv::Point2f pc(fx, fy);
    cv::Mat rot = cv::getRotationMatrix2D(pc, degree, 1.0);
    std::shared_ptr<CVTensor> output_cv;
    if (!expand) {
      // this case means that the shape doesn't change, size stays the same
      // We may not need this memcpy if it is in place.
      RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
      // using inter_nearest to comply with python default
      cv::warpAffine(input_img, output_cv->mat(), rot, input_img.size(), GetCVInterpolationMode(interpolation),
                     cv::BORDER_CONSTANT, fill_color);
    } else {
      // we resize here since the shape changes
      // create a new bounding box with the rotate
      cv::Rect2f bbox = cv::RotatedRect(pc, input_img.size(), degree).boundingRect2f();
      rot.at<double>(0, 2) += bbox.width / 2.0 - input_img.cols / 2.0;
      rot.at<double>(1, 2) += bbox.height / 2.0 - input_img.rows / 2.0;
      // use memcpy and don't compute the new shape since openCV has a rounding problem
      cv::warpAffine(input_img, output_img, rot, bbox.size(), GetCVInterpolationMode(interpolation),
                     cv::BORDER_CONSTANT, fill_color);
      RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_img, input_cv->Rank(), &output_cv));
      RETURN_UNEXPECTED_IF_NULL(output_cv);
    }
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Rotate: " + std::string(e.what()));
  }
  return Status::OK();
}

template <typename T1, typename T2>
void Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> mean,
               std::vector<float> std, bool is_hwc, bool pad = false) {
  // T1 is the type of input tensor, T2 is the type of output tensor
  auto itr_out = (*output)->begin<T2>();
  auto itr = input->begin<T1>();
  auto end = input->end<T1>();
  int64_t num_channels;
  if (is_hwc) {
    num_channels = (*output)->shape()[kChannelIndexHWC];
    while (itr != end) {
      for (size_t i = 0; i < num_channels - static_cast<int>(pad); i++) {
        *itr_out = static_cast<T2>((static_cast<float>(*itr) - mean[i]) / std[i]);
        ++itr_out;
        ++itr;
      }
    }
  } else {
    num_channels = (*output)->shape()[kChannelIndexCHW];
    int64_t height_index = 1;
    int64_t width_index = 2;
    int64_t channel_len = (*output)->shape()[height_index] * (*output)->shape()[width_index];
    while (itr != end) {
      for (size_t i = 0; i < num_channels - static_cast<int>(pad); i++) {
        for (int64_t j = 0; j < channel_len; j++) {
          *itr_out = static_cast<T2>((static_cast<float>(*itr) - mean[i]) / std[i]);
          ++itr_out;
          ++itr;
        }
      }
    }
  }
}

template <typename T>
Status Normalize_caller(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                        const std::vector<float> mean_v, const std::vector<float> std_v, bool is_hwc, bool pad) {
  switch (static_cast<int>(input->type().value())) {
    case DataType::DE_BOOL:
      Normalize<bool, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_INT8:
      Normalize<int8_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_UINT8:
      Normalize<uint8_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_INT16:
      Normalize<int16_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_UINT16:
      Normalize<uint16_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_INT32:
      Normalize<int32_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_UINT32:
      Normalize<uint32_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_INT64:
      Normalize<int64_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_UINT64:
      Normalize<uint64_t, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_FLOAT16:
      Normalize<float16, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_FLOAT32:
      Normalize<float, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    case DataType::DE_FLOAT64:
      Normalize<double, T>(input, output, mean_v, std_v, is_hwc, pad);
      break;
    default:
      std::string op_name = (pad) ? "NormalizePad" : "Normalize";
      RETURN_STATUS_UNEXPECTED(
        op_name + ": unsupported type, currently supported types include " +
        "[bool,int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,int64_t,uint64_t,float16,float,double].");
  }
  return Status::OK();
}

Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> mean,
                 std::vector<float> std, bool is_hwc) {
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), DataType(DataType::DE_FLOAT32), output));
  if (input->Rank() == kMinImageRank) {
    RETURN_IF_NOT_OK((*output)->ExpandDim(kMinImageRank));
  }

  CHECK_FAIL_RETURN_UNEXPECTED((*output)->Rank() == kDefaultImageRank,
                               "Normalize: output image rank should be: " + std::to_string(kDefaultImageRank) +
                                 ", but got: " + std::to_string((*output)->Rank()));
  CHECK_FAIL_RETURN_UNEXPECTED(std.size() == mean.size(),
                               "Normalize: mean and std vectors are not of same size, got size of std: " +
                                 std::to_string(std.size()) + ", and mean size: " + std::to_string(mean.size()));
  int64_t channel_index;
  if (is_hwc) {
    channel_index = kChannelIndexHWC;
  } else {
    channel_index = kChannelIndexCHW;
  }
  // caller provided 1 mean/std value and there is more than one channel --> duplicate mean/std value
  if (mean.size() == 1 && (*output)->shape()[channel_index] != 1) {
    for (int64_t i = 0; i < (*output)->shape()[channel_index] - 1; i++) {
      mean.push_back(mean[0]);
      std.push_back(std[0]);
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED((*output)->shape()[channel_index] == static_cast<dsize_t>(mean.size()),
                               "Normalize: number of channels does not match the size of mean and std vectors, got "
                               "channels: " +
                                 std::to_string((*output)->shape()[channel_index]) +
                                 ", size of mean: " + std::to_string(mean.size()));
  RETURN_IF_NOT_OK(Normalize_caller<float>(input, output, mean, std, is_hwc, false));

  if (input->Rank() == kMinImageRank) {
    (*output)->Squeeze();
  }
  return Status::OK();
}

Status NormalizePad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<float> mean,
                    std::vector<float> std, const std::string &dtype, bool is_hwc) {
  RETURN_IF_NOT_OK(ValidateImageRank("NormalizePad", input->Rank()));
  int64_t channel_index = kChannelIndexCHW;
  if (is_hwc) {
    channel_index = kChannelIndexHWC;
  }
  int32_t channels = 1;
  if (input->Rank() == kDefaultImageRank) {
    channels = static_cast<int>(input->shape()[channel_index]);
  }

  if (is_hwc) {
    TensorShape new_shape = TensorShape({input->shape()[0], input->shape()[1], channels + 1});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, DataType(dtype), output));
    RETURN_IF_NOT_OK((*output)->Zero());
  } else {
    TensorShape new_shape = TensorShape({channels + 1, input->shape()[1], input->shape()[2]});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, DataType(dtype), output));
    RETURN_IF_NOT_OK((*output)->Zero());
  }

  // caller provided 1 mean/std value and there are more than one channel --> duplicate mean/std value
  if (mean.size() == 1 && channels > 1) {
    while (mean.size() < channels) {
      mean.push_back(mean[0]);
      std.push_back(std[0]);
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED((*output)->shape()[channel_index] == static_cast<dsize_t>(mean.size()) + 1,
                               "NormalizePad: number of channels does not match the size of mean and std vectors, got "
                               "channels: " +
                                 std::to_string((*output)->shape()[channel_index] - 1) +
                                 ", size of mean: " + std::to_string(mean.size()));
  if (dtype == "float16") {
    RETURN_IF_NOT_OK(Normalize_caller<float16>(input, output, mean, std, is_hwc, true));
  } else {
    RETURN_IF_NOT_OK(Normalize_caller<float>(input, output, mean, std, is_hwc, true));
  }
  if (input->Rank() == kMinImageRank) {
    (*output)->Squeeze();
  }
  return Status::OK();
}

Status AdjustBrightness(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float alpha) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "AdjustBrightness", {1, 2, 3, 4, 5, 6, 10, 11, 12}, {3}, {3}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustBrightness: load image failed.");
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    output_cv->mat() = input_img * alpha;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustBrightness: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float alpha) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "AdjustContrast", {3, 5, 11}, {3}, {3}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustContrast: load image failed.");
    }
    cv::Mat gray, output_img;
    cv::cvtColor(input_img, gray, CV_RGB2GRAY);
    auto mean_img = cv::mean(gray).val[0];
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    // thread safe: change cv::Mat::zeros to cv::Mat + setTo
    output_img = cv::Mat(input_img.rows, input_img.cols, input_img.depth());
    output_img.setTo(cv::Scalar::all(0));
    output_img = output_img + mean_img;
    cv::cvtColor(output_img, output_img, CV_GRAY2RGB);
    output_img = output_img * (1.0 - alpha) + input_img * alpha;
    output_img.copyTo(output_cv->mat());
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustContrast: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustGamma(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float gamma, float gain) {
  try {
    int num_channels = 1;
    if (input->Rank() < kMinImageRank) {
      RETURN_STATUS_UNEXPECTED("AdjustGamma: input tensor is not in shape of <...,H,W,C> or <H,W>, got shape:" +
                               input->shape().ToString());
    }
    if (input->Rank() > 2) {
      num_channels = input->shape()[-1];
    }
    if (num_channels != 1 && num_channels != 3) {
      RETURN_STATUS_UNEXPECTED("AdjustGamma: channel of input image should be 1 or 3, but got: " +
                               std::to_string(num_channels));
    }
    if (input->type().IsFloat()) {
      for (auto itr = input->begin<float>(); itr != input->end<float>(); itr++) {
        *itr = pow((*itr) * gain, gamma);
        *itr = std::min(std::max((*itr), 0.0f), 1.0f);
      }
      *output = input;
    } else {
      std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
      if (!input_cv->mat().data) {
        RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustGamma: load image failed.");
      }
      cv::Mat input_img = input_cv->mat();
      std::shared_ptr<CVTensor> output_cv;
      RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
      uchar LUT[256] = {};
      auto kMaxPixelValueFloat = static_cast<float>(kMaxBitValue);
      for (int i = 0; i <= kMaxBitValue; i++) {
        float f = i / kMaxPixelValueFloat;
        f = pow(f, gamma);
        LUT[i] =
          static_cast<uchar>(floor(std::min(f * (kMaxPixelValueFloat + 1.f - 1e-3f) * gain, kMaxPixelValueFloat)));
      }
      if (input_img.channels() == 1) {
        cv::MatIterator_<uchar> it = input_img.begin<uchar>();
        cv::MatIterator_<uchar> it_end = input_img.end<uchar>();
        for (; it != it_end; ++it) {
          *it = LUT[(*it)];
        }
      } else {
        cv::MatIterator_<cv::Vec3b> it = input_img.begin<cv::Vec3b>();
        cv::MatIterator_<cv::Vec3b> it_end = input_img.end<cv::Vec3b>();
        for (; it != it_end; ++it) {
          (*it)[0] = LUT[(*it)[0]];
          (*it)[1] = LUT[(*it)[1]];
          (*it)[2] = LUT[(*it)[2]];
        }
      }
      output_cv->mat() = input_img * 1;
      *output = std::static_pointer_cast<Tensor>(output_cv);
    }
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustGamma: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AutoContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float cutoff,
                    const std::vector<uint32_t> &ignore) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AutoContrast: load image failed.");
    }
    if (input_cv->Rank() != kDefaultImageRank && input_cv->Rank() != kMinImageRank) {
      std::string err_msg = "AutoContrast: image rank should be 2 or 3,  but got: " + std::to_string(input_cv->Rank());
      if (input_cv->Rank() == 1) {
        err_msg = err_msg + ", may need to do Decode operation first.";
      }
      RETURN_STATUS_UNEXPECTED("AutoContrast: image rank should be 2 or 3,  but got: " +
                               std::to_string(input_cv->Rank()));
    }
    // Reshape to extend dimension if rank is 2 for algorithm to work. then reshape output to be of rank 2 like input
    if (input_cv->Rank() == kMinImageRank) {
      RETURN_IF_NOT_OK(input_cv->ExpandDim(kMinImageRank));
    }
    // Get number of channels and image matrix
    std::size_t num_of_channels = input_cv->shape()[static_cast<size_t>(kChannelIndexHWC)];
    if (num_of_channels != kMinImageChannel && num_of_channels != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("AutoContrast: channel of input image should be 1 or 3, but got: " +
                               std::to_string(num_of_channels));
    }
    cv::Mat image = input_cv->mat();
    // Separate the image to channels
    std::vector<cv::Mat> planes(num_of_channels);
    cv::split(image, planes);
    cv::Mat b_hist, g_hist, r_hist;
    // Establish the number of bins and set variables for histogram
    int32_t hist_size = 256;
    int32_t channels = 0;
    float range[] = {0, 256};
    const float *hist_range[] = {range};
    bool uniform = true, accumulate = false;
    // Set up lookup table for LUT(Look up table algorithm)
    std::vector<int32_t> table;
    std::vector<cv::Mat> image_result;
    for (std::size_t layer = 0; layer < planes.size(); layer++) {
      // Reset lookup table
      table = std::vector<int32_t>{};
      // Calculate Histogram for channel
      cv::Mat hist;
      cv::calcHist(&planes[layer], 1, &channels, cv::Mat(), hist, 1, &hist_size, hist_range, uniform, accumulate);
      hist.convertTo(hist, CV_32SC1);
      std::vector<int32_t> hist_vec;
      hist.col(0).copyTo(hist_vec);
      // Ignore values in ignore
      for (const auto &item : ignore) {
        hist_vec[item] = 0;
      }
      int32_t hi = kMaxBitValue;
      int32_t lo = 0;
      RETURN_IF_NOT_OK(ComputeUpperAndLowerPercentiles(&hist_vec, cutoff, cutoff, &hi, &lo));
      if (hi <= lo) {
        for (int32_t i = 0; i < 256; i++) {
          table.push_back(i);
        }
      } else {
        const float scale = static_cast<float>(kMaxBitValue) / (hi - lo);
        const float offset = -1 * lo * scale;
        for (int32_t i = 0; i < 256; i++) {
          int32_t ix = static_cast<int32_t>(i * scale + offset);
          ix = std::max(ix, 0);
          ix = std::min(ix, kMaxBitValue);
          table.push_back(ix);
        }
      }
      cv::Mat result_layer;
      cv::LUT(planes[layer], table, result_layer);
      image_result.push_back(result_layer);
    }
    cv::Mat result;
    cv::merge(image_result, result);
    result.convertTo(result, input_cv->mat().type());
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(result, input_cv->Rank(), &output_cv));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    RETURN_IF_NOT_OK((*output)->Reshape(input_cv->shape()));
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AutoContrast: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustSaturation(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float alpha) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "AdjustSaturation", {3, 5, 11}, {3}, {3}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustSaturation: load image failed.");
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    cv::Mat output_img = output_cv->mat();
    cv::Mat gray;
    cv::cvtColor(input_img, gray, CV_RGB2GRAY);
    cv::cvtColor(gray, output_img, CV_GRAY2RGB);
    output_img = output_img * (1.0 - alpha) + input_img * alpha;
    output_img.copyTo(output_cv->mat());
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustSaturation: " + std::string(e.what()));
  }
  return Status::OK();
}

Status AdjustHue(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float hue) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "AdjustHue", {3, 11}, {3}, {3}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] AdjustHue: load image failed.");
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    cv::Mat output_img;
    cv::cvtColor(input_img, output_img, CV_RGB2HSV_FULL);
    for (int y = 0; y < output_img.cols; y++) {
      for (int x = 0; x < output_img.rows; x++) {
        uint8_t cur1 = output_img.at<cv::Vec3b>(cv::Point(y, x))[0];
        uint8_t h_hue = 0;
        h_hue = static_cast<uint8_t>(hue * kMaxBitValue);
        cur1 += h_hue;
        output_img.at<cv::Vec3b>(cv::Point(y, x))[0] = cur1;
      }
    }
    cv::cvtColor(output_img, output_cv->mat(), CV_HSV2RGB_FULL);
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("AdjustHue: " + std::string(e.what()));
  }
  return Status::OK();
}

Status Equalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Equalize: load image failed.");
    }
    if (input_cv->Rank() != kDefaultImageRank && input_cv->Rank() != kMinImageRank) {
      RETURN_STATUS_UNEXPECTED("Equalize: image rank should be 2 or 3,  but got: " + std::to_string(input_cv->Rank()));
    }
    // For greyscale images, extend dimension if rank is 2 and reshape output to be of rank 2.
    if (input_cv->Rank() == kMinImageRank) {
      RETURN_IF_NOT_OK(input_cv->ExpandDim(kMinImageRank));
    }
    // Get number of channels and image matrix
    std::size_t num_of_channels = input_cv->shape()[kChannelIndexHWC];
    if (num_of_channels != kMinImageChannel && num_of_channels != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("Equalize: channel of input image should be 1 or 3, but got: " +
                               std::to_string(num_of_channels));
    }
    cv::Mat image = input_cv->mat();
    // Separate the image to channels
    std::vector<cv::Mat> planes(num_of_channels);
    cv::split(image, planes);
    // Equalize each channel separately
    std::vector<cv::Mat> image_result;
    for (std::size_t layer = 0; layer < planes.size(); layer++) {
      cv::Mat channel_result;
      cv::equalizeHist(planes[layer], channel_result);
      image_result.push_back(channel_result);
    }
    cv::Mat result;
    cv::merge(image_result, result);
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(result, input_cv->Rank(), &output_cv));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    RETURN_IF_NOT_OK((*output)->Reshape(input_cv->shape()));
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Equalize: " + std::string(e.what()));
  }
  return Status::OK();
}

Status ValidateCutOutImage(const std::shared_ptr<Tensor> &input, bool is_hwc, int32_t box_height, int32_t box_width) {
  uint32_t channel_index = is_hwc ? kChannelIndexHWC : kChannelIndexCHW;
  uint32_t height_index = is_hwc ? 0 : 1;
  uint32_t width_index = is_hwc ? 1 : 2;
  std::string right_shape = is_hwc ? "<H,W,C>" : "<C,H,W>";
  int64_t image_h = input->shape()[height_index];
  int64_t image_w = input->shape()[width_index];

  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Size() > channel_index, "CutOut: shape is invalid.");

  if (input->Rank() != kDefaultImageRank) {
    RETURN_STATUS_UNEXPECTED("CutOut: image shape is not " + right_shape +
                             ", but got rank: " + std::to_string(input->Rank()));
  }

  if (box_height > image_h || box_width > image_w) {
    RETURN_STATUS_UNEXPECTED(
      "CutOut: box size is too large for image erase, got box height: " + std::to_string(box_height) +
      "box weight: " + std::to_string(box_width) + ", and image height: " + std::to_string(image_h) +
      ", image width: " + std::to_string(image_w));
  }
  return Status::OK();
}

uchar *GetPtr(const std::shared_ptr<Tensor> &tensor) {
  switch (tensor->type().value()) {
    case DataType::DE_BOOL:
      return reinterpret_cast<uchar *>(&(*tensor->begin<bool>()));
    case DataType::DE_INT8:
      return reinterpret_cast<uchar *>(&(*tensor->begin<int8_t>()));
    case DataType::DE_UINT8:
      return reinterpret_cast<uchar *>(&(*tensor->begin<uint8_t>()));
    case DataType::DE_INT16:
      return reinterpret_cast<uchar *>(&(*tensor->begin<int16_t>()));
    case DataType::DE_UINT16:
      return reinterpret_cast<uchar *>(&(*tensor->begin<uint16_t>()));
    case DataType::DE_INT32:
      return reinterpret_cast<uchar *>(&(*tensor->begin<int32_t>()));
    case DataType::DE_UINT32:
      return reinterpret_cast<uchar *>(&(*tensor->begin<uint32_t>()));
    case DataType::DE_INT64:
      return reinterpret_cast<uchar *>(&(*tensor->begin<int64_t>()));
    case DataType::DE_UINT64:
      return reinterpret_cast<uchar *>(&(*tensor->begin<uint64_t>()));
    case DataType::DE_FLOAT16:
      return reinterpret_cast<uchar *>(&(*tensor->begin<float16>()));
    case DataType::DE_FLOAT32:
      return reinterpret_cast<uchar *>(&(*tensor->begin<float>()));
    case DataType::DE_FLOAT64:
      return reinterpret_cast<uchar *>(&(*tensor->begin<double>()));
    default:
      return nullptr;
  }
}

Status CutOut(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t box_height,
              int32_t box_width, int32_t num_patches, bool bounded, bool random_color, std::mt19937 *rnd,
              std::vector<uint8_t> fill_colors, bool is_hwc) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    RETURN_IF_NOT_OK(ValidateCutOutImage(input_cv, is_hwc, box_height, box_width));
    uint32_t channel_index = is_hwc ? kChannelIndexHWC : kChannelIndexCHW;
    uint32_t height_index = is_hwc ? 0 : 1;
    uint32_t width_index = is_hwc ? 1 : 2;
    uint64_t num_channels = input_cv->shape()[channel_index];
    int64_t image_h = input_cv->shape()[height_index];
    int64_t image_w = input_cv->shape()[width_index];
    uint8_t type_size = input_cv->type().SizeInBytes();
    // for random color
    std::normal_distribution<double> normal_distribution(0, 1);
    std::uniform_int_distribution<int> height_distribution_bound(0, image_h - box_height);
    std::uniform_int_distribution<int> width_distribution_bound(0, image_w - box_width);
    std::uniform_int_distribution<int> height_distribution_unbound(0, image_h + box_height);
    std::uniform_int_distribution<int> width_distribution_unbound(0, image_w + box_width);

    if (fill_colors.empty()) {
      fill_colors = std::vector<uint8_t>(num_channels, 0);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(fill_colors.size() == num_channels,
                                 "Number of fill colors (" + std::to_string(fill_colors.size()) +
                                   ") does not match the number of channels (" + std::to_string(num_channels) + ").");
    // core logic
    // update values based on random erasing or cutout
    for (int32_t i = 0; i < num_patches; i++) {
      // rows in cv mat refers to the height of the cropped box
      // we determine h_start and w_start using two different distributions as erasing is used by two different
      // image augmentations. The bounds are also different in each case.
      int32_t h_start = (bounded) ? height_distribution_bound(*rnd) : (height_distribution_unbound(*rnd) - box_height);
      int32_t w_start = (bounded) ? width_distribution_bound(*rnd) : (width_distribution_unbound(*rnd) - box_width);

      int64_t max_width = (w_start + box_width > image_w) ? image_w : w_start + box_width;
      int64_t max_height = (h_start + box_height > image_h) ? image_h : h_start + box_height;
      // check for starting range >= 0, here the start range is checked after for cut out, for random erasing
      // w_start and h_start will never be less than 0.
      h_start = (h_start < 0) ? 0 : h_start;
      w_start = (w_start < 0) ? 0 : w_start;

      if (is_hwc) {
        uchar *buffer = GetPtr(input_cv);
        int64_t num_bytes = type_size * num_channels * (max_width - w_start);
        for (int x = h_start; x < max_height; x++) {
          auto ret = memset_s(buffer + (x * image_w + w_start) * num_channels * type_size, num_bytes, 0, num_bytes);
          if (ret != EOK) {
            RETURN_STATUS_UNEXPECTED("CutOut: memset_s failed for HWC scenario.");
          }
        }
      } else {
        int64_t num_bytes = type_size * (max_width - w_start);
        for (uint64_t c = 0; c < num_channels; c++) {
          uchar *buffer = GetPtr(input_cv) + (type_size * c * image_h * image_w);
          for (int x = h_start; x < max_height; x++) {
            auto ret = memset_s(buffer + (x * image_w + w_start) * type_size, num_bytes, 0, num_bytes);
            if (ret != EOK) {
              RETURN_STATUS_UNEXPECTED("CutOut: memset_s failed for CHW scenario.");
            }
          }
        }
      }
    }

    *output = std::static_pointer_cast<Tensor>(input);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("CutOut: " + std::string(e.what()));
  }

  return Status::OK();
}

Status Erase(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t top, int32_t left,
             int32_t height, int32_t width, const std::vector<uint8_t> &value, bool inplace) {
  try {
    std::vector<dsize_t> size;
    RETURN_IF_NOT_OK(ImageSize(input, &size));
    int64_t image_h = size[kHeightIndex];
    int64_t image_w = size[kWidthIndex];
    if (height > image_h || width > image_w) {
      RETURN_STATUS_UNEXPECTED(
        "Erase: box size is too large for image erase, got box height: " + std::to_string(height) +
        "box weight: " + std::to_string(width) + ", and image height: " + std::to_string(image_h) +
        ", image width: " + std::to_string(image_w));
    }

    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();

    int32_t h_start = top;
    int32_t w_start = left;
    h_start = (h_start < 0) ? 0 : h_start;
    w_start = (w_start < 0) ? 0 : w_start;

    int32_t max_width = (w_start + width > image_w) ? image_w : w_start + width;
    int32_t max_height = (h_start + height > image_h) ? image_h : h_start + height;
    int32_t true_width = max_width - w_start;
    int32_t true_height = max_height - h_start;

    uint8_t fill_r = value[kRIndex];
    uint8_t fill_g = value[kGIndex];
    uint8_t fill_b = value[kBIndex];

    cv::Rect idx = cv::Rect(w_start, h_start, true_width, true_height);
    cv::Scalar fill_color = cv::Scalar(fill_r, fill_g, fill_b);
    (void)input_img(idx).setTo(fill_color);

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(input_img, input_cv->Rank(), &output_cv));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Erase: " + std::string(e.what()));
  }

  return Status::OK();
}

Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int32_t &pad_top,
           const int32_t &pad_bottom, const int32_t &pad_left, const int32_t &pad_right, const BorderType &border_types,
           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "Pad", {1, 2, 3, 4, 5, 6, 10, 11, 12}, {2, 3}, {1, 3}));

    // input image
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);

    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Pad: load image failed.");
    }

    // get the border type in openCV
    auto b_type = GetCVBorderType(border_types);
    // output image
    cv::Mat out_image;
    if (b_type == cv::BORDER_CONSTANT) {
      cv::Scalar fill_color = cv::Scalar(fill_r, fill_g, fill_b);
      cv::copyMakeBorder(input_cv->mat(), out_image, pad_top, pad_bottom, pad_left, pad_right, b_type, fill_color);
    } else {
      cv::copyMakeBorder(input_cv->mat(), out_image, pad_top, pad_bottom, pad_left, pad_right, b_type);
    }
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(out_image, input_cv->Rank(), &output_cv));
    // pad the dimension if shape information is only 2 dimensional, this is grayscale
    if (input_cv->Rank() == kDefaultImageRank && input_cv->shape()[kChannelIndexHWC] == kMinImageChannel &&
        output_cv->Rank() == kMinImageRank) {
      RETURN_IF_NOT_OK(output_cv->ExpandDim(kChannelIndexHWC));
    }
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Pad: " + std::string(e.what()));
  }
}

Status Perspective(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                   const std::vector<std::vector<int32_t>> &start_points,
                   const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "Perspective", {1, 2, 3, 4, 5, 6, 10, 11, 12}, {2, 3}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Perspective: load image failed.");
    }
    const int kListSize = 4;
    // Get Point
    cv::Point2f cv_src_point[kListSize];
    cv::Point2f cv_dst_point[kListSize];
    for (int i = 0; i < kListSize; i++) {
      cv_src_point[i] = cv::Point2f(start_points[i][0], start_points[i][1]);
      cv_dst_point[i] = cv::Point2f(end_points[i][0], end_points[i][1]);
    }

    // Perspective Operation
    std::shared_ptr<CVTensor> output_cv;
    cv::Mat M = cv::getPerspectiveTransform(cv_src_point, cv_dst_point, cv::DECOMP_LU);
    cv::Mat src_img = input_cv->mat();

    cv::Mat dst_img;
    cv::warpPerspective(src_img, dst_img, M, src_img.size(), GetCVInterpolationMode(interpolation));
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(dst_img, input_cv->Rank(), &output_cv));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Perspective: " + std::string(e.what()));
  }
}

Status RandomLighting(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rnd_r, float rnd_g,
                      float rnd_b) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();

    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED(
        "RandomLighting: Cannot convert from OpenCV type, unknown "
        "CV type. Currently supported data type: [int8, uint8, int16, uint16, "
        "int32, float16, float32, float64].");
    }

    if (input_cv->Rank() != kDefaultImageRank || input_cv->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED(
        "RandomLighting: input tensor is not in shape of <H,W,C> or channel is not 3, got rank: " +
        std::to_string(input_cv->Rank()) + ", and channel: " + std::to_string(input_cv->shape()[kChannelIndexHWC]));
    }
    auto input_type = input->type();
    CHECK_FAIL_RETURN_UNEXPECTED(input_type != DataType::DE_UINT32 && input_type != DataType::DE_UINT64 &&
                                   input_type != DataType::DE_INT64 && !input_type.IsString(),
                                 "RandomLighting: invalid tensor type of uint32, int64, uint64, string or bytes.");

    std::vector<std::vector<float>> eig = {{55.46 * -0.5675, 4.794 * 0.7192, 1.148 * 0.4009},
                                           {55.46 * -0.5808, 4.794 * -0.0045, 1.148 * -0.8140},
                                           {55.46 * -0.5836, 4.794 * -0.6948, 1.148 * 0.4203}};

    float pca_r = eig[0][0] * rnd_r + eig[0][1] * rnd_g + eig[0][2] * rnd_b;
    float pca_g = eig[1][0] * rnd_r + eig[1][1] * rnd_g + eig[1][2] * rnd_b;
    float pca_b = eig[2][0] * rnd_r + eig[2][1] * rnd_g + eig[2][2] * rnd_b;
    for (int row = 0; row < input_img.rows; row++) {
      for (int col = 0; col < input_img.cols; col++) {
        float r = static_cast<float>(input_img.at<cv::Vec3b>(row, col)[0]);
        float g = static_cast<float>(input_img.at<cv::Vec3b>(row, col)[1]);
        float b = static_cast<float>(input_img.at<cv::Vec3b>(row, col)[2]);
        input_img.at<cv::Vec3b>(row, col)[kRIndex] = cv::saturate_cast<uchar>(r + pca_r);
        input_img.at<cv::Vec3b>(row, col)[kGIndex] = cv::saturate_cast<uchar>(g + pca_g);
        input_img.at<cv::Vec3b>(row, col)[kBIndex] = cv::saturate_cast<uchar>(b + pca_b);
      }
    }

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(input_img, input_cv->Rank(), &output_cv));

    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RandomLighting: " + std::string(e.what()));
  }
}

Status RgbaToRgb(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "RgbaToRgb", {3, 5, 11}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    if (input_cv->shape().Size() != kDefaultImageChannel || input_cv->shape()[kChannelIndexHWC] != kMaxImageChannel) {
      std::string err_msg =
        "RgbaToRgb: rank of image is not: " + std::to_string(kDefaultImageChannel) +
        ", but got: " + std::to_string(input_cv->shape().Size()) +
        ", or channels of image should be 4, but got: " + std::to_string(input_cv->shape()[kChannelIndexHWC]);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    TensorShape out_shape = TensorShape({input_cv->shape()[0], input_cv->shape()[1], 3});
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_RGBA2RGB));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbaToRgb: " + std::string(e.what()));
  }
}

Status RgbaToBgr(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "RgbaToBgr", {3, 5, 11}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    if (input_cv->shape().Size() != kDefaultImageChannel || input_cv->shape()[kChannelIndexHWC] != kMaxImageChannel) {
      std::string err_msg =
        "RgbaToBgr: rank of image is not: " + std::to_string(kDefaultImageChannel) +
        ", but got: " + std::to_string(input_cv->shape().Size()) +
        ", or channels of image should be 4, but got: " + std::to_string(input_cv->shape()[kChannelIndexHWC]);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    TensorShape out_shape = TensorShape({input_cv->shape()[0], input_cv->shape()[1], 3});
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_RGBA2BGR));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbaToBgr: " + std::string(e.what()));
  }
}

Status RgbToBgr(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "RgbToBgr", {3, 4, 5, 6, 10, 11, 12}));
    auto input_type = input->type();
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] RgbToBgr: load image failed.");
    }
    if (input_cv->Rank() != kDefaultImageRank || input_cv->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("RgbToBgr: input tensor is not in shape of <H,W,C> or channel is not 3, got rank: " +
                               std::to_string(input_cv->Rank()) +
                               ", and channel: " + std::to_string(input_cv->shape()[2]));
    }

    cv::Mat image = input_cv->mat().clone();
    if (input_type == DataType::DE_FLOAT16 || input_type == DataType::DE_INT16 || input_type == DataType::DE_UINT16) {
      for (int i = 0; i < input_cv->mat().rows; ++i) {
        cv::Vec3s *p1 = input_cv->mat().ptr<cv::Vec3s>(i);
        cv::Vec3s *p2 = image.ptr<cv::Vec3s>(i);
        for (int j = 0; j < input_cv->mat().cols; ++j) {
          p2[j][kBIndex] = p1[j][kRIndex];
          p2[j][kGIndex] = p1[j][kGIndex];
          p2[j][kRIndex] = p1[j][kBIndex];
        }
      }
    } else if (input_type == DataType::DE_FLOAT32 || input_type == DataType::DE_INT32) {
      for (int i = 0; i < input_cv->mat().rows; ++i) {
        cv::Vec3f *p1 = input_cv->mat().ptr<cv::Vec3f>(i);
        cv::Vec3f *p2 = image.ptr<cv::Vec3f>(i);
        for (int j = 0; j < input_cv->mat().cols; ++j) {
          p2[j][kBIndex] = p1[j][kRIndex];
          p2[j][kGIndex] = p1[j][kGIndex];
          p2[j][kRIndex] = p1[j][kBIndex];
        }
      }
    } else if (input_type == DataType::DE_FLOAT64) {
      for (int i = 0; i < input_cv->mat().rows; ++i) {
        cv::Vec3d *p1 = input_cv->mat().ptr<cv::Vec3d>(i);
        cv::Vec3d *p2 = image.ptr<cv::Vec3d>(i);
        for (int j = 0; j < input_cv->mat().cols; ++j) {
          p2[j][kBIndex] = p1[j][kRIndex];
          p2[j][kGIndex] = p1[j][kGIndex];
          p2[j][kRIndex] = p1[j][kBIndex];
        }
      }
    } else {
      cv::cvtColor(input_cv->mat(), image, cv::COLOR_RGB2BGR);
    }

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(image, input_cv->Rank(), &output_cv));

    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbToBgr: " + std::string(e.what()));
  }
}

Status RgbToGray(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "RgbToGray", {3, 5, 11}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    if (input_cv->Rank() != kDefaultImageRank || input_cv->shape()[kChannelIndexHWC] != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("RgbToGray: image shape is not <H,W,C> or channel is not 3, got rank: " +
                               std::to_string(input_cv->Rank()) + ", and shape: " + input_cv->shape().ToString());
    }
    TensorShape out_shape = TensorShape({input_cv->shape()[0], input_cv->shape()[1]});
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(out_shape, input_cv->type(), &output_cv));
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_RGB2GRAY));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbToGray: " + std::string(e.what()));
  }
}

Status GetJpegImageInfo(const std::shared_ptr<Tensor> &input, int *img_width, int *img_height) {
  struct jpeg_decompress_struct cinfo {};
  struct JpegErrorManagerCustom jerr {};
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = JpegErrorExitCustom;
  try {
    jpeg_create_decompress(&cinfo);
    JpegSetSource(&cinfo, input->GetBuffer(), input->SizeInBytes());
    (void)jpeg_read_header(&cinfo, TRUE);
    jpeg_calc_output_dimensions(&cinfo);
    RETURN_IF_NOT_OK(CheckJpegExit(&cinfo));
  } catch (std::runtime_error &e) {
    jpeg_destroy_decompress(&cinfo);
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  *img_height = cinfo.output_height;
  *img_width = cinfo.output_width;
  jpeg_destroy_decompress(&cinfo);
  return Status::OK();
}

Status GetAffineMatrix(const std::shared_ptr<Tensor> &input, std::vector<float_t> *matrix, float_t degrees,
                       const std::vector<float_t> &translation, float_t scale, const std::vector<float_t> &shear) {
  CHECK_FAIL_RETURN_UNEXPECTED(translation.size() >= 2, "AffineOp::Compute translation_ size should >= 2");
  float_t translation_x = translation[0];
  float_t translation_y = translation[1];
  float_t degrees_tmp = 0.0;
  RETURN_IF_NOT_OK(DegreesToRadians(degrees, &degrees_tmp));
  CHECK_FAIL_RETURN_UNEXPECTED(shear.size() >= 2, "AffineOp::Compute shear_ size should >= 2");
  float_t shear_x = shear[0];
  float_t shear_y = shear[1];
  RETURN_IF_NOT_OK(DegreesToRadians(shear_x, &shear_x));
  RETURN_IF_NOT_OK(DegreesToRadians(-1 * shear_y, &shear_y));

  // Apply Affine Transformation
  //       T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
  //       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
  //       RSS is rotation with scale and shear matrix
  //       RSS(a, s, (sx, sy)) =
  //       = R(a) * S(s) * SHy(sy) * SHx(sx)
  //       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
  //         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
  //         [ 0                    , 0                                      , 1 ]
  //
  // where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
  // SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
  //          [0, 1      ]              [-tan(s), 1]
  //
  // Thus, the affine matrix is M = T * C * RSS * C^-1

  // image is hwc, rows = shape()[0]
  float_t cx = ((input->shape()[1] - 1) / 2.0);
  float_t cy = ((input->shape()[0] - 1) / 2.0);

  CHECK_FAIL_RETURN_UNEXPECTED(cos(shear_y) != 0.0, "AffineOp: cos(shear_y) should not be zero.");

  // Calculate RSS
  *matrix = std::vector<float_t>{
    static_cast<float>(scale * cos(degrees_tmp + shear_y) / cos(shear_y)),
    static_cast<float>(scale * (-1 * cos(degrees_tmp + shear_y) * tan(shear_x) / cos(shear_y) - sin(degrees_tmp))),
    0,
    static_cast<float>(scale * sin(degrees_tmp + shear_y) / cos(shear_y)),
    static_cast<float>(scale * (-1 * sin(degrees_tmp + shear_y) * tan(shear_x) / cos(shear_y) + cos(degrees_tmp))),
    0};
  // Compute T * C * RSS * C^-1
  // Compute T * C * RSS * C^-1
  (*matrix)[2] = (1 - (*matrix)[0]) * cx - (*matrix)[1] * cy + translation_x;
  (*matrix)[5] = (1 - (*matrix)[4]) * cy - (*matrix)[3] * cx + translation_y;
  return Status::OK();
}

Status Affine(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float_t degrees,
              const std::vector<float_t> &translation, float_t scale, const std::vector<float_t> &shear,
              InterpolationMode interpolation, const std::vector<uint8_t> &fill_value) {
  try {
    RETURN_IF_NOT_OK(ValidateImageRank("Affine", input->Rank()));
    dsize_t channel = 1;
    RETURN_IF_NOT_OK(ImageNumChannels(input, &channel));
    CHECK_FAIL_RETURN_UNEXPECTED(channel <= kMaxImageChannel || interpolation != InterpolationMode::kCubic,
                                 "Affine: interpolation can not be CUBIC when image channel is greater than 4.");
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Affine: load image failed.");
    }

    std::vector<float_t> matrix;
    RETURN_IF_NOT_OK(GetAffineMatrix(input, &matrix, degrees, translation, scale, shear));
    cv::Mat affine_mat(matrix);
    affine_mat = affine_mat.reshape(1, {2, 3});

    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv));
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    cv::warpAffine(input_cv->mat(), output_cv->mat(), affine_mat, input_cv->mat().size(),
                   GetCVInterpolationMode(interpolation), cv::BORDER_CONSTANT,
                   cv::Scalar(fill_value[kRIndex], fill_value[kGIndex], fill_value[kBIndex]));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Affine: " + std::string(e.what()));
  }
}

Status GaussianBlur(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t kernel_x,
                    int32_t kernel_y, float sigma_x, float sigma_y) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (input_cv->mat().data == nullptr) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] GaussianBlur: load image failed.");
    }
    cv::Mat output_cv_mat;
    cv::GaussianBlur(input_cv->mat(), output_cv_mat, cv::Size(kernel_x, kernel_y), static_cast<double>(sigma_x),
                     static_cast<double>(sigma_y));
    std::shared_ptr<CVTensor> output_cv;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(output_cv_mat, input_cv->Rank(), &output_cv));
    (*output) = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("GaussianBlur: " + std::string(e.what()));
  }
}

Status ComputePatchSize(const std::shared_ptr<CVTensor> &input_cv,
                        std::shared_ptr<std::pair<int32_t, int32_t>> *patch_size, int32_t num_height, int32_t num_width,
                        SliceMode slice_mode) {
  if (input_cv->mat().data == nullptr) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] SlicePatches: Tensor could not convert to CV Tensor.");
  }
  RETURN_IF_NOT_OK(ValidateImageRank("Affine", input_cv->Rank()));

  cv::Mat in_img = input_cv->mat();
  cv::Size s = in_img.size();
  if (num_height == 0 || num_height > s.height) {
    RETURN_STATUS_UNEXPECTED(
      "SlicePatches: The number of patches on height axis equals 0 or is greater than height, got number of patches:" +
      std::to_string(num_height));
  }
  if (num_width == 0 || num_width > s.width) {
    RETURN_STATUS_UNEXPECTED(
      "SlicePatches: The number of patches on width axis equals 0 or is greater than width, got number of patches:" +
      std::to_string(num_width));
  }
  int32_t patch_h = s.height / num_height;
  if (s.height % num_height != 0) {
    if (slice_mode == SliceMode::kPad) {
      patch_h += 1;  // patch_h * num_height - s.height
    }
  }
  int32_t patch_w = s.width / num_width;
  if (s.width % num_width != 0) {
    if (slice_mode == SliceMode::kPad) {
      patch_w += 1;  // patch_w * num_width - s.width
    }
  }
  (*patch_size)->first = patch_h;
  (*patch_size)->second = patch_w;
  return Status::OK();
}

Status SlicePatches(const std::shared_ptr<Tensor> &input, std::vector<std::shared_ptr<Tensor>> *output,
                    int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value) {
  if (num_height == DEFAULT_NUM_HEIGHT && num_width == DEFAULT_NUM_WIDTH) {
    (*output).push_back(input);
    return Status::OK();
  }

  auto patch_size = std::make_shared<std::pair<int32_t, int32_t>>(0, 0);
  int32_t patch_h = 0;
  int32_t patch_w = 0;

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  RETURN_IF_NOT_OK(ComputePatchSize(input_cv, &patch_size, num_height, num_width, slice_mode));
  std::tie(patch_h, patch_w) = *patch_size;

  cv::Mat in_img = input_cv->mat();
  cv::Size s = in_img.size();
  try {
    cv::Mat out_img;
    if (slice_mode == SliceMode::kPad) {  // padding on right and bottom directions
      auto padding_h = patch_h * num_height - s.height;
      auto padding_w = patch_w * num_width - s.width;
      out_img = cv::Mat(s.height + padding_h, s.width + padding_w, in_img.type(), cv::Scalar::all(fill_value));
      in_img.copyTo(out_img(cv::Rect(0, 0, s.width, s.height)));
    } else {
      out_img = in_img;
    }
    for (int i = 0; i < num_height; ++i) {
      for (int j = 0; j < num_width; ++j) {
        std::shared_ptr<CVTensor> patch_cv;
        cv::Rect rect(j * patch_w, i * patch_h, patch_w, patch_h);
        cv::Mat patch(out_img(rect));
        RETURN_IF_NOT_OK(CVTensor::CreateFromMat(patch, input_cv->Rank(), &patch_cv));
        (*output).push_back(std::static_pointer_cast<Tensor>(patch_cv));
      }
    }
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("SlicePatches: " + std::string(e.what()));
  }
}

Status Solarize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                const std::vector<float> &threshold) {
  try {
    RETURN_IF_NOT_OK(ValidateImage(input, "Solarize", {1, 2, 3, 4, 5, 6, 11, 12}, {2, 3}, {1, 3}));
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Solarize: load image failed.");
    }

    std::shared_ptr<CVTensor> mask_mat_tensor;
    std::shared_ptr<CVTensor> output_cv_tensor;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(input_img, input_cv->Rank(), &mask_mat_tensor));

    RETURN_IF_NOT_OK(CVTensor::CreateEmpty(input_cv->shape(), input_cv->type(), &output_cv_tensor));
    RETURN_UNEXPECTED_IF_NULL(mask_mat_tensor);
    RETURN_UNEXPECTED_IF_NULL(output_cv_tensor);

    auto threshold_min = threshold[0], threshold_max = threshold[1];

    if (threshold_min == threshold_max) {
      mask_mat_tensor->mat().setTo(0, ~(input_cv->mat() >= threshold_min));
    } else {
      mask_mat_tensor->mat().setTo(0, ~((input_cv->mat() >= threshold_min) & (input_cv->mat() <= threshold_max)));
    }

    // solarize desired portion
    const float max_size = 255.f;
    output_cv_tensor->mat() = cv::Scalar::all(max_size) - mask_mat_tensor->mat();
    input_cv->mat().copyTo(output_cv_tensor->mat(), input_cv->mat() < threshold_min);
    if (threshold_min < threshold_max) {
      input_cv->mat().copyTo(output_cv_tensor->mat(), input_cv->mat() > threshold_max);
    }

    *output = std::static_pointer_cast<Tensor>(output_cv_tensor);
  }

  catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Solarize: " + std::string(e.what()));
  }
  return Status::OK();
}

Status ToTensor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const DataType &data_type) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] ToTensor: load image failed.");
    }
    if (input_cv->Rank() == kMinImageRank) {
      // If input tensor is 2D, we assume we have HW dimensions
      RETURN_IF_NOT_OK(input_cv->ExpandDim(kMinImageRank));
    }
    CHECK_FAIL_RETURN_UNEXPECTED(
      input_cv->shape().Size() > kChannelIndexHWC,
      "ToTensor: rank of input data should be greater than: " + std::to_string(kChannelIndexHWC) +
        ", but got:" + std::to_string(input_cv->shape().Size()));
    int num_channels = static_cast<int>(input_cv->shape()[kChannelIndexHWC]);
    if (input_cv->shape().Size() != kDefaultImageRank) {
      RETURN_STATUS_UNEXPECTED("ToTensor: image shape should be <H,W,C>, but got rank: " +
                               std::to_string(input_cv->shape().Size()));
    }

    int height = static_cast<int>(input_cv->shape()[0]);
    int width = static_cast<int>(input_cv->shape()[1]);

    // OpenCv has a bug in extractChannel when the type is float16.
    // To avoid the segfault, we cast to float32 first.
    if (input_cv->type() == DataType(DataType::DE_FLOAT16)) {
      RETURN_IF_NOT_OK(TypeCast(input_cv, output, DataType(DataType::DE_FLOAT32)));
      input_cv = CVTensor::AsCVTensor(*output);
    }

    std::shared_ptr<CVTensor> output_cv;
    // Reshape from HCW to CHW
    RETURN_IF_NOT_OK(
      CVTensor::CreateEmpty(TensorShape{num_channels, height, width}, DataType(DataType::DE_FLOAT32), &output_cv));
    // Rescale tensor by dividing by 255
    const float kMaxBitValueinFloat = static_cast<float>(kMaxBitValue);
    for (int i = 0; i < num_channels; ++i) {
      cv::Mat mat_t;
      cv::extractChannel(input_cv->mat(), mat_t, i);
      cv::Mat mat;
      RETURN_IF_NOT_OK(output_cv->MatAtIndex({i}, &mat));
      mat_t.convertTo(mat, CV_32F, 1 / kMaxBitValueinFloat, 0);
    }

    // Process tensor output according to desired output data type
    if (data_type != DataType(DataType::DE_FLOAT32)) {
      RETURN_IF_NOT_OK(TypeCast(output_cv, output, data_type));
    } else {
      *output = std::move(output_cv);
    }
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("ToTensor: " + std::string(e.what()));
  }
}

// round half to even
float Round(float value) {
  float kHalf = 0.5;
  const int32_t kEven = 2;
  float rnd = round(value);
  float rnd_l = floor(value);
  float rnd_h = ceil(value);
  if (value - rnd_l == kHalf) {
    if (common::IsDoubleEqual(fmod(rnd, kEven), 0.0)) {
      return rnd;
    } else if (value > 0) {
      return rnd_l;
    } else {
      return rnd_h;
    }
  }
  return rnd;
}

std::vector<float> Linspace(float start, float end, int n, float scale, float offset, bool round) {
  std::vector<float> linear(n);
  float step = (n == 1) ? 0 : ((end - start) / (n - 1));
  for (size_t i = 0; i < linear.size(); ++i) {
    linear[i] = (start + i * step) * scale + offset;
    if (round) {
      linear[i] = Round(linear[i]);
    }
  }
  return linear;
}

Status ApplyAugment(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::string &op_name,
                    float magnitude, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value) {
  if (op_name == "ShearX") {
    float_t shear = magnitude * 180 / CV_PI;
    AffineOp affine(0.0, {0, 0}, 1.0, {shear, 0.0}, interpolation, fill_value);
    RETURN_IF_NOT_OK(affine.Compute(input, output));
  } else if (op_name == "ShearY") {
    float_t shear = magnitude * 180 / CV_PI;
    AffineOp affine(0.0, {0, 0}, 1.0, {0.0, shear}, interpolation, fill_value);
    RETURN_IF_NOT_OK(affine.Compute(input, output));
  } else if (op_name == "TranslateX") {
    float_t translate = static_cast<int>(magnitude);
    AffineOp affine(0.0, {translate, 0}, 1.0, {0.0, 0.0}, interpolation, fill_value);
    RETURN_IF_NOT_OK(affine.Compute(input, output));
  } else if (op_name == "TranslateY") {
    float_t translate = static_cast<int>(magnitude);
    AffineOp affine(0.0, {0, translate}, 1.0, {0.0, 0.0}, interpolation, fill_value);
    RETURN_IF_NOT_OK(affine.Compute(input, output));
  } else if (op_name == "Rotate") {
    RETURN_IF_NOT_OK(Rotate(input, output, {}, magnitude, interpolation, false, fill_value[kRIndex],
                            fill_value[kBIndex], fill_value[kGIndex]));
  } else if (op_name == "Brightness") {
    RETURN_IF_NOT_OK(AdjustBrightness(input, output, 1 + magnitude));
  } else if (op_name == "Color") {
    RETURN_IF_NOT_OK(AdjustSaturation(input, output, 1 + magnitude));
  } else if (op_name == "Contrast") {
    RETURN_IF_NOT_OK(AdjustContrast(input, output, 1 + magnitude));
  } else if (op_name == "Sharpness") {
    SharpnessOp sharpness(1 + magnitude);
    RETURN_IF_NOT_OK(sharpness.Compute(input, output));
  } else if (op_name == "Posterize") {
    PosterizeOp posterize(static_cast<int>(magnitude));
    RETURN_IF_NOT_OK(posterize.Compute(input, output));
  } else if (op_name == "Solarize") {
    RETURN_IF_NOT_OK(Solarize(input, output, {magnitude, magnitude}));
  } else if (op_name == "AutoContrast") {
    RETURN_IF_NOT_OK(AutoContrast(input, output, 0.0, {}));
  } else if (op_name == "Equalize") {
    RETURN_IF_NOT_OK(Equalize(input, output));
  } else if (op_name == "Identity") {
    *output = std::static_pointer_cast<Tensor>(input);
  } else if (op_name == "Invert") {
    InvertOp invert;
    RETURN_IF_NOT_OK(invert.Compute(input, output));
  } else {
    RETURN_STATUS_UNEXPECTED("ApplyAugment: the provided operator " + op_name + " is not supported.");
  }
  return Status::OK();
}

Status EncodeJpeg(const std::shared_ptr<Tensor> &image, std::shared_ptr<Tensor> *output, int quality) {
  RETURN_UNEXPECTED_IF_NULL(output);

  std::string err_msg;
  if (image->type() != DataType::DE_UINT8) {
    err_msg = "EncodeJpeg: The type of the image data should be UINT8, but got " + image->type().ToString() + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  TensorShape shape = image->shape();
  int rank = shape.Rank();
  if (rank < kMinImageRank || rank > kDefaultImageRank) {
    err_msg = "EncodeJpeg: The image has invalid dimensions. It should have two or three dimensions, but got ";
    err_msg += std::to_string(rank) + " dimensions.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int channels;
  if (rank == kDefaultImageRank) {
    channels = shape[kMinImageRank];
    if (channels != kMinImageChannel && channels != kDefaultImageChannel) {
      err_msg = "EncodeJpeg: The image has invalid channels. It should have 1 or 3 channels, but got ";
      err_msg += std::to_string(channels) + " channels.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  } else {
    channels = 1;
  }

  if (quality < kMinJpegQuality || quality > kMaxJpegQuality) {
    err_msg = "EncodeJpeg: Invalid quality " + std::to_string(quality) + ", should be in range of [" +
              std::to_string(kMinJpegQuality) + ", " + std::to_string(kMaxJpegQuality) + "].";

    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,  quality, cv::IMWRITE_JPEG_PROGRESSIVE,  0,
                             cv::IMWRITE_JPEG_OPTIMIZE, 0,       cv::IMWRITE_JPEG_RST_INTERVAL, 0};

  std::vector<unsigned char> buffer;
  cv::Mat image_matrix;

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(image);
  image_matrix = input_cv->mat();
  if (!image_matrix.data) {
    RETURN_STATUS_UNEXPECTED("EncodeJpeg: Load the image tensor failed.");
  }

  if (channels == kMinImageChannel) {
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".JPEG", image_matrix, buffer, params),
                                 "EncodeJpeg: Failed to encode image.");
  } else {
    cv::Mat image_bgr;
    cv::cvtColor(image_matrix, image_bgr, cv::COLOR_RGB2BGR);
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".JPEG", image_bgr, buffer, params),
                                 "EncodeJpeg: Failed to encode image.");
  }

  TensorShape tensor_shape = TensorShape({(long int)buffer.size()});
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(tensor_shape, DataType(DataType::DE_UINT8), buffer.data(), output));

  return Status::OK();
}

Status EncodePng(const std::shared_ptr<Tensor> &image, std::shared_ptr<Tensor> *output, int compression_level) {
  RETURN_UNEXPECTED_IF_NULL(output);

  std::string err_msg;
  if (image->type() != DataType::DE_UINT8) {
    err_msg = "EncodePng: The type of the image data should be UINT8, but got " + image->type().ToString() + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  TensorShape shape = image->shape();
  int rank = shape.Rank();
  if (rank < kMinImageRank || rank > kDefaultImageRank) {
    err_msg = "EncodePng: The image has invalid dimensions. It should have two or three dimensions, but got ";
    err_msg += std::to_string(rank) + " dimensions.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int channels;
  if (rank == kDefaultImageRank) {
    channels = shape[kMinImageRank];
    if (channels != kMinImageChannel && channels != kDefaultImageChannel) {
      err_msg = "EncodePng: The image has invalid channels. It should have 1 or 3 channels, but got ";
      err_msg += std::to_string(channels) + " channels.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  } else {
    channels = 1;
  }

  if (compression_level < kMinPngCompression || compression_level > kMaxPngCompression) {
    err_msg = "EncodePng: Invalid compression_level " + std::to_string(compression_level) +
              ", should be in range of [" + std::to_string(kMinPngCompression) + ", " +
              std::to_string(kMaxPngCompression) + "].";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, compression_level, cv::IMWRITE_PNG_STRATEGY,
                             cv::IMWRITE_PNG_STRATEGY_RLE};
  std::vector<unsigned char> buffer;
  cv::Mat image_matrix;

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(image);
  image_matrix = input_cv->mat();
  if (!image_matrix.data) {
    RETURN_STATUS_UNEXPECTED("EncodePng: Load the image tensor failed.");
  }

  if (channels == kMinImageChannel) {
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".PNG", image_matrix, buffer, params),
                                 "EncodePng: Failed to encode image.");
  } else {
    cv::Mat image_bgr;
    cv::cvtColor(image_matrix, image_bgr, cv::COLOR_RGB2BGR);
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".PNG", image_bgr, buffer, params), "EncodePng: Failed to encode image.");
  }

  TensorShape tensor_shape = TensorShape({(long int)buffer.size()});
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(tensor_shape, DataType(DataType::DE_UINT8), buffer.data(), output));

  return Status::OK();
}

Status ReadFile(const std::string &filename, std::shared_ptr<Tensor> *output) {
  RETURN_UNEXPECTED_IF_NULL(output);

  auto realpath = FileUtils::GetRealPath(filename.c_str());
  if (!realpath.has_value()) {
    RETURN_STATUS_UNEXPECTED("ReadFile: Invalid file path, " + filename + " does not exist.");
  }
  if (!Path(realpath.value()).IsFile()) {
    RETURN_STATUS_UNEXPECTED("ReadFile: Invalid file path, " + filename + " is not a regular file.");
  }

  RETURN_IF_NOT_OK(Tensor::CreateFromFile(realpath.value(), output));
  return Status::OK();
}

Status ReadImage(const std::string &filename, std::shared_ptr<Tensor> *output, ImageReadMode mode) {
  RETURN_UNEXPECTED_IF_NULL(output);

  auto realpath = FileUtils::GetRealPath(filename.c_str());
  if (!realpath.has_value()) {
    std::string err_msg = "ReadImage: Invalid file path, " + filename + " does not exist.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (!Path(realpath.value()).IsFile()) {
    RETURN_STATUS_UNEXPECTED("ReadImage: Invalid file path, " + filename + " is not a regular file.");
  }

  cv::Mat image;
  int cv_mode = static_cast<int>(mode) - 1;
  image = cv::imread(realpath.value(), cv_mode);
  if (image.data == nullptr) {
    RETURN_STATUS_UNEXPECTED("ReadImage: Failed to read file " + filename);
  }

  std::shared_ptr<CVTensor> output_cv;
  if (mode == ImageReadMode::kCOLOR || image.channels() > 1) {
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGRA2RGB);
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(image_rgb, kDefaultImageRank, &output_cv));
  } else {
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(image, kDefaultImageRank, &output_cv));
  }
  *output = std::static_pointer_cast<Tensor>(output_cv);

  return Status::OK();
}

Status WriteFile(const std::string &filename, const std::shared_ptr<Tensor> &data) {
  std::string err_msg;

  if (data->type() != DataType::DE_UINT8) {
    err_msg = "WriteFile: The type of the elements of data should be UINT8, but got " + data->type().ToString() + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  long int data_size = data->Size();
  const char *data_buffer;
  if (data_size >= kDeMaxDim || data_size < 0) {
    err_msg = "WriteFile: Invalid data->Size() , should be >= 0 && < " + std::to_string(kDeMaxDim);
    err_msg += " , but got " + std::to_string(data_size) + " for " + filename;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (data_size > 0) {
    data_buffer = (const char *)data->GetBuffer();
    if (data_buffer == nullptr) {
      err_msg = "WriteFile: Invalid data->GetBufferSize() , should not be nullptr.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    TensorShape shape = data->shape();
    int rank = shape.Rank();
    if (rank != kMinImageChannel) {
      err_msg = "WriteFile: The data has invalid dimensions. It should have only one dimension, but got ";
      err_msg += std::to_string(rank) + " dimensions.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  Path file(filename);
  if (!file.Exists()) {
    int file_descriptor;
    RETURN_IF_NOT_OK(file.CreateFile(&file_descriptor));
    RETURN_IF_NOT_OK(file.CloseFile(file_descriptor));
  }
  auto realpath = FileUtils::GetRealPath(filename.c_str());
  if (!realpath.has_value()) {
    RETURN_STATUS_UNEXPECTED("WriteFile: Invalid file path, " + filename + " failed to get the real path.");
  }
  if (!Path(realpath.value()).IsFile()) {
    RETURN_STATUS_UNEXPECTED("WriteFile: Invalid file path, " + filename + " is not a regular file.");
  }

  std::ofstream fs(realpath.value().c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
  CHECK_FAIL_RETURN_UNEXPECTED(!fs.fail(), "WriteFile: Failed to open the file: " + filename + " for writing.");

  if (data_size > 0) {
    fs.write(data_buffer, data_size);
    if (fs.fail()) {
      err_msg = "WriteFile: Failed to write the file " + filename;
      fs.close();
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  fs.close();
  return Status::OK();
}

Status WriteJpeg(const std::string &filename, const std::shared_ptr<Tensor> &image, int quality) {
  std::string err_msg;

  if (image->type() != DataType::DE_UINT8) {
    err_msg = "WriteJpeg: The type of the elements of image should be UINT8, but got " + image->type().ToString() + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  TensorShape shape = image->shape();
  int rank = shape.Rank();
  if (rank < kMinImageRank || rank > kDefaultImageRank) {
    err_msg = "WriteJpeg: The image has invalid dimensions. It should have two or three dimensions, but got ";
    err_msg += std::to_string(rank) + " dimensions.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int channels;
  if (rank == kDefaultImageRank) {
    channels = shape[kMinImageRank];
    if (channels != kMinImageChannel && channels != kDefaultImageChannel) {
      err_msg = "WriteJpeg: The image has invalid channels. It should have 1 or 3 channels, but got ";
      err_msg += std::to_string(channels) + " channels.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  } else {
    channels = 1;
  }

  if (quality < kMinJpegQuality || quality > kMaxJpegQuality) {
    err_msg = "WriteJpeg: Invalid quality " + std::to_string(quality) + ", should be in range of [" +
              std::to_string(kMinJpegQuality) + ", " + std::to_string(kMaxJpegQuality) + "].";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,  quality, cv::IMWRITE_JPEG_PROGRESSIVE,  0,
                             cv::IMWRITE_JPEG_OPTIMIZE, 0,       cv::IMWRITE_JPEG_RST_INTERVAL, 0};

  std::vector<unsigned char> buffer;
  cv::Mat image_matrix;

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(image);
  image_matrix = input_cv->mat();
  if (!image_matrix.data) {
    RETURN_STATUS_UNEXPECTED("WriteJpeg: Load the image tensor failed.");
  }

  if (channels == kMinImageChannel) {
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".JPEG", image_matrix, buffer, params),
                                 "WriteJpeg: Failed to encode image.");
  } else {
    cv::Mat image_bgr;
    cv::cvtColor(image_matrix, image_bgr, cv::COLOR_RGB2BGR);
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".JPEG", image_bgr, buffer, params),
                                 "WriteJpeg: Failed to encode image.");
  }

  Path file(filename);
  if (!file.Exists()) {
    int file_descriptor;
    RETURN_IF_NOT_OK(file.CreateFile(&file_descriptor));
    RETURN_IF_NOT_OK(file.CloseFile(file_descriptor));
  }
  auto realpath = FileUtils::GetRealPath(filename.c_str());
  if (!realpath.has_value()) {
    RETURN_STATUS_UNEXPECTED("WriteJpeg: Invalid file path, " + filename + " failed to get the real path.");
  }
  if (!Path(realpath.value()).IsFile()) {
    RETURN_STATUS_UNEXPECTED("WriteJpeg: Invalid file path, " + filename + " is not a regular file.");
  }

  std::ofstream fs(realpath.value().c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
  CHECK_FAIL_RETURN_UNEXPECTED(!fs.fail(), "WriteJpeg: Failed to open the file " + filename + " for writing.");

  fs.write((const char *)buffer.data(), (long int)buffer.size());
  if (fs.fail()) {
    fs.close();
    RETURN_STATUS_UNEXPECTED("WriteJpeg: Failed to write the file " + filename);
  }
  fs.close();
  return Status::OK();
}

Status WritePng(const std::string &filename, const std::shared_ptr<Tensor> &image, int compression_level) {
  std::string err_msg;

  if (image->type() != DataType::DE_UINT8) {
    err_msg = "WritePng: The type of the elements of image should be UINT8, but got " + image->type().ToString() + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  TensorShape shape = image->shape();
  int rank = shape.Rank();
  if (rank < kMinImageRank || rank > kDefaultImageRank) {
    err_msg = "WritePng: The image has invalid dimensions. It should have two or three dimensions, but got ";
    err_msg += std::to_string(rank) + " dimensions.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int channels;
  if (rank == kDefaultImageRank) {
    channels = shape[kMinImageRank];
    if (channels != kMinImageChannel && channels != kDefaultImageChannel) {
      err_msg = "WritePng: The image has invalid channels. It should have 1 or 3 channels, but got ";
      err_msg += std::to_string(channels) + " channels.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  } else {
    channels = 1;
  }

  if (compression_level < kMinPngCompression || compression_level > kMaxPngCompression) {
    err_msg = "WritePng: Invalid compression_level " + std::to_string(compression_level) + ", should be in range of [" +
              std::to_string(kMinPngCompression) + ", " + std::to_string(kMaxPngCompression) + "].";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, compression_level, cv::IMWRITE_PNG_STRATEGY,
                             cv::IMWRITE_PNG_STRATEGY_RLE};
  std::vector<unsigned char> buffer;
  cv::Mat image_matrix;

  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(image);
  image_matrix = input_cv->mat();
  if (!image_matrix.data) {
    RETURN_STATUS_UNEXPECTED("WritePng: Load the image tensor failed.");
  }

  if (channels == kMinImageChannel) {
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".PNG", image_matrix, buffer, params),
                                 "WritePng: Failed to encode image.");
  } else {
    cv::Mat image_bgr;
    cv::cvtColor(image_matrix, image_bgr, cv::COLOR_RGB2BGR);
    CHECK_FAIL_RETURN_UNEXPECTED(cv::imencode(".PNG", image_bgr, buffer, params), "WritePng: Failed to encode image.");
  }

  Path file(filename);
  if (!file.Exists()) {
    int file_descriptor;
    RETURN_IF_NOT_OK(file.CreateFile(&file_descriptor));
    RETURN_IF_NOT_OK(file.CloseFile(file_descriptor));
  }
  auto realpath = FileUtils::GetRealPath(filename.c_str());
  if (!realpath.has_value()) {
    RETURN_STATUS_UNEXPECTED("WritePng: Invalid file path, " + filename + " failed to get the real path.");
  }
  struct stat sb;
  stat(realpath.value().c_str(), &sb);
  if (S_ISREG(sb.st_mode) == 0) {
    RETURN_STATUS_UNEXPECTED("WritePng: Invalid file path, " + filename + " is not a regular file.");
  }

  std::ofstream fs(realpath.value().c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
  CHECK_FAIL_RETURN_UNEXPECTED(!fs.fail(), "WritePng: Failed to open the file " + filename + " for writing.");

  fs.write((const char *)buffer.data(), (long int)buffer.size());
  if (fs.fail()) {
    fs.close();
    RETURN_STATUS_UNEXPECTED("WritePng: Failed to write the file " + filename);
  }
  fs.close();
  return Status::OK();
}

// support list
const unsigned char kBmpMagic[] = "\x42\x4D";
constexpr dsize_t kBmpMagicLen = 2;
const unsigned char kTiffMagic1[] = "\x4D\x4D";
const unsigned char kTiffMagic2[] = "\x49\x49";
constexpr dsize_t kTiffMagicLen = 2;

Status DumpImageAndAppendStatus(const std::shared_ptr<Tensor> &image, const Status &status) {
  Status local_status = status;
  std::string file_name = "./abnormal_image.";
  std::string file_suffix = "";
  std::string error_info = local_status.GetErrDescription();
  if (image->SizeInBytes() == 0) {
    return local_status;
  }

  if (memcmp(image->GetBuffer(), kJpegMagic, kJpegMagicLen) == 0) {  // support
    file_suffix = "jpg";
  } else if (memcmp(image->GetBuffer(), kPngMagic, kPngMagicLen) == 0) {  // support
    file_suffix = "png";
  } else if (memcmp(image->GetBuffer(), kBmpMagic, kBmpMagicLen) == 0) {  // support
    file_suffix = "bmp";
  } else if (memcmp(image->GetBuffer(), kTiffMagic1, kTiffMagicLen) == 0 ||  // support
             memcmp(image->GetBuffer(), kTiffMagic2, kTiffMagicLen) == 0) {
    file_suffix = "tif";
  } else {
    file_suffix = "exception";
    error_info += " Unknown image type.";
  }

  auto ret = WriteFile(file_name + file_suffix, image);
  if (ret == Status::OK()) {
    error_info += " Dump the abnormal image to [" + (file_name + file_suffix) +
                  "]. You can check this image first through the image viewer. If you find that " +
                  "the image is abnormal, delete it from the dataset and re-run.";
  }
  local_status.SetErrDescription(error_info);
  return local_status;
}

// unsupported list
const unsigned char kGifMagic[] = "\x47\x49\x46";
constexpr dsize_t kGifMagicLen = 3;
const unsigned char kWebpMagic[] = "\x00\x57\x45\x42";
constexpr dsize_t kWebpMagicLen = 4;

Status CheckUnsupportedImage(const std::shared_ptr<Tensor> &image) {
  bool unsupport_flag = false;

  std::string file_name = "./unsupported_image.";
  std::string file_suffix = "";
  if (image->SizeInBytes() == 0) {
    RETURN_STATUS_UNEXPECTED("Image file size is 0.");
  }

  if (memcmp(image->GetBuffer(), kGifMagic, kGifMagicLen) == 0) {  // unsupported
    file_suffix = "gif";
    unsupport_flag = true;
  } else if (memcmp(image->GetBuffer() + 7, kWebpMagic, kWebpMagicLen) == 0) {  // unsupported: skip the 7 bytes
    file_suffix = "webp";
    unsupport_flag = true;
  }

  if (unsupport_flag) {
    auto ret = WriteFile(file_name + file_suffix, image);
    if (ret == Status::OK()) {
      RETURN_STATUS_UNEXPECTED("Unsupported image type [" + file_suffix + "] and dump the image to [" +
                               (file_name + file_suffix) + "]. Please delete it from the dataset and re-run.");
    } else {
      ret.SetErrDescription("Unsupported image type [" + file_suffix + "], but dump the image failed. " +
                            "Error info: " + ret.GetErrDescription());
      return ret;
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
