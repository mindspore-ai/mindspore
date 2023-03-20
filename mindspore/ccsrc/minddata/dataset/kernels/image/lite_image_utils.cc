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
#include "minddata/dataset/kernels/image/lite_image_utils.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/image/lite_cv/image_process.h"
#include "minddata/dataset/kernels/image/lite_cv/lite_mat.h"
#include "minddata/dataset/kernels/image/math_utils.h"
#include "minddata/dataset/util/random.h"
#if defined(ENABLE_CLOUD_FUSION_INFERENCE)
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/resize_cubic_op.h"
#endif

constexpr int64_t hw_shape = 2;
constexpr int64_t hwc_rank = 3;

#define MAX_INT_PRECISION 16777216  // float int precision is 16777216
namespace mindspore {
namespace dataset {
#if defined(ENABLE_CLOUD_FUSION_INFERENCE)
bool IsNonEmptyPNG(const std::shared_ptr<Tensor> &input) {
  const unsigned char kPngMagic[] = "\x89\x50\x4E\x47";
  constexpr dsize_t kPngMagicLen = 4;
  return input->SizeInBytes() > kPngMagicLen && memcmp(input->GetBuffer(), kPngMagic, kPngMagicLen) == 0;
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
#endif

bool IsNonEmptyJPEG(const std::shared_ptr<Tensor> &input) {
  const unsigned char *kJpegMagic = (unsigned char *)"\xFF\xD8\xFF";
  constexpr size_t kJpegMagicLen = 3;
  return input->SizeInBytes() > kJpegMagicLen && memcmp(input->GetBuffer(), kJpegMagic, kJpegMagicLen) == 0;
}

static void JpegInitSource(j_decompress_ptr cinfo) {}

static boolean JpegFillInputBuffer(j_decompress_ptr cinfo) {
  if (cinfo->src->bytes_in_buffer == 0) {
    // Under ARM platform raise runtime_error may cause core problem,
    // so we catch runtime_error and just return FALSE.
    try {
      ERREXIT(cinfo, JERR_INPUT_EMPTY);
    } catch (const std::exception &e) {
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
#if defined(_WIN32) || defined(_WIN64) || defined(ENABLE_ARM32)
  // the following line skips CI because it uses underlying C type
  cinfo->src->skip_input_data = reinterpret_cast<void (*)(j_decompress_ptr, long)>(JpegSkipInputData);  // NOLINT.
#else
  cinfo->src->skip_input_data = JpegSkipInputData;
#endif
  cinfo->src->resync_to_restart = jpeg_resync_to_restart;
  cinfo->src->term_source = JpegTermSource;
  cinfo->src->bytes_in_buffer = datasize;
  cinfo->src->next_input_byte = static_cast<const JOCTET *>(data);
}

static Status JpegReadScanlines(jpeg_decompress_struct *const cinfo, int max_scanlines_to_read, JSAMPLE *buffer,
                                int buffer_size, int crop_w, int crop_w_aligned, int offset, int stride) {
  // scanlines will be read to this buffer first, must have the number
  // of components equal to the number of components in the image
  int64_t scanline_size = crop_w_aligned * cinfo->output_components;
  std::vector<JSAMPLE> scanline(scanline_size);
  JSAMPLE *scanline_ptr = &scanline[0];
  while (cinfo->output_scanline < static_cast<unsigned int>(max_scanlines_to_read)) {
    int num_lines_read = 0;
    try {
      num_lines_read = jpeg_read_scanlines(cinfo, &scanline_ptr, 1);
    } catch (const std::exception &e) {
      RETURN_STATUS_UNEXPECTED("Decode: jpeg_read_scanlines error.");
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
          r = (k * c) / kMaxPixelValue;
          g = (k * m) / kMaxPixelValue;
          b = (k * y) / kMaxPixelValue;
        } else {
          r = (kMaxPixelValue - c) * (kMaxPixelValue - k) / kMaxPixelValue;
          g = (kMaxPixelValue - m) * (kMaxPixelValue - k) / kMaxPixelValue;
          b = (kMaxPixelValue - y) * (kMaxPixelValue - k) / kMaxPixelValue;
        }
        constexpr int buffer_rgb_val_size = 3;
        constexpr int channel_red = 0;
        constexpr int channel_green = 1;
        constexpr int channel_blue = 2;
        buffer[buffer_rgb_val_size * i + channel_red] = r;
        buffer[buffer_rgb_val_size * i + channel_green] = g;
        buffer[buffer_rgb_val_size * i + channel_blue] = b;
      }
    } else if (num_lines_read > 0) {
      auto copy_status = memcpy_s(buffer, buffer_size, scanline_ptr + offset, stride);
      if (copy_status != 0) {
        jpeg_destroy_decompress(cinfo);
        RETURN_STATUS_UNEXPECTED("Decode: memcpy_s failed");
      }
    } else {
      jpeg_destroy_decompress(cinfo);
      std::string err_msg = "Decode: failed to decompress image.";
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
      std::string err_msg = "Decode: failed to decompress image.";
      RETURN_STATUS_UNEXPECTED(err_msg);
  }
}

void JpegErrorExitCustom(j_common_ptr cinfo) {
  char jpeg_last_error_msg[JMSG_LENGTH_MAX];
  (*(cinfo->err->format_message))(cinfo, jpeg_last_error_msg);
  throw std::runtime_error(jpeg_last_error_msg);
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
  } catch (const std::exception &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - crop_w) > crop_x, "invalid crop width");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - crop_h) > crop_y, "invalid crop height");
  if (crop_x == 0 && crop_y == 0 && crop_w == 0 && crop_h == 0) {
    crop_w = cinfo.output_width;
    crop_h = cinfo.output_height;
  } else if (crop_w == 0 || static_cast<unsigned int>(crop_w + crop_x) > cinfo.output_width || crop_h == 0 ||
             static_cast<unsigned int>(crop_h + crop_y) > cinfo.output_height) {
    return DestroyDecompressAndReturnError("Decode: invalid crop size");
  }
  const int mcu_size = cinfo.min_DCT_scaled_size;
  CHECK_FAIL_RETURN_UNEXPECTED(mcu_size != 0, "Invalid data.");
  unsigned int crop_x_aligned = (crop_x / mcu_size) * mcu_size;
  unsigned int crop_w_aligned = crop_w + crop_x - crop_x_aligned;
  try {
    (void)jpeg_start_decompress(&cinfo);
    jpeg_crop_scanline(&cinfo, &crop_x_aligned, &crop_w_aligned);
  } catch (const std::exception &e) {
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
  // stride refers to output tensor, which has 3 components at most
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - skipped_scanlines) > crop_h,
                               "Invalid crop height.");
  const int max_scanlines_to_read = skipped_scanlines + crop_h;
  // stride refers to output tensor, which has 3 components at most
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() / crop_w) > kOutNumComponents,
                               "Invalid crop width.");
  const int stride = crop_w * kOutNumComponents;
  // offset is calculated for scanlines read from the image, therefore
  // has the same number of components as the image
  const int offset = (crop_x - crop_x_aligned) * cinfo.output_components;
  RETURN_IF_NOT_OK(
    JpegReadScanlines(&cinfo, max_scanlines_to_read, buffer, buffer_size, crop_w, crop_w_aligned, offset, stride));
  *output = output_tensor;
  jpeg_destroy_decompress(&cinfo);
  return Status::OK();
}

static LDataType GetLiteCVDataType(const DataType &data_type) {
  if (data_type == DataType::DE_UINT8) {
    return LDataType::UINT8;
  } else if (data_type == DataType::DE_FLOAT32) {
    return LDataType::FLOAT32;
  } else {
    return LDataType::UNKNOWN;
  }
}

#if defined(ENABLE_CLOUD_FUSION_INFERENCE)
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
    const dsize_t rank_num = 3;
    RETURN_IF_NOT_OK(CVTensor::CreateFromMat(img_mat, rank_num, &output_cv));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Decode: " + std::string(e.what()));
  }
}
#endif

Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  if (IsNonEmptyJPEG(input)) {
    return JpegCropAndDecode(input, output);
  } else {
#if defined(ENABLE_CLOUD_FUSION_INFERENCE)
    return DecodeCv(input, output);
#else
    RETURN_STATUS_UNEXPECTED("Decode: Decode only supports jpeg for android");
#endif
  }
}

Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h) {
  if (input->Rank() != 3 && input->Rank() != 2) {
    RETURN_STATUS_UNEXPECTED("Crop: image shape is not <H,W,C> or <H,W>");
  }

  if (input->type() != DataType::DE_FLOAT32 && input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("Crop: image datatype is not float32 or uint8");
  }

  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - y) > h, "Invalid crop height.");
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - x) > w, "Invalid crop width.");
  // account for integer overflow
  if (y < 0 || (y + h) > input->shape()[0] || (y + h) < 0) {
    RETURN_STATUS_UNEXPECTED(
      "Crop: invalid y coordinate value for crop"
      "y coordinate value exceeds the boundary of the image.");
  }
  // account for integer overflow
  if (x < 0 || (x + w) > input->shape()[1] || (x + w) < 0) {
    RETURN_STATUS_UNEXPECTED(
      "Crop: invalid x coordinate value for crop"
      "x coordinate value exceeds the boundary of the image.");
  }

  try {
    LiteMat lite_mat_rgb;
    TensorShape shape{h, w};
    if (input->Rank() == 2) {
      lite_mat_rgb.Init(input->shape()[1], input->shape()[0],
                        const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                        GetLiteCVDataType(input->type()));
    } else {  // rank == 3
      lite_mat_rgb.Init(input->shape()[1], input->shape()[0], input->shape()[2],
                        const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                        GetLiteCVDataType(input->type()));
      int num_channels = input->shape()[2];
      shape = shape.AppendDim(num_channels);
    }

    std::shared_ptr<Tensor> output_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, input->type(), &output_tensor));

    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    LiteMat lite_mat_cut;

    lite_mat_cut.Init(w, h, lite_mat_rgb.channel_, reinterpret_cast<void *>(buffer), GetLiteCVDataType(input->type()));

    bool ret = Crop(lite_mat_rgb, lite_mat_cut, x, y, w, h);
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Crop: image crop failed.");

    *output = output_tensor;
    return Status::OK();
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Crop: " + std::string(e.what()));
  }
  return Status::OK();
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
  } catch (const std::exception &e) {
    jpeg_destroy_decompress(&cinfo);
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  *img_height = cinfo.output_height;
  *img_width = cinfo.output_width;
  jpeg_destroy_decompress(&cinfo);
  return Status::OK();
}

Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                 const std::vector<float> &vec_mean, const std::vector<float> &vec_std) {
  if (input->Rank() != 3) {
    RETURN_STATUS_UNEXPECTED("Normalize: image shape is not <H,W,C>.");
  }

  if (input->type() != DataType::DE_UINT8 && input->type() != DataType::DE_FLOAT32) {
    RETURN_STATUS_UNEXPECTED("Normalize: image datatype is not uint8 or float32.");
  }

  try {
    LiteMat lite_mat_norm;
    bool ret = false;
    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));

    if (input->type() == DataType::DE_UINT8) {
      LiteMat lite_mat_float;
      // change input to float
      ret = ConvertTo(lite_mat_rgb, lite_mat_float, 1.0);
      CHECK_FAIL_RETURN_UNEXPECTED(ret, "Normalize: convert to float datatype failed.");
      ret = SubStractMeanNormalize(lite_mat_float, lite_mat_norm, vec_mean, vec_std);
    } else {  // float32
      ret = SubStractMeanNormalize(lite_mat_rgb, lite_mat_norm, vec_mean, vec_std);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Normalize: normalize failed.");

    std::shared_ptr<Tensor> output_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(input->shape(), DataType(DataType::DE_FLOAT32),
                                              static_cast<uchar *>(lite_mat_norm.data_ptr_), &output_tensor));

    *output = output_tensor;
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Normalize: " + std::string(e.what()));
  }
  return Status::OK();
}

#if defined(ENABLE_CLOUD_FUSION_INFERENCE)
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

    LiteMat im_in, im_out;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({output_height, output_width, 3});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input_cv->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    im_out.Init(output_width, output_height, static_cast<int>(input_cv->shape()[kChannelIndexHWC]),
                reinterpret_cast<void *>(buffer), LDataType::UINT8);
    im_in.Init(static_cast<int>(input_cv->shape()[1]), static_cast<int>(input_cv->shape()[0]),
               static_cast<int>(input_cv->shape()[kChannelIndexHWC]), input_cv->mat().data, LDataType::UINT8);
    if (ResizeCubic(im_in, im_out, output_width, output_height) == false) {
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

#else
Status Resize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t output_height,
              int32_t output_width, double fx, double fy, InterpolationMode mode) {
  if (mode != InterpolationMode::kLinear) {
    RETURN_STATUS_UNEXPECTED("Resize: Only Liner interpolation is supported currently.");
  }
  if (input->Rank() != 3 && input->Rank() != 2) {
    RETURN_STATUS_UNEXPECTED("Resize: input image is not in shape of <H,W,C> or <H,W>");
  }
  if (input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("Resize: image datatype is not uint8.");
  }
  // resize image too large or too small
  const int height_width_scale_limit = 1000;
  if (output_height == 0 || output_height > input->shape()[0] * height_width_scale_limit || output_width == 0 ||
      output_width > input->shape()[1] * height_width_scale_limit) {
    std::string err_msg =
      "Resize: the resizing width or height 1) is too big, it's up to "
      "1000 times the original image; 2) can not be 0.";
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  try {
    LiteMat lite_mat_rgb;
    TensorShape shape{output_height, output_width};
    if (input->Rank() == 2) {
      lite_mat_rgb.Init(input->shape()[1], input->shape()[0],
                        const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                        GetLiteCVDataType(input->type()));
    } else {  // rank == 3
      lite_mat_rgb.Init(input->shape()[1], input->shape()[0], input->shape()[2],
                        const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                        GetLiteCVDataType(input->type()));
      int num_channels = input->shape()[2];
      shape = shape.AppendDim(num_channels);
    }

    LiteMat lite_mat_resize;
    std::shared_ptr<Tensor> output_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, input->type(), &output_tensor));

    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));

    lite_mat_resize.Init(output_width, output_height, lite_mat_rgb.channel_, reinterpret_cast<void *>(buffer),
                         GetLiteCVDataType(input->type()));

    bool ret = ResizeBilinear(lite_mat_rgb, lite_mat_resize, output_width, output_height);
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Resize: bilinear resize failed.");

    *output = output_tensor;
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Resize: " + std::string(e.what()));
  }
  return Status::OK();
}
#endif

Status ResizePreserve(const TensorRow &inputs, int32_t height, int32_t width, int32_t img_orientation,
                      TensorRow *outputs) {
  constexpr int64_t size = 3;
  outputs->resize(size);
  CHECK_FAIL_RETURN_UNEXPECTED(inputs.size() > 0,
                               "Invalid input, should be greater than 0, but got " + std::to_string(inputs.size()));
  std::shared_ptr<Tensor> input = inputs[0];
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Size() >= 3, "Invalid input shape, should be greater than 3 dimensions.");
  LiteMat lite_mat_src(input->shape()[1], input->shape()[0], input->shape()[2],
                       const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                       GetLiteCVDataType(input->type()));

  LiteMat lite_mat_dst;
  std::shared_ptr<Tensor> image_tensor;
  TensorShape new_shape = TensorShape({height, width, input->shape()[2]});
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, DataType(DataType::DE_FLOAT32), &image_tensor));
  uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*image_tensor->begin<uint8_t>()));
  lite_mat_dst.Init(width, height, input->shape()[2], reinterpret_cast<void *>(buffer), LDataType::FLOAT32);

  float ratioShiftWShiftH[3] = {0};
  float invM[2][3] = {{0, 0, 0}, {0, 0, 0}};
  bool ret =
    ResizePreserveARWithFiller(lite_mat_src, lite_mat_dst, height, width, &ratioShiftWShiftH, &invM, img_orientation);
  CHECK_FAIL_RETURN_UNEXPECTED(ret, "Resize: bilinear resize failed.");

  std::shared_ptr<Tensor> ratio_tensor;
  TensorShape ratio_shape = TensorShape({3});
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(ratio_shape, DataType(DataType::DE_FLOAT32),
                                            reinterpret_cast<uint8_t *>(&ratioShiftWShiftH), &ratio_tensor));

  std::shared_ptr<Tensor> invM_tensor;
  TensorShape invM_shape = TensorShape({2, 3});
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(invM_shape, DataType(DataType::DE_FLOAT32),
                                            reinterpret_cast<uint8_t *>(&invM), &invM_tensor));

  (*outputs)[0] = image_tensor;
  (*outputs)[1] = ratio_tensor;
  (*outputs)[2] = invM_tensor;
  return Status::OK();
}

Status RgbToBgr(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  if (input->Rank() != hwc_rank) {
    RETURN_STATUS_UNEXPECTED("RgbToBgr: input image is not in shape of <H,W,C>");
  }
  if (input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("RgbToBgr: image datatype is not uint8.");
  }

  try {
    int output_height = input->shape()[0];
    int output_width = input->shape()[1];

    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));
    LiteMat lite_mat_convert;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({output_height, output_width, 3});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    lite_mat_convert.Init(output_width, output_height, 3, reinterpret_cast<void *>(buffer),
                          GetLiteCVDataType(input->type()));

    bool ret =
      ConvertRgbToBgr(lite_mat_rgb, GetLiteCVDataType(input->type()), output_width, output_height, lite_mat_convert);
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "RgbToBgr: RGBToBGR failed.");

    *output = output_tensor;
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbToBgr: " + std::string(e.what()));
  }
  return Status::OK();
}

Status RgbToGray(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  if (input->Rank() != 3) {
    RETURN_STATUS_UNEXPECTED("RgbToGray: input image is not in shape of <H,W,C>");
  }
  if (input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("RgbToGray: image datatype is not uint8.");
  }

  try {
    int output_height = input->shape()[0];
    int output_width = input->shape()[1];

    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));
    LiteMat lite_mat_convert;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({output_height, output_width, 1});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    lite_mat_convert.Init(output_width, output_height, 1, reinterpret_cast<void *>(buffer),
                          GetLiteCVDataType(input->type()));

    bool ret =
      ConvertRgbToGray(lite_mat_rgb, GetLiteCVDataType(input->type()), output_width, output_height, lite_mat_convert);
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "RgbToGray: RGBToGRAY failed.");

    *output = output_tensor;
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("RgbToGray: " + std::string(e.what()));
  }
  return Status::OK();
}

Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int32_t &pad_top,
           const int32_t &pad_bottom, const int32_t &pad_left, const int32_t &pad_right, const BorderType &border_types,
           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b) {
  if (input->Rank() != 3) {
    RETURN_STATUS_UNEXPECTED("Pad: input image is not in shape of <H,W,C>");
  }

  if (input->type() != DataType::DE_FLOAT32 && input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("Pad: image datatype is not uint8 or float32.");
  }

  if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0) {
    RETURN_STATUS_UNEXPECTED(
      "Pad: "
      "the top, bottom, left, right of pad must be greater than 0.");
  }

  try {
    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));
    LiteMat lite_mat_pad;

    std::shared_ptr<Tensor> output_tensor;

    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - lite_mat_rgb.width_) > pad_left,
                                 "Invalid pad width.");
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - lite_mat_rgb.width_ + pad_left) > pad_right,
                                 "Invalid pad width.");
    int pad_width = lite_mat_rgb.width_ + pad_left + pad_right;
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - lite_mat_rgb.height_) > pad_top,
                                 "Invalid pad height.");
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - lite_mat_rgb.height_ + pad_top) > pad_bottom,
                                 "Invalid pad height.");
    int pad_height = lite_mat_rgb.height_ + pad_top + pad_bottom;
    TensorShape new_shape = TensorShape({pad_height, pad_width, input->shape()[2]});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), &output_tensor));

    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));

    lite_mat_pad.Init(pad_width, pad_height, lite_mat_rgb.channel_, reinterpret_cast<void *>(buffer),
                      GetLiteCVDataType(input->type()));

    bool ret = Pad(lite_mat_rgb, lite_mat_pad, pad_top, pad_bottom, pad_left, pad_right,
                   PaddBorderType::PADD_BORDER_CONSTANT, fill_r, fill_g, fill_b);
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Pad: pad failed.");

    *output = output_tensor;
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Pad: " + std::string(e.what()));
  }
  return Status::OK();
}

static Status RotateAngleWithOutMirror(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                                       const uint64_t orientation) {
  try {
    int height = 0;
    int width = 0;
    double M[6] = {};

    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));

    if (orientation == 3) {
      height = lite_mat_rgb.height_;
      width = lite_mat_rgb.width_;
      M[0] = -1.0f;
      M[1] = 0.0f;
      M[2] = lite_mat_rgb.width_ - 1;
      M[3] = 0.0f;
      M[4] = -1.0f;
      M[5] = lite_mat_rgb.height_ - 1;
    } else if (orientation == 6) {
      height = lite_mat_rgb.width_;
      width = lite_mat_rgb.height_;
      M[0] = 0.0f;
      M[1] = -1.0f;
      M[2] = lite_mat_rgb.height_ - 1;
      M[3] = 1.0f;
      M[4] = 0.0f;
      M[5] = 0.0f;
    } else if (orientation == 8) {
      height = lite_mat_rgb.width_;
      width = lite_mat_rgb.height_;
      M[0] = 0.0f;
      M[1] = 1.0f;
      M[2] = 0.0f;
      M[3] = -1.0f;
      M[4] = 0.0f;
      M[5] = lite_mat_rgb.width_ - 1.0f;
    } else {
    }

    std::vector<size_t> dsize;
    dsize.push_back(width);
    dsize.push_back(height);
    LiteMat lite_mat_affine;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({height, width, input->shape()[2]});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    lite_mat_affine.Init(width, height, lite_mat_rgb.channel_, reinterpret_cast<void *>(buffer),
                         GetLiteCVDataType(input->type()));

    bool ret = Affine(lite_mat_rgb, lite_mat_affine, M, dsize, UINT8_C3(0, 0, 0));
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Rotate: rotate failed.");

    *output = output_tensor;
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Rotate: " + std::string(e.what()));
  }
  return Status::OK();
}

static Status RotateAngleWithMirror(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                                    const uint64_t orientation) {
  try {
    int height = 0;
    int width = 0;
    double M[6] = {};

    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));

    if (orientation == 2) {
      height = lite_mat_rgb.height_;
      width = lite_mat_rgb.width_;
      M[0] = -1.0f;
      M[1] = 0.0f;
      M[2] = lite_mat_rgb.width_ - 1;
      M[3] = 0.0f;
      M[4] = 1.0f;
      M[5] = 0.0f;
    } else if (orientation == 5) {
      height = lite_mat_rgb.width_;
      width = lite_mat_rgb.height_;
      M[0] = 0.0f;
      M[1] = 1.0f;
      M[2] = 0.0f;
      M[3] = 1.0f;
      M[4] = 0.0f;
      M[5] = 0.0f;
    } else if (orientation == 7) {
      height = lite_mat_rgb.width_;
      width = lite_mat_rgb.height_;
      M[0] = 0.0f;
      M[1] = -1.0f;
      M[2] = lite_mat_rgb.height_ - 1;
      M[3] = -1.0f;
      M[4] = 0.0f;
      M[5] = lite_mat_rgb.width_ - 1;
    } else if (orientation == 4) {
      height = lite_mat_rgb.height_;
      width = lite_mat_rgb.width_;
      M[0] = 1.0f;
      M[1] = 0.0f;
      M[2] = 0.0f;
      M[3] = 0.0f;
      M[4] = -1.0f;
      M[5] = lite_mat_rgb.height_ - 1;
    } else {
    }
    std::vector<size_t> dsize;
    dsize.push_back(width);
    dsize.push_back(height);
    LiteMat lite_mat_affine;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({height, width, input->shape()[2]});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    lite_mat_affine.Init(width, height, lite_mat_rgb.channel_, reinterpret_cast<void *>(buffer),
                         GetLiteCVDataType(input->type()));

    bool ret = Affine(lite_mat_rgb, lite_mat_affine, M, dsize, UINT8_C3(0, 0, 0));
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Rotate: rotate failed.");

    *output = output_tensor;
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Rotate: " + std::string(e.what()));
  }
  return Status::OK();
}

static bool IsMirror(int orientation) {
  if (orientation == 2 || orientation == 4 || orientation == 5 || orientation == 7) {
    return true;
  }
  return false;
}
// rotate the image by EXIF orientation
Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const uint64_t orientation) {
  if (input->Rank() != hw_shape && input->Rank() != hwc_rank) {
    RETURN_STATUS_UNEXPECTED("Rotate: input image is not in shape of <H,W,C> or <H,W>");
  }

  if (input->type() != DataType::DE_FLOAT32 && input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("Rotate: image datatype is not float32 or uint8.");
  }

  if (!IsMirror(orientation)) {
    return RotateAngleWithOutMirror(input, output, orientation);
  } else {
    return RotateAngleWithMirror(input, output, orientation);
  }
}

Status GetAffineMatrix(const std::shared_ptr<Tensor> &input, std::vector<float_t> *matrix, float_t degrees,
                       const std::vector<float_t> &translation, float_t scale, const std::vector<float_t> &shear) {
  CHECK_FAIL_RETURN_UNEXPECTED(translation.size() >= 2, "AffineOp::Compute translation_ size should >= 2");
  float_t translation_x = translation[0];
  float_t translation_y = translation[1];
  float_t degrees_tmp = 0.0;
  RETURN_IF_NOT_OK(DegreesToRadians(degrees, &degrees_tmp));
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
    CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Size() >= 3, "Invalid input shape, should be 3.");
    if (interpolation != InterpolationMode::kLinear) {
      MS_LOG(WARNING) << "Only Bilinear interpolation supported for now";
    }
    std::vector<float_t> matrix;
    RETURN_IF_NOT_OK(GetAffineMatrix(input, &matrix, degrees, translation, scale, shear));
    int height = 0;
    int width = 0;
    CHECK_FAIL_RETURN_UNEXPECTED(matrix.size() <= 6, "Invalid mat shape.");
    double M[6] = {};
    for (int i = 0; i < matrix.size(); i++) {
      M[i] = static_cast<double>(matrix[i]);
    }

    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));

    height = lite_mat_rgb.height_;
    width = lite_mat_rgb.width_;
    std::vector<size_t> dsize;
    dsize.push_back(width);
    dsize.push_back(height);
    LiteMat lite_mat_affine;
    std::shared_ptr<Tensor> output_tensor;
    TensorShape new_shape = TensorShape({height, width, input->shape()[2]});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    lite_mat_affine.Init(width, height, lite_mat_rgb.channel_, reinterpret_cast<void *>(buffer),
                         GetLiteCVDataType(input->type()));

    bool ret = Affine(lite_mat_rgb, lite_mat_affine, M, dsize,
                      UINT8_C3(fill_value[kRIndex], fill_value[kGIndex], fill_value[kBIndex]));
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Affine: affine failed.");

    *output = output_tensor;
    return Status::OK();
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Affine: " + std::string(e.what()));
  }
}

Status GaussianBlur(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t kernel_x,
                    int32_t kernel_y, float sigma_x, float sigma_y) {
  try {
    LiteMat lite_mat_input;
    if (input->Rank() == 3) {
      if (input->shape()[2] != 1 && input->shape()[2] != 3) {
        RETURN_STATUS_UNEXPECTED("GaussianBlur: input image is not in channel of 1 or 3");
      }
      lite_mat_input = LiteMat(input->shape()[1], input->shape()[0], input->shape()[2],
                               const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                               GetLiteCVDataType(input->type()));
    } else if (input->Rank() == 2) {
      lite_mat_input = LiteMat(input->shape()[1], input->shape()[0],
                               const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                               GetLiteCVDataType(input->type()));
    } else {
      RETURN_STATUS_UNEXPECTED("GaussianBlur: input image is not in shape of <H,W,C> or <H,W>");
    }

    std::shared_ptr<Tensor> output_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), &output_tensor));
    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
    LiteMat lite_mat_output;
    lite_mat_output.Init(lite_mat_input.width_, lite_mat_input.height_, lite_mat_input.channel_,
                         reinterpret_cast<void *>(buffer), GetLiteCVDataType(input->type()));
    bool ret = GaussianBlur(lite_mat_input, lite_mat_output, {kernel_x, kernel_y}, static_cast<double>(sigma_x),
                            static_cast<double>(sigma_y));
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "GaussianBlur: GaussianBlur failed.");
    *output = output_tensor;
    return Status::OK();
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("GaussianBlur: " + std::string(e.what()));
  }
}

Status ImageNumChannels(const std::shared_ptr<Tensor> &image, dsize_t *channels) {
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

Status ValidateImageRank(const std::string &op_name, int32_t rank) {
  if (rank != 2 && rank != 3) {
    std::string err_msg = op_name + ": image shape is not <H,W,C> or <H, W>, but got rank:" + std::to_string(rank);
    if (rank == 1) {
      err_msg = err_msg + ", may need to do Decode operation first.";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status HwcToChw(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  try {
    if (input->Rank() <= 3) {
      int output_height = input->shape()[0];
      int output_width = input->shape()[1];
      int output_channel = input->shape()[2];
      LiteMat lite_mat_hwc(input->shape()[1], input->shape()[0], input->shape()[2],
                           const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                           GetLiteCVDataType(input->type()));
      LiteMat lite_mat_chw;
      std::shared_ptr<Tensor> output_tensor;
      TensorShape new_shape = TensorShape({output_channel, output_height, output_width});
      RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), &output_tensor));
      uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));
      lite_mat_chw.Init(output_height, output_channel, output_width, reinterpret_cast<void *>(buffer),
                        GetLiteCVDataType(input->type()));
      bool ret = HWC2CHW(lite_mat_hwc, lite_mat_chw);
      CHECK_FAIL_RETURN_UNEXPECTED(ret, "HwcToChw: HwcToChw failed.");
      *output = output_tensor;
    } else {
      RETURN_STATUS_UNEXPECTED("HwcToChw: input image is not in shape of <H,W,C> or <H,W>");
    }
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("HwcToChw: " + std::string(e.what()));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
