/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/kernels/image/image_utils.h"
#include <opencv2/imgproc/types_c.h>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#include "common/utils.h"
#include "dataset/core/constants.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/tensor.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/util/random.h"

#define MAX_INT_PRECISION 16777216  // float int precision is 16777216
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

Status Flip(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, int flip_code) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));

  std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(input_cv->shape(), input_cv->type());
  RETURN_UNEXPECTED_IF_NULL(output_cv);
  (void)output_cv->StartAddr();
  if (input_cv->mat().data) {
    try {
      cv::flip(input_cv->mat(), output_cv->mat(), flip_code);
      *output = std::static_pointer_cast<Tensor>(output_cv);
      return Status::OK();
    } catch (const cv::Exception &e) {
      RETURN_STATUS_UNEXPECTED("Error in flip op.");
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor, the input data is null");
  }
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
    RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
  }
  if (input_cv->Rank() != 3 && input_cv->Rank() != 2) {
    RETURN_STATUS_UNEXPECTED("Input Tensor is not in shape of <H,W,C> or <H,W>");
  }
  cv::Mat in_image = input_cv->mat();
  // resize image too large or too small
  if (output_height == 0 || output_height > in_image.rows * 1000 || output_width == 0 ||
      output_width > in_image.cols * 1000) {
    std::string err_msg =
      "The resizing width or height 1) is too big, it's up to "
      "1000 times the original image; 2) can not be 0.";
    return Status(StatusCode::kShapeMisMatch, err_msg);
  }
  try {
    TensorShape shape{output_height, output_width};
    if (input_cv->Rank() == 3) shape = shape.AppendDim(input_cv->shape()[2]);
    std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(shape, input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    auto cv_mode = GetCVInterpolationMode(mode);
    cv::resize(in_image, output_cv->mat(), cv::Size(output_width, output_height), fx, fy, cv_mode);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in image resize.");
  }
}

bool HasJpegMagic(const unsigned char *data, size_t data_size) {
  const unsigned char *kJpegMagic = (unsigned char *)"\xFF\xD8\xFF";
  constexpr size_t kJpegMagicLen = 3;
  return data_size >= kJpegMagicLen && memcmp(data, kJpegMagic, kJpegMagicLen) == 0;
}

Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  if (input->StartAddr() == nullptr) {
    RETURN_STATUS_UNEXPECTED("Tensor is nullptr");
  }
  if (HasJpegMagic(input->StartAddr(), input->SizeInBytes())) {
    return JpegCropAndDecode(input, output);
  } else {
    return DecodeCv(input, output);
  }
}

Status DecodeCv(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
  }
  try {
    cv::Mat img_mat = cv::imdecode(input_cv->mat(), cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (img_mat.data == nullptr) {
      std::string err = "Error in decoding\t";
      RETURN_STATUS_UNEXPECTED(err);
    }
    cv::cvtColor(img_mat, img_mat, static_cast<int>(cv::COLOR_BGR2RGB));
    std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(img_mat);
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in image Decode");
  }
}

static void JpegInitSource(j_decompress_ptr cinfo) {}

static boolean JpegFillInputBuffer(j_decompress_ptr cinfo) {
  if (cinfo->src->bytes_in_buffer == 0) {
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
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
#if defined(_WIN32) || defined(_WIN64)
  cinfo->src->skip_input_data = reinterpret_cast<void (*)(j_decompress_ptr, long)>(JpegSkipInputData);
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
    int num_lines_read = jpeg_read_scanlines(cinfo, &scanline_ptr, 1);
    if (cinfo->out_color_space == JCS_CMYK && num_lines_read > 0) {
      for (int i = 0; i < crop_w; ++i) {
        int cmyk_pixel = 4 * i + offset;
        const int c = scanline_ptr[cmyk_pixel];
        const int m = scanline_ptr[cmyk_pixel + 1];
        const int y = scanline_ptr[cmyk_pixel + 2];
        const int k = scanline_ptr[cmyk_pixel + 3];
        int r, g, b;
        if (cinfo->saw_Adobe_marker) {
          r = (k * c) / 255;
          g = (k * m) / 255;
          b = (k * y) / 255;
        } else {
          r = (255 - c) * (255 - k) / 255;
          g = (255 - m) * (255 - k) / 255;
          b = (255 - y) * (255 - k) / 255;
        }
        buffer[3 * i + 0] = r;
        buffer[3 * i + 1] = g;
        buffer[3 * i + 2] = b;
      }
    } else if (num_lines_read > 0) {
      int copy_status = memcpy_s(buffer, buffer_size, scanline_ptr + offset, stride);
      if (copy_status != 0) {
        jpeg_destroy_decompress(cinfo);
        RETURN_STATUS_UNEXPECTED("memcpy failed");
      }
    } else {
      jpeg_destroy_decompress(cinfo);
      std::string err_msg = "failed to read scanline";
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
      std::string err_msg = "wrong number of components";
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
    JpegSetSource(&cinfo, input->StartAddr(), input->SizeInBytes());
    (void)jpeg_read_header(&cinfo, TRUE);
    RETURN_IF_NOT_OK(JpegSetColorSpace(&cinfo));
    jpeg_calc_output_dimensions(&cinfo);
  } catch (std::runtime_error &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  if (crop_x == 0 && crop_y == 0 && crop_w == 0 && crop_h == 0) {
    crop_w = cinfo.output_width;
    crop_h = cinfo.output_height;
  } else if (crop_w == 0 || static_cast<unsigned int>(crop_w + crop_x) > cinfo.output_width || crop_h == 0 ||
             static_cast<unsigned int>(crop_h + crop_y) > cinfo.output_height) {
    return DestroyDecompressAndReturnError("Crop window is not valid");
  }
  const int mcu_size = cinfo.min_DCT_scaled_size;
  unsigned int crop_x_aligned = (crop_x / mcu_size) * mcu_size;
  unsigned int crop_w_aligned = crop_w + crop_x - crop_x_aligned;
  try {
    (void)jpeg_start_decompress(&cinfo);
    jpeg_crop_scanline(&cinfo, &crop_x_aligned, &crop_w_aligned);
  } catch (std::runtime_error &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  JDIMENSION skipped_scanlines = jpeg_skip_scanlines(&cinfo, crop_y);
  // three number of output components, always convert to RGB and output
  constexpr int kOutNumComponents = 3;
  TensorShape ts = TensorShape({crop_h, crop_w, kOutNumComponents});
  auto output_tensor = std::make_shared<Tensor>(ts, DataType(DataType::DE_UINT8));
  const int buffer_size = output_tensor->SizeInBytes();
  JSAMPLE *buffer = static_cast<JSAMPLE *>(output_tensor->StartAddr());
  const int max_scanlines_to_read = skipped_scanlines + crop_h;
  // stride refers to output tensor, which has 3 components at most
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

Status Rescale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rescale, float shift) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
  }
  cv::Mat input_image = input_cv->mat();
  std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(input_cv->shape(), DataType(DataType::DE_FLOAT32));
  RETURN_UNEXPECTED_IF_NULL(output_cv);
  try {
    input_image.convertTo(output_cv->mat(), CV_32F, rescale, shift);
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in image rescale");
  }
  return Status::OK();
}

Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!input_cv->mat().data) {
    RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
  }
  if (input_cv->Rank() != 3 && input_cv->Rank() != 2) {
    RETURN_STATUS_UNEXPECTED("Shape not <H,W,C> or <H,W>");
  }
  try {
    TensorShape shape{h, w};
    if (input_cv->Rank() == 3) shape = shape.AppendDim(input_cv->shape()[2]);
    std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(shape, input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    cv::Rect roi(x, y, w, h);
    (input_cv->mat())(roi).copyTo(output_cv->mat());
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Unexpected error in crop.");
  }
}

Status HwcToChw(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    if (input_cv->shape().Size() != 3 && input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("The shape is incorrect: number of channels is not equal 3");
    }
    cv::Mat output_img;

    int height = input_cv->shape()[0];
    int width = input_cv->shape()[1];
    int num_channels = input_cv->shape()[2];

    auto output_cv = std::make_unique<CVTensor>(TensorShape{num_channels, height, width}, input_cv->type());
    for (int i = 0; i < num_channels; ++i) {
      cv::Mat mat;
      RETURN_IF_NOT_OK(output_cv->Mat({i}, &mat));
      cv::extractChannel(input_cv->mat(), mat, i);
    }
    *output = std::move(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Unexpected error in ChannelSwap.");
  }
}

Status SwapRedAndBlue(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(std::move(input));
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    if (input_cv->shape().Size() != 3 && input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("The shape is incorrect: number of channels is not equal 3");
    }
    auto output_cv = std::make_shared<CVTensor>(input_cv->shape(), input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    cv::cvtColor(input_cv->mat(), output_cv->mat(), static_cast<int>(cv::COLOR_BGR2RGB));
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Unexpected error in ChangeMode.");
  }
}

Status CropAndResize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y,
                     int crop_height, int crop_width, int target_height, int target_width, InterpolationMode mode) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    if (input_cv->Rank() != 3 && input_cv->Rank() != 2) {
      RETURN_STATUS_UNEXPECTED("Ishape not <H,W,C> or <H,W>");
    }
    // image too large or too small
    if (crop_height == 0 || crop_width == 0 || target_height == 0 || target_height > crop_height * 1000 ||
        target_width == 0 || target_height > crop_width * 1000) {
      std::string err_msg =
        "The resizing width or height 1) is too big, it's up to "
        "1000 times the original image; 2) can not be 0.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    cv::Rect roi(x, y, crop_width, crop_height);
    auto cv_mode = GetCVInterpolationMode(mode);
    cv::Mat cv_in = input_cv->mat();
    TensorShape shape{target_height, target_width};
    if (input_cv->Rank() == 3) shape = shape.AppendDim(input_cv->shape()[2]);
    std::shared_ptr<CVTensor> cvt_out = std::make_shared<CVTensor>(shape, input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(cvt_out);
    cv::resize(cv_in(roi), cvt_out->mat(), cv::Size(target_width, target_height), 0, 0, cv_mode);
    *output = std::static_pointer_cast<Tensor>(cvt_out);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Unexpected error in CropAndResize.");
  }
}

Status Rotate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float fx, float fy, float degree,
              InterpolationMode interpolation, bool expand, uint8_t fill_r, uint8_t fill_g, uint8_t fill_b) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    cv::Mat input_img = input_cv->mat();
    if (input_img.cols > (MAX_INT_PRECISION * 2) || input_img.rows > (MAX_INT_PRECISION * 2)) {
      RETURN_STATUS_UNEXPECTED("Image too large center not precise");
    }
    // default to center of image
    if (fx == -1 && fy == -1) {
      fx = (input_img.cols - 1) / 2.0;
      fy = (input_img.rows - 1) / 2.0;
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
      output_cv = std::make_shared<CVTensor>(input_cv->shape(), input_cv->type());
      RETURN_UNEXPECTED_IF_NULL(output_cv);
      // using inter_nearest to comply with python default
      cv::warpAffine(input_img, output_cv->mat(), rot, input_img.size(), GetCVInterpolationMode(interpolation),
                     cv::BORDER_CONSTANT, fill_color);
    } else {
      // we resize here since the shape changes
      // create a new bounding box with the rotate
      cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), input_img.size(), degree).boundingRect2f();
      rot.at<double>(0, 2) += bbox.width / 2.0 - input_img.cols / 2.0;
      rot.at<double>(1, 2) += bbox.height / 2.0 - input_img.rows / 2.0;
      // use memcpy and don't compute the new shape since openCV has a rounding problem
      cv::warpAffine(input_img, output_img, rot, bbox.size(), GetCVInterpolationMode(interpolation),
                     cv::BORDER_CONSTANT, fill_color);
      output_cv = std::make_shared<CVTensor>(output_img);
      RETURN_UNEXPECTED_IF_NULL(output_cv);
    }
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in image rotation");
  }
  return Status::OK();
}

Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                 const std::shared_ptr<Tensor> &mean, const std::shared_ptr<Tensor> &std) {
  std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
  if (!(input_cv->mat().data && input_cv->Rank() == 3)) {
    RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
  }
  cv::Mat in_image = input_cv->mat();
  std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(input_cv->shape(), DataType(DataType::DE_FLOAT32));
  RETURN_UNEXPECTED_IF_NULL(output_cv);
  mean->Squeeze();
  if (mean->type() != DataType::DE_FLOAT32 || mean->Rank() != 1 || mean->shape()[0] != 3) {
    std::string err_msg = "Mean tensor should be of size 3 and type float.";
    return Status(StatusCode::kShapeMisMatch, err_msg);
  }
  std->Squeeze();
  if (std->type() != DataType::DE_FLOAT32 || std->Rank() != 1 || std->shape()[0] != 3) {
    std::string err_msg = "Std tensor should be of size 3 and type float.";
    return Status(StatusCode::kShapeMisMatch, err_msg);
  }
  try {
    // NOTE: We are assuming the input image is in RGB and the mean
    // and std are in RGB
    cv::Mat rgb[3];
    cv::split(in_image, rgb);
    for (uint8_t i = 0; i < 3; i++) {
      float mean_c, std_c;
      RETURN_IF_NOT_OK(mean->GetItemAt<float>(&mean_c, {i}));
      RETURN_IF_NOT_OK(std->GetItemAt<float>(&std_c, {i}));
      rgb[i].convertTo(rgb[i], CV_32F, 1.0 / std_c, (-mean_c / std_c));
    }
    cv::merge(rgb, 3, output_cv->mat());
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Unexpected error in Normalize");
  }
}

Status AdjustBrightness(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    if (input_cv->Rank() != 3 && input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("Shape not <H,W,3> or <H,W>");
    }
    auto output_cv = std::make_shared<CVTensor>(input_cv->shape(), input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    output_cv->mat() = input_img * alpha;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in adjust brightness");
  }
  return Status::OK();
}

Status AdjustContrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    if (input_cv->Rank() != 3 && input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("Shape not <H,W,3> or <H,W>");
    }
    cv::Mat gray, output_img;
    cv::cvtColor(input_img, gray, CV_RGB2GRAY);
    int mean_img = static_cast<int>(cv::mean(gray).val[0] + 0.5);
    std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(input_cv->shape(), input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    output_img = cv::Mat::zeros(input_img.rows, input_img.cols, CV_8UC1);
    output_img = output_img + mean_img;
    cv::cvtColor(output_img, output_img, CV_GRAY2RGB);
    output_cv->mat() = output_img * (1.0 - alpha) + input_img * alpha;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in adjust contrast");
  }
  return Status::OK();
}

Status AdjustSaturation(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &alpha) {
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    if (input_cv->Rank() != 3 && input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("Shape not <H,W,3> or <H,W>");
    }
    auto output_cv = std::make_shared<CVTensor>(input_cv->shape(), input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    cv::Mat output_img = output_cv->mat();
    cv::Mat gray;
    cv::cvtColor(input_img, gray, CV_RGB2GRAY);
    cv::cvtColor(gray, output_img, CV_GRAY2RGB);
    output_cv->mat() = output_img * (1.0 - alpha) + input_img * alpha;
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in adjust saturation");
  }
  return Status::OK();
}

Status AdjustHue(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const float &hue) {
  if (hue > 0.5 || hue < -0.5) {
    MS_LOG(ERROR) << "Hue factor is not in [-0.5, 0.5].";
    RETURN_STATUS_UNEXPECTED("hue_factor is not in [-0.5, 0.5].");
  }
  try {
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    cv::Mat input_img = input_cv->mat();
    if (!input_cv->mat().data) {
      RETURN_STATUS_UNEXPECTED("Could not convert to CV Tensor");
    }
    if (input_cv->Rank() != 3 && input_cv->shape()[2] != 3) {
      RETURN_STATUS_UNEXPECTED("Shape not <H,W,3> or <H,W>");
    }
    auto output_cv = std::make_shared<CVTensor>(input_cv->shape(), input_cv->type());
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    cv::Mat output_img;
    cv::cvtColor(input_img, output_img, CV_RGB2HSV_FULL);
    for (int y = 0; y < output_img.cols; y++) {
      for (int x = 0; x < output_img.rows; x++) {
        uint8_t cur1 = output_img.at<cv::Vec3b>(cv::Point(y, x))[0];
        uint8_t h_hue = 0;
        h_hue = static_cast<uint8_t>(hue * 255);
        cur1 += h_hue;
        output_img.at<cv::Vec3b>(cv::Point(y, x))[0] = cur1;
      }
    }
    cv::cvtColor(output_img, output_cv->mat(), CV_HSV2RGB_FULL);
    *output = std::static_pointer_cast<Tensor>(output_cv);
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in adjust hue");
  }
  return Status::OK();
}

Status GenerateRandomCropBox(int input_height, int input_width, float ratio, float lb, float ub, int max_itr,
                             cv::Rect *crop_box, uint32_t seed) {
  try {
    std::mt19937 rnd;
    rnd.seed(GetSeed());
    if (input_height <= 0 || input_width <= 0 || ratio <= 0.0 || lb <= 0.0 || lb > ub) {
      RETURN_STATUS_UNEXPECTED("Invalid inputs GenerateRandomCropBox");
    }
    std::uniform_real_distribution<float> rd_crop_ratio(lb, ub);
    float crop_ratio;
    int crop_width, crop_height;
    bool crop_success = false;
    int64_t input_area = input_height * input_width;
    for (auto i = 0; i < max_itr; i++) {
      crop_ratio = rd_crop_ratio(rnd);
      crop_width = static_cast<int32_t>(std::round(std::sqrt(input_area * static_cast<double>(crop_ratio) / ratio)));
      crop_height = static_cast<int32_t>(std::round(crop_width * ratio));
      if (crop_width <= input_width && crop_height <= input_height) {
        crop_success = true;
        break;
      }
    }
    if (crop_success == false) {
      ratio = static_cast<float>(input_height) / input_width;
      crop_ratio = rd_crop_ratio(rnd);
      crop_width = static_cast<int>(std::lround(std::sqrt(input_area * static_cast<double>(crop_ratio) / ratio)));
      crop_height = static_cast<int>(std::lround(crop_width * ratio));
      crop_height = (crop_height > input_height) ? input_height : crop_height;
      crop_width = (crop_width > input_width) ? input_width : crop_width;
    }
    std::uniform_int_distribution<> rd_x(0, input_width - crop_width);
    std::uniform_int_distribution<> rd_y(0, input_height - crop_height);
    *crop_box = cv::Rect(rd_x(rnd), rd_y(rnd), crop_width, crop_height);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("error in GenerateRandomCropBox.");
  }
}

Status CheckOverlapConstraint(const cv::Rect &crop_box, const std::vector<cv::Rect> &bounding_boxes,
                              float min_intersect_ratio, bool *is_satisfied) {
  try {
    // not satisfied if the crop box contains no pixel
    if (crop_box.area() < 1.0) {
      *is_satisfied = false;
    }
    for (const auto &b_box : bounding_boxes) {
      const float b_box_area = b_box.area();
      // not satisfied if the bounding box contains no pixel
      if (b_box_area < 1.0) {
        continue;
      }
      const float intersect_ratio = (crop_box & b_box).area() / b_box_area;
      if (intersect_ratio >= min_intersect_ratio) {
        *is_satisfied = true;
        break;
      }
    }
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("error in CheckOverlapConstraint.");
  }
}

Status Erase(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t box_height,
             int32_t box_width, int32_t num_patches, bool bounded, bool random_color, uint8_t fill_r, uint8_t fill_g,
             uint8_t fill_b) {
  try {
    std::mt19937 rnd;
    rnd.seed(GetSeed());
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    if (input_cv->mat().data == nullptr || (input_cv->Rank() != 3 && input_cv->shape()[2] != 3)) {
      RETURN_STATUS_UNEXPECTED("bad CV Tensor input for erase");
    }
    cv::Mat input_img = input_cv->mat();
    int32_t image_h = input_cv->shape()[0];
    int32_t image_w = input_cv->shape()[1];
    // check if erase size is bigger than image itself
    if (box_height > image_h || box_width > image_w) {
      RETURN_STATUS_UNEXPECTED("input box size too large for image erase");
    }

    // for random color
    std::normal_distribution<double> normal_distribution(0, 1);
    std::uniform_int_distribution<int> height_distribution_bound(0, image_h - box_height);
    std::uniform_int_distribution<int> width_distribution_bound(0, image_w - box_width);
    std::uniform_int_distribution<int> height_distribution_unbound(0, image_h + box_height);
    std::uniform_int_distribution<int> width_distribution_unbound(0, image_w + box_width);
    // core logic
    // update values based on random erasing or cutout

    for (int32_t i = 0; i < num_patches; i++) {
      // rows in cv mat refers to the height of the cropped box
      // we determine h_start and w_start using two different distributions as erasing is used by two different
      // image augmentations. The bounds are also different in each case.
      int32_t h_start = (bounded) ? height_distribution_bound(rnd) : (height_distribution_unbound(rnd) - box_height);
      int32_t w_start = (bounded) ? width_distribution_bound(rnd) : (width_distribution_unbound(rnd) - box_width);

      int32_t max_width = (w_start + box_width > image_w) ? image_w : w_start + box_width;
      int32_t max_height = (h_start + box_height > image_h) ? image_h : h_start + box_height;
      // check for starting range >= 0, here the start range is checked after for cut out, for random erasing
      // w_start and h_start will never be less than 0.
      h_start = (h_start < 0) ? 0 : h_start;
      w_start = (w_start < 0) ? 0 : w_start;
      for (int y = w_start; y < max_width; y++) {
        for (int x = h_start; x < max_height; x++) {
          if (random_color) {
            // fill each box with a random value
            input_img.at<cv::Vec3b>(cv::Point(y, x))[0] = static_cast<int32_t>(normal_distribution(rnd));
            input_img.at<cv::Vec3b>(cv::Point(y, x))[1] = static_cast<int32_t>(normal_distribution(rnd));
            input_img.at<cv::Vec3b>(cv::Point(y, x))[2] = static_cast<int32_t>(normal_distribution(rnd));
          } else {
            input_img.at<cv::Vec3b>(cv::Point(y, x))[0] = fill_r;
            input_img.at<cv::Vec3b>(cv::Point(y, x))[1] = fill_g;
            input_img.at<cv::Vec3b>(cv::Point(y, x))[2] = fill_b;
          }
        }
      }
    }
    *output = std::static_pointer_cast<Tensor>(input);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Error in erasing");
  }
}

Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const int32_t &pad_top,
           const int32_t &pad_bottom, const int32_t &pad_left, const int32_t &pad_right, const BorderType &border_types,
           uint8_t fill_r, uint8_t fill_g, uint8_t fill_b) {
  try {
    // input image
    std::shared_ptr<CVTensor> input_cv = CVTensor::AsCVTensor(input);
    // get the border type in openCV
    auto b_type = GetCVBorderType(border_types);
    // output image
    cv::Mat out_image;
    if (b_type == cv::BORDER_CONSTANT) {
      cv::Scalar fill_color = cv::Scalar(fill_b, fill_g, fill_r);
      cv::copyMakeBorder(input_cv->mat(), out_image, pad_top, pad_bottom, pad_left, pad_right, b_type, fill_color);
    } else {
      cv::copyMakeBorder(input_cv->mat(), out_image, pad_top, pad_bottom, pad_left, pad_right, b_type);
    }
    std::shared_ptr<CVTensor> output_cv = std::make_shared<CVTensor>(out_image);
    RETURN_UNEXPECTED_IF_NULL(output_cv);
    *output = std::static_pointer_cast<Tensor>(output_cv);
    return Status::OK();
  } catch (const cv::Exception &e) {
    RETURN_STATUS_UNEXPECTED("Unexpected error in pad");
  }
}
}  // namespace dataset
}  // namespace mindspore
