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
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <utility>
#include "utils/ms_utils.h"
#include "minddata/dataset/kernels/image/lite_cv/lite_mat.h"
#include "minddata/dataset/kernels/image/lite_cv/image_process.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/util/random.h"

#define MAX_INT_PRECISION 16777216  // float int precision is 16777216
namespace mindspore {
namespace dataset {
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
    } catch (std::runtime_error &e) {
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
  } catch (std::runtime_error &e) {
    return DestroyDecompressAndReturnError(e.what());
  }
  if (crop_x == 0 && crop_y == 0 && crop_w == 0 && crop_h == 0) {
    crop_w = cinfo.output_width;
    crop_h = cinfo.output_height;
  } else if (crop_w == 0 || static_cast<unsigned int>(crop_w + crop_x) > cinfo.output_width || crop_h == 0 ||
             static_cast<unsigned int>(crop_h + crop_y) > cinfo.output_height) {
    return DestroyDecompressAndReturnError("Decode: invalid crop size");
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
  std::shared_ptr<Tensor> output_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(ts, DataType(DataType::DE_UINT8), &output_tensor));
  const int buffer_size = output_tensor->SizeInBytes();
  JSAMPLE *buffer = reinterpret_cast<JSAMPLE *>(&(*output_tensor->begin<uint8_t>()));
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

static LDataType GetLiteCVDataType(DataType data_type) {
  if (data_type == DataType::DE_UINT8) {
    return LDataType::UINT8;
  } else if (data_type == DataType::DE_FLOAT32) {
    return LDataType::FLOAT32;
  } else {
    return LDataType::UNKNOWN;
  }
}

Status Decode(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  if (IsNonEmptyJPEG(input)) {
    return JpegCropAndDecode(input, output);
  } else {
    RETURN_STATUS_UNEXPECTED("Decode: Decode only supports jpeg for android");
  }
}

Status Crop(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int x, int y, int w, int h) {
  if (input->Rank() != 3 && input->Rank() != 2) {
    RETURN_STATUS_UNEXPECTED("Crop: image shape is not <H,W,C> or <H,W>");
  }

  if (input->type() != DataType::DE_FLOAT32 && input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("Crop: image datatype is not float32 or uint8");
  }

  // account for integer overflow
  if (y < 0 || (y + h) > input->shape()[0] || (y + h) < 0) {
    RETURN_STATUS_UNEXPECTED("Crop: invalid y coordinate value for crop");
  }
  // account for integer overflow
  if (x < 0 || (x + w) > input->shape()[1] || (x + w) < 0) {
    RETURN_STATUS_UNEXPECTED("Crop: invalid x coordinate value for crop");
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
  } catch (std::runtime_error &e) {
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
  } catch (std::runtime_error &e) {
    jpeg_destroy_decompress(&cinfo);
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  *img_height = cinfo.output_height;
  *img_width = cinfo.output_width;
  jpeg_destroy_decompress(&cinfo);
  return Status::OK();
}

Status Normalize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                 const std::shared_ptr<Tensor> &mean, const std::shared_ptr<Tensor> &std) {
  if (input->Rank() != 3) {
    RETURN_STATUS_UNEXPECTED("Normalize: image shape is not <H,W,C>.");
  }

  if (input->type() != DataType::DE_UINT8 && input->type() != DataType::DE_FLOAT32) {
    RETURN_STATUS_UNEXPECTED("Normalize: image datatype is not uint8 or float32.");
  }

  mean->Squeeze();
  if (mean->type() != DataType::DE_FLOAT32 || mean->Rank() != 1 || mean->shape()[0] != 3) {
    std::string err_msg = "Normalize: mean should be of size 3 and type float.";
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  std->Squeeze();
  if (std->type() != DataType::DE_FLOAT32 || std->Rank() != 1 || std->shape()[0] != 3) {
    std::string err_msg = "Normalize: std should be of size 3 and type float.";
    return Status(StatusCode::kMDShapeMisMatch, err_msg);
  }
  // convert mean, std back to vector
  std::vector<float> vec_mean;
  std::vector<float> vec_std;
  try {
    for (uint8_t i = 0; i < 3; i++) {
      float mean_c, std_c;
      RETURN_IF_NOT_OK(mean->GetItemAt<float>(&mean_c, {i}));
      RETURN_IF_NOT_OK(std->GetItemAt<float>(&std_c, {i}));
      vec_mean.push_back(mean_c);
      vec_std.push_back(std_c);
    }

    LiteMat lite_mat_norm;
    bool ret = false;
    LiteMat lite_mat_rgb(input->shape()[1], input->shape()[0], input->shape()[2],
                         const_cast<void *>(reinterpret_cast<const void *>(input->GetBuffer())),
                         GetLiteCVDataType(input->type()));

    std::shared_ptr<Tensor> output_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), DataType(DataType::DE_FLOAT32), &output_tensor));

    uint8_t *buffer = reinterpret_cast<uint8_t *>(&(*output_tensor->begin<uint8_t>()));

    lite_mat_norm.Init(lite_mat_rgb.width_, lite_mat_rgb.height_, lite_mat_rgb.channel_,
                       reinterpret_cast<void *>(buffer), GetLiteCVDataType(input->type()));

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

    *output = output_tensor;
  } catch (std::runtime_error &e) {
    RETURN_STATUS_UNEXPECTED("Normalize: " + std::string(e.what()));
  }
  return Status::OK();
}

Status Resize(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t output_height,
              int32_t output_width, double fx, double fy, InterpolationMode mode) {
  if (input->Rank() != 3 && input->Rank() != 2) {
    RETURN_STATUS_UNEXPECTED("Resize: input image is not in shape of <H,W,C> or <H,W>");
  }
  if (input->type() != DataType::DE_UINT8) {
    RETURN_STATUS_UNEXPECTED("Resize: image datatype is not uint8.");
  }
  // resize image too large or too small
  if (output_height == 0 || output_height > input->shape()[0] * 1000 || output_width == 0 ||
      output_width > input->shape()[1] * 1000) {
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
  } catch (std::runtime_error &e) {
    RETURN_STATUS_UNEXPECTED("Resize: " + std::string(e.what()));
  }
  return Status::OK();
}

Status ResizePreserve(const TensorRow &inputs, int32_t height, int32_t width, int32_t img_orientation,
                      TensorRow *outputs) {
  outputs->resize(3);
  std::shared_ptr<Tensor> input = inputs[0];
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
  } catch (std::runtime_error &e) {
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

    int pad_width = lite_mat_rgb.width_ + pad_left + pad_right;
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
  } catch (std::runtime_error &e) {
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
  } catch (std::runtime_error &e) {
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
  } catch (std::runtime_error &e) {
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
  if (input->Rank() != 3) {
    RETURN_STATUS_UNEXPECTED("Rotate: input image is not in shape of <H,W,C>");
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

Status Affine(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::vector<float_t> &mat,
              InterpolationMode interpolation, uint8_t fill_r, uint8_t fill_g, uint8_t fill_b) {
  try {
    if (interpolation != InterpolationMode::kLinear) {
      MS_LOG(WARNING) << "Only Bilinear interpolation supported for now";
    }
    int height = 0;
    int width = 0;
    double M[6] = {};
    for (int i = 0; i < mat.size(); i++) {
      M[i] = static_cast<double>(mat[i]);
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

    bool ret = Affine(lite_mat_rgb, lite_mat_affine, M, dsize, UINT8_C3(fill_r, fill_g, fill_b));
    CHECK_FAIL_RETURN_UNEXPECTED(ret, "Affine: affine failed.");

    *output = output_tensor;
    return Status::OK();
  } catch (std::runtime_error &e) {
    RETURN_STATUS_UNEXPECTED("Affine: " + std::string(e.what()));
  }
}

}  // namespace dataset
}  // namespace mindspore
