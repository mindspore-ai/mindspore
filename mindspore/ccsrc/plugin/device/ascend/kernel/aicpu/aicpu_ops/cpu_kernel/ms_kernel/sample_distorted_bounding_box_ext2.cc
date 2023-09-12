/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "ms_kernel/sample_distorted_bounding_box_ext2.h"

#include <random>
#include <vector>
#include <securec.h>
#include "common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 3;
const uint32_t IMAGE_SIZE_NUM = 3;
const uint32_t SHAPE_BOUNDING_BOXES_SIZE = 3;
const uint32_t BOUNDING_BOXES_SIZE = 4;

const char *kSDBBExt2 = "SampleDistortedBoundingBoxExt2";

#define SDBBExt2CpuKernel_COMPUTE_CASE(DTYPE, TYPE, CTX)                         \
  case (DTYPE): {                                                                \
    uint32_t result = SDBBExt2Compute<TYPE>(CTX);                                \
    if (result != KERNEL_STATUS_OK) {                                            \
      KERNEL_LOG_ERROR("SampleDistortedBoundingBoxExt2 kernel compute failed."); \
      return result;                                                             \
    }                                                                            \
    break;                                                                       \
  }
}  // namespace

namespace aicpu {
uint64_t SDBBExt2CpuKernel::New64() {
  std::random_device device("/dev/urandom");
  static std::mt19937_64 rng = std::mt19937_64(device());
  return (rng)();
}

void SDBBExt2CpuKernel::InitPhiloxRandom(int64_t seed, int64_t seed2) {
  if (seed == 0 && seed2 == 0) {
    seed = New64();
    seed2 = New64();
  }
  generator_ = random::PhiloxRandom(seed, seed2);
}

float SDBBExt2CpuKernel::RandFloat() {
  uint32_t x = GenerateSingle();
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;

  float result;
  auto ret = memcpy_s(&result, sizeof(result), &val, sizeof(val));
  if (ret != EOK) {
    KERNEL_LOG_ERROR("For 'SampleDistortedBoundingBoxExt2', memcpy_s failed, ret=%d.", ret);
  }
  return result - 1.0f;
}

uint32_t SDBBExt2CpuKernel::Uniform(uint32_t n) {
  if (n == 0) {
    return GenerateSingle() * n;
  } else if (0 == (n & (n - 1))) {
    return GenerateSingle() & (n - 1);
  } else {
    const uint32_t range = ~static_cast<uint32_t>(0);
    const uint32_t rem = (range % n) + 1;
    uint32_t rnd;
    do {
      rnd = GenerateSingle();
    } while (rnd < rem);
    return rnd % n;
  }
}

SDBBExt2CpuKernel::ResultElementType SDBBExt2CpuKernel::GenerateSingle() {
  if (used_result_index_ == random::PhiloxRandom::kResultElementCount) {
    unused_results_ = generator_();
    used_result_index_ = 0;
  }
  return unused_results_[used_result_index_++];
}

bool SDBBExt2CpuKernel::SatisfiesOverlapConstraints(const Rectangle &crop, float minimum_object_covered,
                                                    const std::vector<Rectangle> &bounding_boxes) {
  const float kMinArea = 1.0;
  if (crop.Area() < kMinArea) {
    return false;
  }

  bool is_object_covered = false;
  for (const auto &bbox : bounding_boxes) {
    const float object_area = bbox.Area();
    if (object_area < kMinArea) {
      continue;
    }

    if (object_area == 0) {
      continue;
    }
    const float object_covered = crop.Intersect(bbox).Area() / object_area;
    if (object_covered >= minimum_object_covered) {
      is_object_covered = true;
      break;
    }
  }
  return is_object_covered;
}

bool SDBBExt2CpuKernel::GenerateRandomCrop(int original_width, int original_height, float min_relative_crop_area,
                                           float max_relative_crop_area, float aspect_ratio, Rectangle *crop_rect) {
  if (max_relative_crop_area <= 0.0 || aspect_ratio <= 0.0 || original_width <= 0 || original_height <= 0 ||
      min_relative_crop_area > max_relative_crop_area) {
    return false;
  }

  const float min_area = min_relative_crop_area * original_width * original_height;
  const float max_area = max_relative_crop_area * original_width * original_height;

  int height = static_cast<int>(lrintf(std::sqrt(min_area / aspect_ratio)));

  int max_height = static_cast<int>(lrintf(std::sqrt(max_area / aspect_ratio)));
  if (lrintf(max_height * aspect_ratio) > original_width) {
    const float kEps = 0.0000001;
    const float kBias = 0.5;

    max_height = static_cast<int>((original_width + kBias - kEps) / aspect_ratio);
    if (lrintf(max_height * aspect_ratio) > original_width) {
      max_height -= 1;
    }
  }

  if (max_height > original_height) {
    max_height = original_height;
  }

  if (height >= max_height) {
    height = max_height;
  }

  if (height < max_height) {
    height += Uniform(max_height - height + 1);
  }
  int width = static_cast<int>(lrintf(height * aspect_ratio));
  float area = static_cast<float>(width * height);
  if (area < min_area) {
    height += 1;
    width = static_cast<int>(lrintf(height * aspect_ratio));
    area = width * height;
  }

  if (area > max_area) {
    height -= 1;
    width = static_cast<int>(lrintf(height * aspect_ratio));
    area = width * height;
  }

  if (area < min_area || area > max_area || width > original_width || height > original_height || width <= 0 ||
      height <= 0) {
    return false;
  }

  int y = 0;
  if (height < original_height) {
    y = Uniform(original_height - height);
  }
  int x = 0;
  if (width < original_width) {
    x = Uniform(original_width - width);
  }

  crop_rect->min_x_ = x;
  crop_rect->min_y_ = y;
  crop_rect->max_x_ = x + width;
  crop_rect->max_y_ = y + height;
  return true;
}

uint32_t SDBBExt2CpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SampleDistortedBoundingBoxExt2 check input and output number failed.");
  KERNEL_HANDLE_ERROR(SDBBExt2Check(ctx), "SampleDistortedBoundingBoxExt2 check params or bcast failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SDBBExt2CpuKernel_COMPUTE_CASE(DT_UINT8, uint8_t, ctx) SDBBExt2CpuKernel_COMPUTE_CASE(DT_INT8, int8_t, ctx)
      SDBBExt2CpuKernel_COMPUTE_CASE(DT_INT16, int16_t, ctx) SDBBExt2CpuKernel_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        SDBBExt2CpuKernel_COMPUTE_CASE(DT_INT64, int64_t, ctx) default
        : KERNEL_LOG_ERROR("SampleDistortedBoundingBoxExt2 kernel data type [%s] not support.",
                           DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SDBBExt2CpuKernel::SDBBExt2Check(const CpuKernelContext &ctx) {
  auto image_size = ctx.Input(0);
  auto bounding_boxes = ctx.Input(1);
  auto min_object_covered = ctx.Input(2);
  auto begin = ctx.Output(0);
  auto size = ctx.Output(1);
  auto bboxes = ctx.Output(2);
  KERNEL_CHECK_NULLPTR(image_size->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(bounding_boxes->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(min_object_covered->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 2 data failed.")
  KERNEL_CHECK_NULLPTR(begin->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed")
  KERNEL_CHECK_NULLPTR(size->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 1 data failed")
  KERNEL_CHECK_NULLPTR(bboxes->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 2 data failed")

  auto attr_seed = ctx.GetAttr("seed");
  KERNEL_CHECK_NULLPTR(attr_seed, KERNEL_STATUS_PARAM_INVALID, "Get seed attr failed.")
  seed = attr_seed->GetInt();

  auto attr_seed2 = ctx.GetAttr("seed2");
  KERNEL_CHECK_NULLPTR(attr_seed2, KERNEL_STATUS_PARAM_INVALID, "Get seed2 attr failed.")
  seed2 = attr_seed2->GetInt();

  auto attr_aspect_ratio_range = ctx.GetAttr("aspect_ratio_range");
  KERNEL_CHECK_NULLPTR(attr_aspect_ratio_range, KERNEL_STATUS_PARAM_INVALID, "Get aspect_ratio_range attr failed.")
  aspect_ratio_range = attr_aspect_ratio_range->GetListFloat();

  auto attr_area_range = ctx.GetAttr("area_range");
  KERNEL_CHECK_NULLPTR(attr_area_range, KERNEL_STATUS_PARAM_INVALID, "Get area_range attr failed.")
  area_range = attr_area_range->GetListFloat();

  auto attr_max_attempts = ctx.GetAttr("max_attempts");
  KERNEL_CHECK_NULLPTR(attr_max_attempts, KERNEL_STATUS_PARAM_INVALID, "Get max_attempts attr failed.")
  max_attempts = attr_max_attempts->GetInt();

  auto attr_use_image_if_no_bounding_boxes = ctx.GetAttr("use_image_if_no_bounding_boxes");
  KERNEL_CHECK_NULLPTR(attr_use_image_if_no_bounding_boxes, KERNEL_STATUS_PARAM_INVALID,
                       "Get use_image_if_no_bounding_boxes attr failed.")
  use_image_if_no_bounding_boxes = attr_use_image_if_no_bounding_boxes->GetBool();

  KERNEL_CHECK_NULLPTR(image_size->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input image_size shape failed.")
  KERNEL_CHECK_NULLPTR(bounding_boxes->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input bounding_boxes shape failed.")
  KERNEL_CHECK_NULLPTR(min_object_covered->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input min_object_covered shape failed.")

  std::vector<int64_t> shape_image_size = image_size->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_bounding_boxes = bounding_boxes->GetTensorShape()->GetDimSizes();

  KERNEL_CHECK_FALSE((shape_image_size.size() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "image_size must be 1-dimensional, got: [%d].", shape_image_size.size())
  KERNEL_CHECK_FALSE((shape_image_size.at(0) == IMAGE_SIZE_NUM), KERNEL_STATUS_PARAM_INVALID,
                     "image_size must contain 3 elements, got: [%d].", shape_image_size.size())

  KERNEL_CHECK_FALSE((shape_bounding_boxes.size() == SHAPE_BOUNDING_BOXES_SIZE), KERNEL_STATUS_PARAM_INVALID,
                     "input boxes must be 3-dimensional [batch, num_boxes, "
                     "coords], got: [%d].",
                     shape_bounding_boxes.size())

  KERNEL_CHECK_FALSE((shape_bounding_boxes.at(shape_bounding_boxes.size() - 1) == BOUNDING_BOXES_SIZE),
                     KERNEL_STATUS_PARAM_INVALID, "bounding boxes must have shape [4], got: [%d].",
                     shape_bounding_boxes.at(shape_bounding_boxes.size() - 1))

  const int aspect_ratio_range_size = 2;
  KERNEL_CHECK_FALSE((aspect_ratio_range.size() == aspect_ratio_range_size), KERNEL_STATUS_PARAM_INVALID,
                     "Aspect ratio range field must specify 2 dimensions.")
  KERNEL_CHECK_FALSE((aspect_ratio_range[0] > 0 && aspect_ratio_range[1] > 0), KERNEL_STATUS_PARAM_INVALID,
                     "Aspect ratio range must be positive: [%f], [%f].", aspect_ratio_range[0], aspect_ratio_range[1])

  const int area_range_size = 2;
  KERNEL_CHECK_FALSE((area_range.size() == area_range_size), KERNEL_STATUS_PARAM_INVALID,
                     "Area range field must specify 2 dimensions.")
  KERNEL_CHECK_FALSE((area_range[0] > 0 && area_range[1] > 0), KERNEL_STATUS_PARAM_INVALID,
                     "Area range must be positive: [%f], [%f].", area_range[0], area_range[1])
  KERNEL_CHECK_FALSE((area_range[0] <= 1 && area_range[1] <= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Area range must be less then or equal to 1.0: [%f], [%f].", area_range[0], area_range[1])

  KERNEL_CHECK_FALSE((max_attempts > 0), KERNEL_STATUS_PARAM_INVALID, "Max attempts must be positive: [%d]",
                     max_attempts)
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SDBBExt2CpuKernel::SDBBExt2Compute(CpuKernelContext &ctx) {
  auto image_size = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto bounding_boxes = reinterpret_cast<float *>(ctx.Input(1)->GetData());
  auto min_object_covered = reinterpret_cast<float *>(ctx.Input(2)->GetData());
  auto begin = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto size = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto bboxes = reinterpret_cast<float *>(ctx.Output(2)->GetData());

  const int32_t height = static_cast<int32_t>(image_size[0]);
  const int32_t width = static_cast<int32_t>(image_size[1]);
  if (!(height > 0 && width > 0)) {
    KERNEL_LOG_ERROR("Image height and width must be positive, got: [%d] and [%d]", height, width);
    return KERNEL_STATUS_INNER_ERROR;
  }
  float min_object_covered_val = 0.0;
  min_object_covered_val = *min_object_covered;
  if (min_object_covered_val < 0.0 || min_object_covered_val > 1.0) {
    KERNEL_LOG_ERROR("min_object_covered must be in [0.0, 1.0], got: [%f]", min_object_covered_val);
    return KERNEL_STATUS_INNER_ERROR;
  }
  const int index_y_min = 0;
  const int index_x_min = 1;
  const int index_y_max = 2;
  const int index_x_max = 3;
  const int kBBoxSize = 4;
  std::vector<Rectangle> boxes;
  int64_t size_bounding_boxes = ctx.Input(1)->NumElements();
  if (size_bounding_boxes > 0) {
    for (int b = 0; b < size_bounding_boxes / kBBoxSize; ++b) {
      if (!(bounding_boxes[b * kBBoxSize + index_x_min] < bounding_boxes[b * kBBoxSize + index_x_max])) {
        KERNEL_LOG_ERROR("x_min must be less than x_max, got: [%f] and [%f]",
                         bounding_boxes[b * kBBoxSize + index_x_min], bounding_boxes[b * kBBoxSize + index_x_max]);
        return KERNEL_STATUS_INNER_ERROR;
      }
      if (!(bounding_boxes[b * kBBoxSize + index_y_min] < bounding_boxes[b * kBBoxSize + index_y_max])) {
        KERNEL_LOG_ERROR("y_min must be less than y_max, got: [%f] and [%f]",
                         bounding_boxes[b * kBBoxSize + index_y_min], bounding_boxes[b * kBBoxSize + index_y_max]);
        return KERNEL_STATUS_INNER_ERROR;
      }
      for (int i = 0; i < kBBoxSize; ++i) {
        if (bounding_boxes[b * kBBoxSize + i] < 0.0 || bounding_boxes[b * kBBoxSize + i] > 1.0) {
          KERNEL_LOG_ERROR("All bounding box coordinates must be in [0.0, 1.0], got: [%f]",
                           bounding_boxes[b * kBBoxSize + i]);
          return KERNEL_STATUS_INNER_ERROR;
        }
      }
      const int32_t x_min = static_cast<int32_t>(bounding_boxes[b * kBBoxSize + index_x_min] * width);
      const int32_t y_min = static_cast<int32_t>(bounding_boxes[b * kBBoxSize + index_y_min] * height);
      const int32_t x_max = static_cast<int32_t>(bounding_boxes[b * kBBoxSize + index_x_max] * width);
      const int32_t y_max = static_cast<int32_t>(bounding_boxes[b * kBBoxSize + index_y_max] * height);
      boxes.push_back(Rectangle(x_min, y_min, x_max, y_max));
    }
  }

  const Rectangle image_rect(0, 0, width, height);
  if (boxes.empty()) {
    if (!use_image_if_no_bounding_boxes) {
      KERNEL_LOG_ERROR(
        "No bounding boxes provided as input. One must "
        "enable use_image_if_no_bounding_boxes if you wish "
        "to not provide any bounding boxes.");
      return KERNEL_STATUS_INNER_ERROR;
    }

    boxes.push_back(image_rect);
  }

  const float min_sample_area = area_range[0];
  const float max_sample_area = area_range[1];
  const float min_sample_aspect_ratio = aspect_ratio_range[0];
  const float max_sample_aspect_ratio = aspect_ratio_range[1];

  InitPhiloxRandom(seed, seed2);

  Rectangle crop_rect;
  bool sample_generated = false;
  for (int i = 0; i < max_attempts; ++i) {
    const float sample_aspect_ratio =
      RandFloat() * (max_sample_aspect_ratio - min_sample_aspect_ratio) + min_sample_aspect_ratio;
    if (GenerateRandomCrop(width, height, min_sample_area, max_sample_area, sample_aspect_ratio, &crop_rect)) {
      if (SatisfiesOverlapConstraints(crop_rect, min_object_covered_val, boxes)) {
        sample_generated = true;
        break;
      }
    }
  }

  if (!sample_generated) {
    crop_rect = image_rect;
  }

  // Determine the cropping parameters from the bounding box.
  const int target_width = crop_rect.max_x_ - crop_rect.min_x_;
  const int target_height = crop_rect.max_y_ - crop_rect.min_y_;
  const int offset_width = crop_rect.min_x_;
  const int offset_height = crop_rect.min_y_;

  if (width < target_width + offset_width) {
    KERNEL_LOG_ERROR("width must be >= target_width + offset_width: [%d] vs [%d] + [%d]", width, target_width,
                     offset_width);
    return KERNEL_STATUS_INNER_ERROR;
  }

  if (height < target_height + offset_height) {
    KERNEL_LOG_ERROR("height must be >= target_height + offset_height: [%d] vs [%d] + [%d]", height, target_height,
                     offset_height);
    return KERNEL_STATUS_INNER_ERROR;
  }

  begin[0] = static_cast<T>(offset_height);
  size[0] = static_cast<T>(target_height);
  begin[1] = static_cast<T>(offset_width);
  size[1] = static_cast<T>(target_width);

  bboxes[index_y_min] = static_cast<float>(crop_rect.min_y_) / static_cast<float>(height);
  bboxes[index_x_min] = static_cast<float>(crop_rect.min_x_) / static_cast<float>(width);
  bboxes[index_y_max] = static_cast<float>(crop_rect.max_y_) / static_cast<float>(height);
  bboxes[index_x_max] = static_cast<float>(crop_rect.max_x_) / static_cast<float>(width);

  // Retain all of the channels.
  const int32_t begin_channels = 3;
  const int32_t size_channels = 3;
  begin[begin_channels - 1] = static_cast<T>(0);
  size[size_channels - 1] = static_cast<T>(-1);

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSDBBExt2, SDBBExt2CpuKernel);
}  // namespace aicpu
