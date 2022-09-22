/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/proposal.h"
#include <cmath>
#include <cstring>
#include <memory>
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace proposal {
constexpr int kNumInput2 = 2;
constexpr int kScoreSizeIndex = 2;
constexpr int kKeyConfidenceIndex = 4;
constexpr int kPredWeightIndex = 2;
constexpr int KPredHeightIndex = 3;
uint32_t RpnTmpBufSize(uint32_t num_ratio_anchors, uint32_t num_scale_anchors, uint32_t input_height,
                       uint32_t input_width) {
  uint32_t anchors_num = num_ratio_anchors * num_scale_anchors * input_height * input_width;
  uint32_t anchors_size = sizeof(uint32_t) * COORDI_NUM * anchors_num;
  uint32_t bbox_delta_size = anchors_size;
  uint32_t proposal_size = sizeof(uint32_t) * PROPOSAL_WIDTH * anchors_num;
  uint32_t ratio_anchors_size = sizeof(float) * num_ratio_anchors * COORDI_NUM;
  uint32_t scale_anchors_size = sizeof(float) * num_ratio_anchors * num_scale_anchors * COORDI_NUM;
  uint32_t score_size = sizeof(float) * anchors_num * kScoreSizeIndex;
  uint32_t stack_size = sizeof(Stack) * anchors_num;
  uint32_t total_size =
    anchors_size + bbox_delta_size + proposal_size + ratio_anchors_size + scale_anchors_size + score_size + stack_size;
  return total_size;
}

static float exp_coef[10][16] = {
  {1.0f, 1.00024f, 1.00049f, 1.00073f, 1.00098f, 1.00122f, 1.00147f, 1.00171f, 1.00196f, 1.0022f, 1.00244f, 1.00269f,
   1.00293f, 1.00318f, 1.00342f, 1.00367f},
  {1.0f, 1.00391f, 1.00784f, 1.01179f, 1.01575f, 1.01972f, 1.02371f, 1.02772f, 1.03174f, 1.03578f, 1.03984f, 1.04391f,
   1.04799f, 1.05209f, 1.05621f, 1.06034f},
  {1.0f, 1.06449f, 1.13315f, 1.20623f, 1.28403f, 1.36684f, 1.45499f, 1.54883f, 1.64872f, 1.75505f, 1.86825f, 1.98874f,
   2.117f, 2.25353f, 2.39888f, 2.55359f},
  {1.0f, 2.71828f, 7.38906f, 20.0855f, 54.5981f, 148.413f, 403.429f, 1096.63f, 2980.96f, 8103.08f, 22026.5f, 59874.1f,
   162755.0f, 442413.0f, 1.2026e+006f, 3.26902e+006f},
  {1.0f, 8.88611e+006f, 7.8963e+013f, 7.01674e+020f, 6.23515e+027f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f,
   5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f, 5.54062e+034f,
   5.54062e+034f},
  {1.0f, 0.999756f, 0.999512f, 0.999268f, 0.999024f, 0.99878f, 0.998536f, 0.998292f, 0.998049f, 0.997805f, 0.997562f,
   0.997318f, 0.997075f, 0.996831f, 0.996588f, 0.996345f},
  {1.0f, 0.996101f, 0.992218f, 0.98835f, 0.984496f, 0.980658f, 0.976835f, 0.973027f, 0.969233f, 0.965455f, 0.961691f,
   0.957941f, 0.954207f, 0.950487f, 0.946781f, 0.94309f},
  {1.0f, 0.939413f, 0.882497f, 0.829029f, 0.778801f, 0.731616f, 0.687289f, 0.645649f, 0.606531f, 0.569783f, 0.535261f,
   0.502832f, 0.472367f, 0.443747f, 0.416862f, 0.391606f},
  {1.0f, 0.367879f, 0.135335f, 0.0497871f, 0.0183156f, 0.00673795f, 0.00247875f, 0.000911882f, 0.000335463f,
   0.00012341f, 4.53999e-005f, 1.67017e-005f, 6.14421e-006f, 2.26033e-006f, 8.31529e-007f, 3.05902e-007f},
  {1.0f, 1.12535e-007f, 1.26642e-014f, 1.42516e-021f, 1.60381e-028f, 1.80485e-035f, 2.03048e-042f, 0.0f, 0.0f, 0.0f,
   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
static float QuickExp(int32_t value) {
  if (value & 0x80000000) {
    value = ~value + 0x00000001;
    return exp_coef[5][value & 0x0000000F] * exp_coef[6][(value >> 4) & 0x0000000F] *
           exp_coef[7][(value >> 8) & 0x0000000F] * exp_coef[8][(value >> 12) & 0x0000000F] *
           exp_coef[9][(value >> 16) & 0x0000000F];
  } else {
    return exp_coef[0][value & 0x0000000F] * exp_coef[1][(value >> 4) & 0x0000000F] *
           exp_coef[2][(value >> 8) & 0x0000000F] * exp_coef[3][(value >> 12) & 0x0000000F] *
           exp_coef[4][(value >> 16) & 0x0000000F];
  }
}

static int32_t SoftMax(float *src, uint32_t num) {
  float max = 0;
  float sum = 0;
  uint32_t i = 0;

  for (i = 0; i < num; ++i) {
    if (max < src[i]) {
      max = src[i];
    }
  }

  for (i = 0; i < num; ++i) {
    src[i] = QuickExp(static_cast<int32_t>((src[i] - max) * QUANT_BASE));
    sum += src[i];
  }

  for (i = 0; i < num; ++i) {
    src[i] /= sum;
  }
  return RET_OK;
}
static void Argswap(int32_t *src1, int32_t *src2) {
  for (uint32_t i = 0; i < PROPOSAL_WIDTH; i++) {
    int32_t tmp = src1[i];
    src1[i] = src2[i];
    src2[i] = tmp;
  }
}

static int32_t NonRecursiveArgQuickSort(int32_t *array, int32_t low, int32_t high, Stack *stack, int32_t max_num) {
  int32_t top = 0;
  stack[top].min_ = low;
  stack[top].max_ = high;

  while (top > -1) {
    low = stack[top].min_;
    high = stack[top].max_;
    int32_t i = low;
    int32_t j = high;

    int32_t key_confidence = array[PROPOSAL_WIDTH * low + kKeyConfidenceIndex];
    top--;
    while (i < j) {
      while ((i < j) && (key_confidence > array[j * PROPOSAL_WIDTH + 4])) {
        j--;
      }
      if (i < j) {
        Argswap(&array[i * PROPOSAL_WIDTH], &array[j * PROPOSAL_WIDTH]);
        i++;
      }

      while ((i < j) && (key_confidence < array[i * PROPOSAL_WIDTH + 4])) {
        i++;
      }
      if (i < j) {
        Argswap(&array[i * PROPOSAL_WIDTH], &array[j * PROPOSAL_WIDTH]);
        j--;
      }
    }

    if (low <= max_num) {
      if (low < i - 1) {
        top++;
        stack[top].min_ = low;
        stack[top].max_ = i - 1;
      }

      if (high > i + 1) {
        top++;
        stack[top].min_ = i + 1;
        stack[top].max_ = high;
      }
    }
  }
  return RET_OK;
}

static int32_t FilterLowScoreBbox(int32_t *proposals, uint32_t anchors_num, uint32_t filter_thresh,
                                  uint32_t *num_after_filter) {
  if (proposals == nullptr) {
    LOGE("inputs proposals is nullptr");
    return RET_ERROR;
  }
  uint32_t proposal_cnt = anchors_num;

  if (filter_thresh > 0) {
    uint32_t i;
    for (i = 0; i < anchors_num; i++) {
      if (proposals[PROPOSAL_WIDTH * i + 4] < static_cast<int32_t>(filter_thresh)) {
        proposals[PROPOSAL_WIDTH * i + 5] = 1;
      }
    }

    proposal_cnt = 0;
    for (i = 0; i < anchors_num; i++) {
      if (proposals[PROPOSAL_WIDTH * i + 5] == 0) {
        proposals[PROPOSAL_WIDTH * proposal_cnt] = proposals[PROPOSAL_WIDTH * i];
        proposals[PROPOSAL_WIDTH * proposal_cnt + 1] = proposals[PROPOSAL_WIDTH * i + 1];
        proposals[PROPOSAL_WIDTH * proposal_cnt + 2] = proposals[PROPOSAL_WIDTH * i + 2];
        proposals[PROPOSAL_WIDTH * proposal_cnt + 3] = proposals[PROPOSAL_WIDTH * i + 3];
        proposals[PROPOSAL_WIDTH * proposal_cnt + 4] = proposals[PROPOSAL_WIDTH * i + 4];
        proposals[PROPOSAL_WIDTH * proposal_cnt + 5] = proposals[PROPOSAL_WIDTH * i + 5];
        proposal_cnt++;
      }
    }
  }
  *num_after_filter = proposal_cnt;
  return RET_OK;
}

static int32_t SVP_NNIE_Overlap(int32_t x_min1, int32_t y_min1, int32_t x_max1, int32_t y_max1, int32_t x_min2,
                                int32_t y_min2, int32_t x_max2, int32_t y_max2, int32_t *area_sum,
                                int32_t *area_inter) {
  /*** Check the input, and change the Return value  ***/
  int32_t inter = 0;
  int32_t total = 0;
  int32_t x_min = 0;
  int32_t y_min = 0;
  int32_t x_max = 0;
  int32_t y_max = 0;
  int32_t area1 = 0;
  int32_t area2 = 0;
  int32_t inter_width = 0;
  int32_t inter_height = 0;

  x_min = MAX(x_min1, x_min2);
  y_min = MAX(y_min1, y_min2);
  x_max = MIN(x_max1, x_max2);
  y_max = MIN(y_max1, y_max2);

  inter_width = x_max - x_min + 1;
  inter_height = y_max - y_min + 1;

  inter_width = (inter_width >= 0) ? inter_width : 0;
  inter_height = (inter_height >= 0) ? inter_height : 0;

  inter = inter_width * inter_height;
  area1 = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1);
  area2 = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1);

  total = area1 + area2 - inter;

  *area_sum = total;
  *area_inter = inter;
  return RET_OK;
}

static int32_t SVP_NNIE_NonMaxSuppression(int32_t *proposals, uint32_t anchors_num, uint32_t nms_thresh,
                                          uint32_t max_roi_num) {
  if (proposals == nullptr) {
    LOGE("inputs proposals is nullptr");
    return RET_ERROR;
  }
  /****** define variables *******/
  int32_t x_min1;
  int32_t y_min1;
  int32_t x_max1;
  int32_t y_max1;
  int32_t x_min2;
  int32_t y_min2;
  int32_t x_max2;
  int32_t y_max2;
  int32_t s32AreaTotal = 0;
  int32_t area_inter = 0;
  uint32_t i;
  uint32_t j;
  uint32_t num = 0;
  bool bNoOverlap;
  for (i = 0; i < anchors_num && num < max_roi_num; i++) {
    if (proposals[PROPOSAL_WIDTH * i + 5] == 0) {
      num++;
      x_min1 = proposals[PROPOSAL_WIDTH * i];
      y_min1 = proposals[PROPOSAL_WIDTH * i + 1];
      x_max1 = proposals[PROPOSAL_WIDTH * i + 2];
      y_max1 = proposals[PROPOSAL_WIDTH * i + 3];
      for (j = i + 1; j < anchors_num; j++) {
        if (proposals[PROPOSAL_WIDTH * j + 5] == 0) {
          x_min2 = proposals[PROPOSAL_WIDTH * j];
          y_min2 = proposals[PROPOSAL_WIDTH * j + 1];
          x_max2 = proposals[PROPOSAL_WIDTH * j + 2];
          y_max2 = proposals[PROPOSAL_WIDTH * j + 3];
          bNoOverlap = (x_min2 > x_max1) || (x_max2 < x_min1) || (y_min2 > y_max1) || (y_max2 < y_min1);
          if (bNoOverlap) {
            continue;
          }
          (void)SVP_NNIE_Overlap(x_min1, y_min1, x_max1, y_max1, x_min2, y_min2, x_max2, y_max2, &s32AreaTotal,
                                 &area_inter);
          if (area_inter * QUANT_BASE > static_cast<int32_t>(nms_thresh * s32AreaTotal)) {
            if (proposals[PROPOSAL_WIDTH * i + 4] >= proposals[PROPOSAL_WIDTH * j + 4]) {
              proposals[PROPOSAL_WIDTH * j + 5] = 1;
            } else {
              proposals[PROPOSAL_WIDTH * i + 5] = 1;
            }
          }
        }
      }
    }
  }
  return RET_OK;
}

static void Rpn(float **inputs, uint32_t num_ratio_anchors, uint32_t num_scale_anchors, uint32_t *scales,
                uint32_t *ratios, uint32_t ori_image_height, uint32_t ori_image_width, uint32_t *inputs_height,
                uint32_t *inputs_width, uint32_t *inputs_channel, uint32_t inputs_stride, uint32_t max_rois,
                uint32_t min_size, uint32_t spatial_scale, uint32_t nms_thresh, uint32_t filter_thresh,
                uint32_t num_before_nms, char *pu32MemPool, float *proposal_result, uint32_t dst_stride,
                uint32_t *num_rois) {
  /******************** define parameters ****************/
  uint32_t size;
  int32_t *anchors = nullptr;
  int32_t *bbox_delta = nullptr;
  int32_t *proposals = nullptr;
  int32_t *ptr1 = nullptr;
  int32_t *ptr2 = nullptr;
  int32_t *ptr3 = nullptr;
  uint32_t num_after_filter = 0;
  uint32_t num_anchors;
  float base_w;
  float base_h;
  float base_x_ctr;
  float base_y_ctr;
  float *ratio_anchors = nullptr;
  float *f32_ptr = nullptr;
  float *f32_ptr2 = nullptr;
  float *scale_anchors = nullptr;
  float *scores = nullptr;
  float f32_size;
  uint32_t pixel_interval;
  uint32_t src_bbox_index;
  uint32_t src_fg_prob_index;
  uint32_t src_bg_prob_index;
  uint32_t src_bbox_bias;
  uint32_t src_prob_bias;
  uint32_t des_box;
  uint32_t bg_blob_size;
  uint32_t anchors_per_pixel;
  uint32_t map_size;
  uint32_t line_size;
  int32_t proposal_width;
  int32_t proposal_height;
  uint32_t roi_count;
  Stack *stack = nullptr;
  uint32_t c;
  uint32_t h;
  uint32_t w;
  uint32_t i;
  uint32_t j;
  uint32_t p;
  uint32_t q;
  uint32_t z;
  uint32_t base_anchor[4] = {0, 0, (min_size - 1), (min_size - 1)};

  /*********************************** Faster RCNN *********************************************/
  /********* calculate the start pointer of each part in MemPool *********/
  anchors = reinterpret_cast<int32_t *>(pu32MemPool);
  num_anchors = num_ratio_anchors * num_scale_anchors * (inputs_height[0] * inputs_width[0]);
  size = COORDI_NUM * num_anchors;
  pu32MemPool += size * sizeof(int32_t);

  bbox_delta = reinterpret_cast<int32_t *>(pu32MemPool);
  pu32MemPool += size * sizeof(int32_t);

  proposals = reinterpret_cast<int32_t *>(pu32MemPool);
  size = PROPOSAL_WIDTH * num_anchors;
  pu32MemPool += size * sizeof(int32_t);

  ratio_anchors = reinterpret_cast<float *>(static_cast<void *>(pu32MemPool));
  f32_ptr = reinterpret_cast<float *>(static_cast<void *>(pu32MemPool));
  size = num_ratio_anchors * COORDI_NUM;
  f32_ptr = f32_ptr + size;

  scale_anchors = f32_ptr;
  size = num_scale_anchors * num_ratio_anchors * COORDI_NUM;
  f32_ptr = f32_ptr + size;

  scores = f32_ptr;
  size = num_anchors * SCORE_NUM;
  f32_ptr = f32_ptr + size;

  stack = reinterpret_cast<Stack *>(f32_ptr);

  /********************* Generate the base anchor ***********************/
  base_w = static_cast<float>(base_anchor[2] - base_anchor[0] + 1);
  base_h = static_cast<float>(base_anchor[3] - base_anchor[1] + 1);
  base_x_ctr = static_cast<float>(base_anchor[0] + ((base_w - 1) * 0.5));
  base_y_ctr = static_cast<float>(base_anchor[1] + ((base_h - 1) * 0.5));

  /*************** Generate Ratio Anchors for the base anchor ***********/
  f32_ptr = ratio_anchors;
  f32_size = base_w * base_h;
  for (i = 0; i < num_ratio_anchors; i++) {
    float f32_ratios = static_cast<float>(ratios[i]) / QUANT_BASE;
    base_w = sqrt(f32_size / f32_ratios);
    base_w = static_cast<float>(
      1.0 * ((base_w) >= 0 ? static_cast<int32_t>(base_w + HALF_VAL) : static_cast<int32_t>(base_w - HALF_VAL)));
    base_h = base_w * f32_ratios;
    base_h = static_cast<float>(
      1.0 * ((base_h) >= 0 ? static_cast<int32_t>(base_h + HALF_VAL) : static_cast<int32_t>(base_h - HALF_VAL)));

    *f32_ptr++ = static_cast<float>(base_x_ctr - ((base_w - 1) * HALF_VAL));
    *(f32_ptr++) = static_cast<float>(base_y_ctr - ((base_h - 1) * HALF_VAL));
    *(f32_ptr++) = static_cast<float>(base_x_ctr + ((base_w - 1) * HALF_VAL));
    *(f32_ptr++) = static_cast<float>(base_y_ctr + ((base_h - 1) * HALF_VAL));
  }

  /********* Generate Scale Anchors for each Ratio Anchor **********/
  f32_ptr = ratio_anchors;
  f32_ptr2 = scale_anchors;
  /* Generate Scale Anchors for one pixel */
  for (i = 0; i < num_ratio_anchors; i++) {
    for (j = 0; j < num_scale_anchors; j++) {
      base_w = *(f32_ptr + 2) - *(f32_ptr) + 1;
      base_h = *(f32_ptr + 3) - *(f32_ptr + 1) + 1;
      base_x_ctr = static_cast<float>(*(f32_ptr) + ((base_w - 1) * HALF_VAL));
      base_y_ctr = static_cast<float>(*(f32_ptr + 1) + ((base_h - 1) * HALF_VAL));

      *(f32_ptr2++) =
        static_cast<float>(base_x_ctr - ((base_w * (static_cast<float>(scales[j]) / QUANT_BASE) - 1) * HALF_VAL));
      *(f32_ptr2++) =
        static_cast<float>(base_y_ctr - ((base_h * (static_cast<float>(scales[j]) / QUANT_BASE) - 1) * HALF_VAL));
      *(f32_ptr2++) =
        static_cast<float>(base_x_ctr + ((base_w * (static_cast<float>(scales[j]) / QUANT_BASE) - 1) * HALF_VAL));
      *(f32_ptr2++) =
        static_cast<float>(base_y_ctr + ((base_h * (static_cast<float>(scales[j]) / QUANT_BASE) - 1) * HALF_VAL));
    }
    f32_ptr += COORDI_NUM;
  }

  /******************* Copy the anchors to every pixel in the feature map ******************/
  ptr1 = anchors;
  if (spatial_scale == 0) {
    LOGE("inputs spatial_scale is zero.");
    return;
  }
  pixel_interval = QUANT_BASE / spatial_scale;

  for (p = 0; p < inputs_height[0]; p++) {
    for (q = 0; q < inputs_width[0]; q++) {
      f32_ptr2 = scale_anchors;
      for (z = 0; z < num_scale_anchors * num_ratio_anchors; z++) {
        *(ptr1++) = static_cast<int32_t>(q * pixel_interval + *(f32_ptr2++));
        *(ptr1++) = static_cast<int32_t>(p * pixel_interval + *(f32_ptr2++));
        *(ptr1++) = static_cast<int32_t>(q * pixel_interval + *(f32_ptr2++));
        *(ptr1++) = static_cast<int32_t>(p * pixel_interval + *(f32_ptr2++));
      }
    }
  }

  /********** do transpose, convert the blob from (M,C,H,W) to (M,H,W,C) **********/
  map_size = inputs_height[1] * inputs_stride / sizeof(uint32_t);
  anchors_per_pixel = num_ratio_anchors * num_scale_anchors;
  bg_blob_size = anchors_per_pixel * map_size;
  line_size = inputs_stride / sizeof(uint32_t);
  src_prob_bias = 0;
  src_bbox_bias = 0;

  for (c = 0; c < inputs_channel[1]; c++) {
    for (h = 0; h < inputs_height[1]; h++) {
      for (w = 0; w < inputs_width[1]; w++) {
        src_bbox_index = src_bbox_bias + c * map_size + h * line_size + w;
        src_bg_prob_index = src_prob_bias + (c / COORDI_NUM) * map_size + h * line_size + w;
        src_fg_prob_index = bg_blob_size + src_bg_prob_index;

        des_box = (anchors_per_pixel) * (h * inputs_width[1] + w) + c / COORDI_NUM;

        uint32_t des_bbox_delta_index = COORDI_NUM * des_box + c % COORDI_NUM;
        bbox_delta[des_bbox_delta_index] = static_cast<int32_t>(inputs[1][src_bbox_index] * QUANT_BASE);

        uint32_t des_score_index = (SCORE_NUM)*des_box;
        scores[des_score_index] = inputs[0][src_bg_prob_index];
        scores[des_score_index + 1] = inputs[0][src_fg_prob_index];
      }
    }
  }

  /************************* do softmax ****************************/
  f32_ptr = scores;
  for (i = 0; i < num_anchors; i++) {
    SoftMax(f32_ptr, SCORE_NUM);
    f32_ptr += SCORE_NUM;
  }

  /************************* BBox Transform *****************************/
  for (i = 0; i < num_anchors; i++) {
    ptr1 = anchors;
    ptr1 = ptr1 + COORDI_NUM * i;
    ptr2 = proposals;
    ptr2 = ptr2 + PROPOSAL_WIDTH * i;
    ptr3 = bbox_delta;
    ptr3 = ptr3 + COORDI_NUM * i;
    f32_ptr = scores;
    f32_ptr = f32_ptr + i * (SCORE_NUM);

    proposal_width = *(ptr1 + 2) - *(ptr1) + 1;
    proposal_height = *(ptr1 + 3) - *(ptr1 + 1) + 1;
    int32_t proposal_center_x = *(ptr1) + static_cast<int32_t>(proposal_width * HALF_VAL);
    int32_t proposal_center_y = *(ptr1 + 1) + static_cast<int32_t>(proposal_height * HALF_VAL);
    int32_t pred_center_x =
      static_cast<int32_t>((static_cast<float>(*(ptr3)) / QUANT_BASE) * proposal_width + proposal_center_x);
    int32_t pred_center_y =
      static_cast<int32_t>((static_cast<float>(*(ptr3 + 1)) / QUANT_BASE) * proposal_height + proposal_center_y);

    int32_t pred_w = static_cast<int32_t>(proposal_width * QuickExp(static_cast<int32_t>(*(ptr3 + kPredWeightIndex))));
    int32_t pred_h = static_cast<int32_t>(proposal_height * QuickExp(static_cast<int32_t>(*(ptr3 + KPredHeightIndex))));
    *(ptr2) = static_cast<int32_t>(pred_center_x - HALF_VAL * pred_w);
    *(ptr2 + 1) = static_cast<int32_t>(pred_center_y - HALF_VAL * pred_h);
    *(ptr2 + 2) = static_cast<int32_t>(pred_center_x + HALF_VAL * pred_w);
    *(ptr2 + 3) = static_cast<int32_t>(pred_center_y + HALF_VAL * pred_h);
    *(ptr2 + 4) = static_cast<int32_t>(*(f32_ptr + 1) * QUANT_BASE);
    *(ptr2 + 5) = 0;
  }

  /************************ clip bbox *****************************/
  for (i = 0; i < num_anchors; i++) {
    ptr1 = proposals;
    ptr1 = ptr1 + PROPOSAL_WIDTH * i;
    *ptr1 = MAX(MIN(*ptr1, static_cast<int32_t>(ori_image_width) - 1), 0);
    *(ptr1 + 1) = MAX(MIN(*(ptr1 + 1), static_cast<int32_t>(ori_image_height) - 1), 0);
    *(ptr1 + 2) = MAX(MIN(*(ptr1 + 2), static_cast<int32_t>(ori_image_width) - 1), 0);
    *(ptr1 + 3) = MAX(MIN(*(ptr1 + 3), static_cast<int32_t>(ori_image_height) - 1), 0);
  }

  /************ remove the bboxes which are too small *************/
  for (i = 0; i < num_anchors; i++) {
    ptr1 = proposals;
    ptr1 = ptr1 + PROPOSAL_WIDTH * i;
    proposal_width = *(ptr1 + 2) - *(ptr1) + 1;
    proposal_height = *(ptr1 + 3) - *(ptr1 + 1) + 1;
    if (proposal_width < static_cast<int32_t>(min_size) || proposal_height < static_cast<int32_t>(min_size)) {
      *(ptr1 + 5) = 1;
    }
  }

  /********** remove low score bboxes ************/
  (void)FilterLowScoreBbox(proposals, num_anchors, filter_thresh, &num_after_filter);

  /********** sort ***********/
  (void)NonRecursiveArgQuickSort(proposals, 0, num_after_filter - 1, stack, static_cast<int32_t>(num_before_nms));
  num_after_filter = (num_after_filter < num_before_nms) ? num_after_filter : num_before_nms;

  /* do nms to remove highly overlapped bbox */
  (void)SVP_NNIE_NonMaxSuppression(proposals, num_after_filter, nms_thresh, max_rois); /* function NMS */

  /************** write the final result to output ***************/
  roi_count = 0;
  for (i = 0; i < num_after_filter; i++) {
    ptr1 = proposals;
    ptr1 = ptr1 + PROPOSAL_WIDTH * i;
    if (*(ptr1 + 5) == 0) {
      proposal_result[dst_stride / sizeof(uint32_t) * roi_count] = *ptr1;
      proposal_result[dst_stride / sizeof(uint32_t) * roi_count + 1] = *(ptr1 + 1);
      proposal_result[dst_stride / sizeof(uint32_t) * roi_count + 2] = *(ptr1 + 2);
      proposal_result[dst_stride / sizeof(uint32_t) * roi_count + 3] = *(ptr1 + 3);
      roi_count++;
    }
    if (roi_count >= max_rois) {
      break;
    }
  }

  *num_rois = roi_count;
}

int32_t ProposalInit(ProposalParam *param, uint32_t max_roi_num, uint32_t ori_image_height, uint32_t ori_image_width) {
  uint32_t tmp_buf_size = 0;
  uint32_t bbox_buf_size = 0;
  uint32_t total_size = 0;
  param->max_roi_num_ = max_roi_num;

  param->num_ratio_anchors_ = 1;
  param->num_scale_anchors_ = NUM_SCALE_ANCHORS;
  param->scales_[0] = 1.5 * QUANT_BASE;
  param->scales_[1] = 2.1 * QUANT_BASE;
  param->scales_[2] = 2.9 * QUANT_BASE;
  param->scales_[3] = 4.1 * QUANT_BASE;
  param->scales_[4] = 5.8 * QUANT_BASE;
  param->scales_[5] = 8.0 * QUANT_BASE;
  param->scales_[6] = 11.3 * QUANT_BASE;
  param->scales_[7] = 15.8 * QUANT_BASE;
  param->scales_[8] = 22.1 * QUANT_BASE;
  param->ratios_[0] = 2.44 * QUANT_BASE;

  param->ori_image_height_ = ori_image_height;
  param->ori_image_width_ = ori_image_width;
  param->min_size_ = MIN_SIZE;
  param->spatial_scale_ = (uint32_t)(0.0625 * QUANT_BASE);
  param->nms_thresh_ = (uint32_t)(0.7 * QUANT_BASE);
  param->filter_thresh_ = 0;
  param->num_before_nms_ = NUM_NMS;

  param->rpn_bounding_box_.chn_ = 1;
  param->rpn_bounding_box_.height_ = max_roi_num;
  param->rpn_bounding_box_.width_ = COORDI_NUM;
  param->rpn_bounding_box_.stride_ = COORDI_NUM * sizeof(float);
  param->rpn_bounding_box_.num_ = 1;

  tmp_buf_size = RpnTmpBufSize(param->num_ratio_anchors_, param->num_scale_anchors_, param->inputs_height_[0],
                               param->inputs_width_[0]);

  bbox_buf_size = param->rpn_bounding_box_.num_ * param->rpn_bounding_box_.height_ * param->rpn_bounding_box_.stride_;
  total_size = tmp_buf_size + bbox_buf_size;

  if (param->rpn_tmp_buf_ != nullptr) {
    free(param->rpn_tmp_buf_);
    param->rpn_tmp_buf_ = nullptr;
  }
  param->rpn_tmp_buf_ = malloc(total_size);
  if (param->rpn_tmp_buf_ == nullptr) {
    LOGE("malloc buf fail.");
    return RET_ERROR;
  }
  param->rpn_bounding_box_.data_ = reinterpret_cast<char *>(param->rpn_tmp_buf_) + tmp_buf_size;

  return RET_OK;
}

int32_t ProposalRun(ProposalParam *param) {
  for (int i = 0; i < kNumInput2; i++) {
    if (param->inputs_[i] == nullptr) {
      LOGE("inputs is nullptr.");
      return RET_ERROR;
    }
  }
  Rpn(param->inputs_, param->num_ratio_anchors_, param->num_scale_anchors_, param->scales_, param->ratios_,
      param->ori_image_height_, param->ori_image_width_, param->inputs_height_, param->inputs_width_,
      param->inputs_channel_, param->inputs_stride_, param->max_roi_num_, param->min_size_, param->spatial_scale_,
      param->nms_thresh_, param->filter_thresh_, param->num_before_nms_, reinterpret_cast<char *>(param->rpn_tmp_buf_),
      reinterpret_cast<float *>(param->rpn_bounding_box_.data_), param->rpn_bounding_box_.stride_,
      &param->rpn_bounding_box_.height_);
  return RET_OK;
}

void ProposalDeInit(ProposalParam *param) {
  if (param->rpn_tmp_buf_ != 0) {
    free(param->rpn_tmp_buf_);
    param->rpn_tmp_buf_ = 0;
  }
}
}  // namespace proposal
}  // namespace mindspore
