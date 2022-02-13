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

#include "nnacl/fp32/detection_post_process_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/nnacl_utils.h"

float IntersectionOverUnion(const BboxCorner *a, const BboxCorner *b) {
  const float area_a = (a->ymax - a->ymin) * (a->xmax - a->xmin);
  const float area_b = (b->ymax - b->ymin) * (b->xmax - b->xmin);
  if (area_a <= 0 || area_b <= 0) {
    return 0.0f;
  }
  const float ymin = a->ymin > b->ymin ? a->ymin : b->ymin;
  const float xmin = a->xmin > b->xmin ? a->xmin : b->xmin;
  const float ymax = a->ymax < b->ymax ? a->ymax : b->ymax;
  const float xmax = a->xmax < b->xmax ? a->xmax : b->xmax;
  const float h = ymax - ymin > 0.0f ? ymax - ymin : 0.0f;
  const float w = xmax - xmin > 0.0f ? xmax - xmin : 0.0f;
  const float inter = h * w;
  return inter / (area_a + area_b - inter);
}

int DecodeBoxes(int num_boxes, const float *input_boxes, const float *anchors,
                const DetectionPostProcessParameter *param) {
  if (input_boxes == NULL || anchors == NULL || param == NULL) {
    return NNACL_NULL_PTR;
  }
  float *decoded_boxes = (float *)param->decoded_boxes_;
  BboxCenter scaler;
  scaler.y = param->y_scale_;
  scaler.x = param->x_scale_;
  scaler.h = param->h_scale_;
  scaler.w = param->w_scale_;
  for (int i = 0; i < num_boxes; ++i) {
    BboxCenter *box = (BboxCenter *)(input_boxes) + i;
    BboxCenter *anchor = (BboxCenter *)(anchors) + i;
    BboxCorner *decoded_box = (BboxCorner *)(decoded_boxes) + i;
    float y_center = box->y / scaler.y * anchor->h + anchor->y;
    float x_center = box->x / scaler.x * anchor->w + anchor->x;
    const float h_half = 0.5f * expf(box->h / scaler.h) * anchor->h;
    const float w_half = 0.5f * expf(box->w / scaler.w) * anchor->w;
    decoded_box->ymin = y_center - h_half;
    decoded_box->xmin = x_center - w_half;
    decoded_box->ymax = y_center + h_half;
    decoded_box->xmax = x_center + w_half;
  }
  return NNACL_OK;
}

int NmsSingleClass(const int num_boxes, const float *decoded_boxes, const int max_detections, const float *scores,
                   int *selected, void (*PartialArgSort)(const float *, int *, int, int),
                   const DetectionPostProcessParameter *param) {
  if (PartialArgSort == NULL) {
    return NNACL_NULL_PTR;
  }
  uint8_t *nms_candidate = param->nms_candidate_;
  const int output_num = num_boxes < max_detections ? num_boxes : max_detections;
  int possible_candidate_num = num_boxes;
  int selected_num = 0;
  int *indexes = (int *)param->single_class_indexes_;
  for (int i = 0; i < num_boxes; ++i) {
    indexes[i] = i;
    nms_candidate[i] = 1;
  }
  PartialArgSort(scores, indexes, num_boxes, num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    if (possible_candidate_num == 0 || selected_num >= output_num || scores[indexes[i]] < param->nms_score_threshold_) {
      break;
    }
    if (nms_candidate[indexes[i]] == 0) {
      continue;
    }
    selected[selected_num++] = indexes[i];
    nms_candidate[indexes[i]] = 0;
    possible_candidate_num--;
    const BboxCorner *bbox_i = (BboxCorner *)(decoded_boxes) + indexes[i];
    for (int t = i + 1; t < num_boxes; ++t) {
      if (scores[indexes[t]] < param->nms_score_threshold_) break;
      if (nms_candidate[indexes[t]] == 1) {
        const BboxCorner *bbox_t = (BboxCorner *)(decoded_boxes) + indexes[t];
        const float iou = IntersectionOverUnion(bbox_i, bbox_t);
        if (iou > param->nms_iou_threshold_) {
          nms_candidate[indexes[t]] = 0;
          possible_candidate_num--;
        }
      }
    }
  }
  return selected_num;
}

int NmsMultiClassesFastCore(const int num_boxes, const int num_classes_with_bg, const float *input_scores,
                            void (*PartialArgSort)(const float *, int *, int, int),
                            const DetectionPostProcessParameter *param, const int task_id, const int thread_num) {
  if (input_scores == NULL || param == NULL || PartialArgSort == NULL) {
    return NNACL_NULL_PTR;
  }
  if (thread_num == 0) {
    return NNACL_PARAM_INVALID;
  }
  const int first_class_index = num_classes_with_bg - (int)(param->num_classes_);
  const int64_t max_classes_per_anchor =
    param->max_classes_per_detection_ < param->num_classes_ ? param->max_classes_per_detection_ : param->num_classes_;
  float *scores = (float *)param->scores_;
  for (int i = task_id; i < num_boxes; i += thread_num) {
    int *indexes = (int *)param->indexes_ + i * param->num_classes_;
    for (int j = 0; j < param->num_classes_; ++j) {
      indexes[j] = i * num_classes_with_bg + first_class_index + j;
    }
    PartialArgSort(input_scores, indexes, max_classes_per_anchor, param->num_classes_);
    scores[i] = input_scores[indexes[0]];
  }
  return NNACL_OK;
}

int DetectionPostProcessFast(const int num_boxes, const int num_classes_with_bg, const float *input_scores,
                             const float *decoded_boxes, float *output_boxes, float *output_classes,
                             float *output_scores, float *output_num,
                             void (*PartialArgSort)(const float *, int *, int, int),
                             const DetectionPostProcessParameter *param) {
  if (input_scores == NULL || decoded_boxes == NULL || output_boxes == NULL || output_classes == NULL ||
      output_scores == NULL || output_num == NULL || param == NULL || PartialArgSort == NULL) {
    return NNACL_NULL_PTR;
  }
  int out_num = 0;
  const int first_class_index = num_classes_with_bg - (int)(param->num_classes_);
  const int64_t max_classes_per_anchor =
    param->max_classes_per_detection_ < param->num_classes_ ? param->max_classes_per_detection_ : param->num_classes_;
  int *selected = (int *)param->selected_;
  int selected_num = NmsSingleClass(num_boxes, decoded_boxes, param->max_detections_, (float *)param->scores_, selected,
                                    PartialArgSort, param);
  for (int i = 0; i < selected_num; ++i) {
    int *indexes = (int *)param->indexes_ + selected[i] * param->num_classes_;
    BboxCorner *box = (BboxCorner *)(decoded_boxes) + selected[i];
    for (int j = 0; j < max_classes_per_anchor; ++j) {
      *((BboxCorner *)(output_boxes) + out_num) = *box;
      output_scores[out_num] = input_scores[indexes[j]];
      NNACL_ASSERT(num_classes_with_bg != 0);
      output_classes[out_num++] = (float)(indexes[j] % num_classes_with_bg - first_class_index);
    }
  }
  *output_num = (float)out_num;
  for (int i = out_num; i < param->max_detections_ * param->max_classes_per_detection_; ++i) {
    ((BboxCorner *)(output_boxes) + i)->ymin = 0;
    ((BboxCorner *)(output_boxes) + i)->xmin = 0;
    ((BboxCorner *)(output_boxes) + i)->ymax = 0;
    ((BboxCorner *)(output_boxes) + i)->xmax = 0;
    output_scores[i] = 0;
    output_classes[i] = 0;
  }
  return NNACL_OK;
}

int DetectionPostProcessRegular(const int num_boxes, const int num_classes_with_bg, const float *input_scores,
                                float *output_boxes, float *output_classes, float *output_scores, float *output_num,
                                void (*PartialArgSort)(const float *, int *, int, int),
                                const DetectionPostProcessParameter *param) {
  if (input_scores == NULL || output_boxes == NULL || output_classes == NULL || output_scores == NULL ||
      output_num == NULL || param == NULL || PartialArgSort == NULL) {
    return NNACL_NULL_PTR;
  }
  const int first_class_index = num_classes_with_bg - (int)(param->num_classes_);
  float *decoded_boxes = (float *)param->decoded_boxes_;
  int *selected = (int *)param->selected_;
  float *scores = (float *)param->scores_;
  float *all_scores = (float *)param->all_class_scores_;
  int *indexes = (int *)(param->indexes_);
  int *all_indexes = (int *)(param->all_class_indexes_);
  int all_classes_sorted_num = 0;
  int all_classes_output_num = 0;
  for (int j = first_class_index; j < num_classes_with_bg; ++j) {
    // process single class
    for (int i = 0; i < num_boxes; ++i) {
      scores[i] = input_scores[i * num_classes_with_bg + j];
    }
    int selected_num =
      NmsSingleClass(num_boxes, decoded_boxes, param->detections_per_class_, scores, selected, PartialArgSort, param);
    for (int i = 0; i < all_classes_sorted_num; ++i) {
      indexes[i] = all_indexes[i];
      all_indexes[i] = i;
    }
    // process all classes
    for (int i = 0; i < selected_num; ++i) {
      indexes[all_classes_sorted_num] = selected[i] * num_classes_with_bg + j;
      all_indexes[all_classes_sorted_num] = all_classes_sorted_num;
      all_scores[all_classes_sorted_num++] = scores[selected[i]];
    }
    all_classes_output_num =
      all_classes_sorted_num < param->max_detections_ ? all_classes_sorted_num : param->max_detections_;
    PartialArgSort(all_scores, all_indexes, all_classes_output_num, all_classes_sorted_num);
    for (int i = 0; i < all_classes_output_num; ++i) {
      scores[i] = all_scores[all_indexes[i]];
      all_indexes[i] = indexes[all_indexes[i]];
    }
    for (int i = 0; i < all_classes_output_num; ++i) {
      all_scores[i] = scores[i];
    }
    all_classes_sorted_num = all_classes_output_num;
  }
  for (int i = 0; i < param->max_detections_ * param->max_classes_per_detection_; ++i) {
    if (i < all_classes_output_num) {
      NNACL_CHECK_ZERO_RETURN_ERR(num_classes_with_bg);
      const int box_index = all_indexes[i] / num_classes_with_bg;
      const int class_index = all_indexes[i] % num_classes_with_bg - first_class_index;
      *((BboxCorner *)(output_boxes) + i) = *((BboxCorner *)(decoded_boxes) + box_index);
      output_classes[i] = (float)class_index;
      output_scores[i] = all_scores[i];
    } else {
      ((BboxCorner *)(output_boxes) + i)->ymin = 0;
      ((BboxCorner *)(output_boxes) + i)->xmin = 0;
      ((BboxCorner *)(output_boxes) + i)->ymax = 0;
      ((BboxCorner *)(output_boxes) + i)->xmax = 0;
      output_classes[i] = 0.0f;
      output_scores[i] = 0.0f;
    }
  }
  *output_num = (float)all_classes_output_num;
  return NNACL_OK;
}
