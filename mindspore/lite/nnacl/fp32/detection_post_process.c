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

#include "nnacl/fp32/detection_post_process.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int ScoreWithIndexCmp(const void *a, const void *b) {
  ScoreWithIndex *pa = (ScoreWithIndex *)a;
  ScoreWithIndex *pb = (ScoreWithIndex *)b;
  if (pa->score > pb->score) {
    return -1;
  } else if (pa->score < pb->score) {
    return 1;
  } else {
    return 0;
  }
}

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

void DecodeBoxes(const int num_boxes, const float *input_boxes, const float *anchors, const BboxCenter scaler,
                 float *decoded_boxes) {
  for (int i = 0; i < num_boxes; ++i) {
    BboxCenter *box = (BboxCenter *)(input_boxes) + i;
    BboxCenter *anchor = (BboxCenter *)(anchors) + i;
    BboxCorner *decoded_box = (BboxCorner *)(decoded_boxes) + i;
    float y_center = box->y / scaler.y * anchor->h + anchor->y;
    float x_center = box->x / scaler.x * anchor->w + anchor->x;
    float h_half = 0.5f * expf(box->h / scaler.h) * anchor->h;
    float w_half = 0.5f * expf(box->w / scaler.w) * anchor->w;
    decoded_box->ymin = y_center - h_half;
    decoded_box->xmin = x_center - w_half;
    decoded_box->ymax = y_center + h_half;
    decoded_box->xmax = x_center + w_half;
  }
}

int NmsSingleClass(const int candidate_num, const float *decoded_boxes, const int max_detections,
                   ScoreWithIndex *score_with_index, int *selected, const DetectionPostProcessParameter *param) {
  uint8_t *nms_candidate = param->nms_candidate_;
  const int output_num = candidate_num < max_detections ? candidate_num : max_detections;
  int possible_candidate_num = candidate_num;
  int selected_num = 0;
  qsort(score_with_index, candidate_num, sizeof(ScoreWithIndex), ScoreWithIndexCmp);
  for (int i = 0; i < candidate_num; ++i) {
    nms_candidate[i] = 1;
  }
  for (int i = 0; i < candidate_num; ++i) {
    if (possible_candidate_num == 0 || selected_num >= output_num) {
      break;
    }
    if (nms_candidate[i] == 0) {
      continue;
    }
    selected[selected_num++] = score_with_index[i].index;
    nms_candidate[i] = 0;
    possible_candidate_num--;
    for (int t = i + 1; t < candidate_num; ++t) {
      if (nms_candidate[t] == 1) {
        const BboxCorner *bbox_i = (BboxCorner *)(decoded_boxes) + score_with_index[i].index;
        const BboxCorner *bbox_t = (BboxCorner *)(decoded_boxes) + score_with_index[t].index;
        const float iou = IntersectionOverUnion(bbox_i, bbox_t);
        if (iou > param->nms_iou_threshold_) {
          nms_candidate[t] = 0;
          possible_candidate_num--;
        }
      }
    }
  }
  return selected_num;
}

int NmsMultiClassesRegular(const int num_boxes, const int num_classes_with_bg, const float *decoded_boxes,
                           const float *input_scores, float *output_boxes, float *output_classes, float *output_scores,
                           const DetectionPostProcessParameter *param) {
  const int first_class_index = num_classes_with_bg - (int)(param->num_classes_);
  int *selected = (int *)(param->selected_);
  ScoreWithIndex *score_with_index_single = (ScoreWithIndex *)(param->score_with_class_);
  int all_classes_sorted_num = 0;
  int all_classes_output_num = 0;
  ScoreWithIndex *score_with_index_all = (ScoreWithIndex *)(param->score_with_class_all_);
  for (int j = first_class_index; j < num_classes_with_bg; ++j) {
    int candidate_num = 0;
    // process single class
    for (int i = 0; i < num_boxes; ++i) {
      const float score = input_scores[i * num_classes_with_bg + j];
      if (score >= param->nms_score_threshold_) {
        score_with_index_single[candidate_num].score = score;
        score_with_index_single[candidate_num++].index = i;
      }
    }
    int selected_num = NmsSingleClass(candidate_num, decoded_boxes, param->detections_per_class_,
                                      score_with_index_single, selected, param);
    // process all classes
    for (int i = 0; i < selected_num; ++i) {
      // store class to index
      score_with_index_all[all_classes_sorted_num].index = selected[i] * num_classes_with_bg + j;
      score_with_index_all[all_classes_sorted_num++].score = input_scores[selected[i] * num_classes_with_bg + j];
    }
    all_classes_output_num =
      all_classes_sorted_num < param->max_detections_ ? all_classes_sorted_num : param->max_detections_;
    qsort(score_with_index_all, all_classes_sorted_num, sizeof(ScoreWithIndex), ScoreWithIndexCmp);
    all_classes_sorted_num = all_classes_output_num;
  }
  for (int i = 0; i < param->max_detections_ * param->max_classes_per_detection_; ++i) {
    if (i < all_classes_output_num) {
      const int box_index = score_with_index_all[i].index / num_classes_with_bg;
      const int class_index = score_with_index_all[i].index - box_index * num_classes_with_bg - first_class_index;
      *((BboxCorner *)(output_boxes) + i) = *((BboxCorner *)(decoded_boxes) + box_index);
      output_classes[i] = (float)class_index;
      output_scores[i] = score_with_index_all[i].score;
    } else {
      ((BboxCorner *)(output_boxes) + i)->ymin = 0;
      ((BboxCorner *)(output_boxes) + i)->xmin = 0;
      ((BboxCorner *)(output_boxes) + i)->ymax = 0;
      ((BboxCorner *)(output_boxes) + i)->xmax = 0;
      output_classes[i] = 0.0f;
      output_scores[i] = 0.0f;
    }
  }
  return all_classes_output_num;
}

int NmsMultiClassesFast(const int num_boxes, const int num_classes_with_bg, const float *decoded_boxes,
                        const float *input_scores, float *output_boxes, float *output_classes, float *output_scores,
                        const DetectionPostProcessParameter *param) {
  const int first_class_index = num_classes_with_bg - (int)(param->num_classes_);
  const int64_t max_classes_per_anchor =
    param->max_classes_per_detection_ < param->num_classes_ ? param->max_classes_per_detection_ : param->num_classes_;
  int candidate_num = 0;
  ScoreWithIndex *score_with_class_all = (ScoreWithIndex *)(param->score_with_class_all_);
  ScoreWithIndex *score_with_class = (ScoreWithIndex *)(param->score_with_class_);
  int *selected = (int *)(param->selected_);
  int selected_num;
  int output_num = 0;
  for (int i = 0; i < num_boxes; ++i) {
    for (int j = first_class_index; j < num_classes_with_bg; ++j) {
      float score_t = *(input_scores + i * num_classes_with_bg + j);
      score_with_class_all[i * param->num_classes_ + j - first_class_index].score = score_t;
      // save box and class info to index
      score_with_class_all[i * param->num_classes_ + j - first_class_index].index = i * num_classes_with_bg + j;
    }
    qsort(score_with_class_all + i * param->num_classes_, param->num_classes_, sizeof(ScoreWithIndex),
          ScoreWithIndexCmp);
    const float score_max = (score_with_class_all + i * param->num_classes_)->score;
    if (score_max >= param->nms_score_threshold_) {
      score_with_class[candidate_num].index = i;
      score_with_class[candidate_num++].score = score_max;
    }
  }
  selected_num =
    NmsSingleClass(candidate_num, decoded_boxes, param->max_detections_, score_with_class, selected, param);
  for (int i = 0; i < selected_num; ++i) {
    const ScoreWithIndex *box_score_with_class = score_with_class_all + selected[i] * param->num_classes_;
    const int box_index = box_score_with_class->index / num_classes_with_bg;
    for (int j = 0; j < max_classes_per_anchor; ++j) {
      *((BboxCorner *)(output_boxes) + output_num) = *((BboxCorner *)(decoded_boxes) + box_index);
      output_scores[output_num] = (box_score_with_class + j)->score;
      output_classes[output_num++] =
        (float)((box_score_with_class + j)->index % num_classes_with_bg - first_class_index);
    }
  }
  for (int i = output_num; i < param->max_detections_ * param->max_classes_per_detection_; ++i) {
    ((BboxCorner *)(output_boxes) + i)->ymin = 0;
    ((BboxCorner *)(output_boxes) + i)->xmin = 0;
    ((BboxCorner *)(output_boxes) + i)->ymax = 0;
    ((BboxCorner *)(output_boxes) + i)->xmax = 0;
    output_scores[i] = 0;
    output_classes[i] = 0;
  }
  return output_num;
}

int DetectionPostProcess(const int num_boxes, const int num_classes_with_bg, float *input_boxes, float *input_scores,
                         float *input_anchors, float *output_boxes, float *output_classes, float *output_scores,
                         float *output_num, DetectionPostProcessParameter *param) {
  BboxCenter scaler;
  scaler.y = param->y_scale_;
  scaler.x = param->x_scale_;
  scaler.h = param->h_scale_;
  scaler.w = param->w_scale_;
  DecodeBoxes(num_boxes, input_boxes, input_anchors, scaler, param->decoded_boxes_);
  if (param->use_regular_nms_) {
    *output_num = NmsMultiClassesRegular(num_boxes, num_classes_with_bg, param->decoded_boxes_, input_scores,
                                         output_boxes, output_classes, output_scores, param);
  } else {
    *output_num = NmsMultiClassesFast(num_boxes, num_classes_with_bg, param->decoded_boxes_, input_scores, output_boxes,
                                      output_classes, output_scores, param);
  }
  return NNACL_OK;
}
