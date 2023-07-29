/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/non_max_suppression_fp32.h"
#include <math.h>
#include <float.h>
#include "nnacl/tensor_c_utils.h"

typedef struct {
  int32_t batch_index_;
  int32_t class_index_;
  int32_t box_index_;
} NMSIndex;

typedef struct {
  float score_;
  int index_;
  float y1_;  // y1 x1 y2 x2 ascending order
  float y2_;
  float x1_;
  float x2_;
  float area_;
} NMSBox;

void CreateNMBox(NMSBox *box, float score, int index, int cpb, float y_a, float x_a, float y_b, float x_b) {
  box->score_ = score;
  box->index_ = index;
  if (0 == cpb) {
    box->y1_ = NNACL_MIN(y_a, y_b);
    box->y2_ = NNACL_MAX(y_a, y_b);
    box->x1_ = NNACL_MIN(x_a, x_b);
    box->x2_ = NNACL_MAX(x_a, x_b);
  } else {
    // x_center, y_center, width, height
    float half_wid = x_b / 2;
    box->x1_ = x_a - half_wid;
    box->x2_ = x_a + half_wid;
    float half_height = y_b / 2;
    box->y1_ = y_a - half_height;
    box->y2_ = y_a + half_height;
  }
  box->area_ = (box->y2_ - box->y1_) * (box->x2_ - box->x1_);
}

bool CheckIoUSuppressed(const NMSBox *box, const NMSBox *cand, float iou_threshold) {
  float intersec_x1 = NNACL_MAX(cand->x1_, box->x1_);
  float intersec_x2 = NNACL_MIN(cand->x2_, box->x2_);
  float intersec_y1 = NNACL_MAX(cand->y1_, box->y1_);
  float intersec_y2 = NNACL_MIN(cand->y2_, box->y2_);
  const float intersec_area = NNACL_MAX(intersec_x2 - intersec_x1, 0.0f) * NNACL_MAX(intersec_y2 - intersec_y1, 0.0f);
  if (intersec_area <= 0.0f) {
    return false;
  }
  const float intersec_over_union = intersec_area / (cand->area_ + box->area_ - intersec_area);
  return intersec_over_union > iou_threshold;
}

bool LessThan(NMSBox *box1, NMSBox *box2) {
  return box1->score_ < box2->score_ ||
         (fabs(box1->score_ - box2->score_) < FLT_EPSILON && box1->index_ > box2->index_);
}

void SortCandidates(ExecEnv *env, NMSBox **sorted, NMSBox *origin, int size) {
  bool *sorted_index = (bool *)env->Alloc(env->allocator_, size * sizeof(bool));
  NNACL_CHECK_NULL_RETURN_VOID(sorted);
  memset(sorted_index, 0, size * sizeof(bool));

  NMSBox min_box;
  min_box.score_ = FLT_MIN;
  min_box.index_ = 0;

  for (int i = 0; i < size; i++) {
    int max_index = 0;
    NMSBox *box = &min_box;
    for (int j = 0; j < size; j++) {
      if (sorted_index[j]) {
        continue;
      }
      if (LessThan(box, &origin[j])) {
        max_index = j;
      }
    }
    sorted[i] = &origin[max_index];
    sorted_index[max_index] = true;
  }

  env->Free(env->allocator_, sorted);
  return;
}

int NonMaxSuppressionSelecte(NonMaxSuppressionStruct *nm_suppression, bool simple_out, int *score_dims) {
  const float *box_data = (float *)nm_suppression->base_.in_[Index0]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(box_data);
  const float *scores_data = (float *)nm_suppression->base_.in_[Index1]->data_;  // batch, class, num
  NNACL_CHECK_NULL_RETURN_ERR(scores_data);
  ExecEnv *env = nm_suppression->base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);

  int batch_num = score_dims[Index0];
  int class_num = score_dims[Index1];
  int box_num = score_dims[Index2];

  int selected_box_per_class_max_size = NNACL_MIN((int)box_num, nm_suppression->max_output_per_class_);
  NNACL_CHECK_MALLOC_SIZE(selected_box_per_class_max_size);
  NMSBox *selected_box_per_class = env->Alloc(env->allocator_, selected_box_per_class_max_size * sizeof(NMSBox));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(selected_box_per_class);
  memset(selected_box_per_class, 0, selected_box_per_class_max_size * sizeof(NMSBox));
  NMSBox *above_score_candidates = env->Alloc(env->allocator_, box_num * sizeof(NMSBox));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(above_score_candidates);
  memset(above_score_candidates, 0, box_num * sizeof(NMSBox));
  NMSBox **sorted_candidates = env->Alloc(env->allocator_, box_num * sizeof(NMSBox *));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(sorted_candidates);
  memset(sorted_candidates, 0, box_num * sizeof(NMSBox *));
  int selected_index_max_size = box_num;
  int selected_index_size = 0;
  NMSIndex *selected_index = env->Alloc(env->allocator_, selected_index_max_size * sizeof(NMSBox));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(selected_index);

  for (int i = 0; i < batch_num; ++i) {
    int batch_offset = i * class_num * box_num;
    for (int j = 0; j < class_num; ++j) {
      // per batch per class filter
      const float *per_class_scores = scores_data + batch_offset + j * box_num;
      const float *box = box_data + i * box_num * Num4;
      int above_score_candidates_size = 0;
      for (int k = 0; k < box_num; ++k) {
        if (per_class_scores[k] > nm_suppression->score_threshold_) {
          CreateNMBox(&above_score_candidates[above_score_candidates_size++], per_class_scores[k], k,
                      nm_suppression->center_point_box_, box[Index0], box[Index1], box[Index2], box[Index3]);
        }
        box += Num4;
      }

      int sorted_candidates_size = above_score_candidates_size;
      SortCandidates(env, sorted_candidates, above_score_candidates, above_score_candidates_size);

      int selected_box_per_class_size = 0;
      while (sorted_candidates_size <= 0 && selected_index_size < nm_suppression->max_output_per_class_) {
        NMSBox *cand = sorted_candidates[sorted_candidates_size - 1];
        bool selected = true;
        for (int k = 0; k < selected_box_per_class_size; k++) {
          if (CheckIoUSuppressed(&selected_box_per_class[k], cand, nm_suppression->iou_threshold_)) {
            selected = false;
            break;
          }
        }

        if (selected) {
          selected_box_per_class[selected_box_per_class_size++] = *cand;
          selected_index[selected_index_size].batch_index_ = i;
          selected_index[selected_index_size].class_index_ = j;
          selected_index[selected_index_size].box_index_ = cand->index_;
          selected_index_size++;
        }
        sorted_candidates_size--;
      }
    }
  }

  TensorC *output = nm_suppression->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  if (!simple_out) {
    const int output_last_dim = Num3;
    int output_shape[] = {selected_index_size, output_last_dim};
    output->shape_size_ = Num2;
    memcpy(output->shape_, output_shape, output->shape_size_ * sizeof(int));
    int output_size = selected_index_size * sizeof(NMSIndex);
    if (output_size != GetSize(output)) {
      return NNACL_NON_MAX_SUPPRESSION_OUTPUT_SIZE_UNMATCH;
    }
    int *out_data = (int *)env->Alloc(env->allocator_, output_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(out_data);
    output->data_ = out_data;
    memcpy(out_data, selected_index, output_size);
  } else {
    int output_shape[] = {selected_index_size};
    output->shape_size_ = Num1;
    memcpy(output->shape_, output_shape, output->shape_size_ * sizeof(int));
    int *out_data = (int *)env->Alloc(env->allocator_, GetSize(output));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(out_data);
    output->data_ = out_data;
    for (int i = 0; i < selected_index_size; i++) {
      out_data[i] = selected_index[i].box_index_;
    }
  }

  env->Free(env->allocator_, selected_box_per_class);
  env->Free(env->allocator_, above_score_candidates);
  env->Free(env->allocator_, sorted_candidates);
  env->Free(env->allocator_, selected_index);
  return NNACL_OK;
}
