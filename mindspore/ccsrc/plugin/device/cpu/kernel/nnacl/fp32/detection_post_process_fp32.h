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

#ifndef MINDSPORE_NNACL_FP32_DETECTION_POST_PROCESS_H_
#define MINDSPORE_NNACL_FP32_DETECTION_POST_PROCESS_H_

#include "nnacl/op_base.h"
#include "nnacl/detection_post_process_parameter.h"

typedef struct {
  float y;
  float x;
  float h;
  float w;
} BboxCenter;

typedef struct {
  float ymin;
  float xmin;
  float ymax;
  float xmax;
} BboxCorner;

#ifdef __cplusplus
extern "C" {
#endif
int DecodeBoxes(int num_boxes, const float *input_boxes, const float *anchors,
                const DetectionPostProcessParameter *param);

int NmsMultiClassesFastCore(const int num_boxes, const int num_classes_with_bg, const float *input_scores,
                            void (*)(const float *, int *, int, int), const DetectionPostProcessParameter *param,
                            const int task_id, const int thread_num);

int DetectionPostProcessFast(const int num_boxes, const int num_classes_with_bg, const float *input_scores,
                             const float *decoded_boxes, float *output_boxes, float *output_classes,
                             float *output_scores, float *output_num, void (*)(const float *, int *, int, int),
                             const DetectionPostProcessParameter *param);

int DetectionPostProcessRegular(const int num_boxes, const int num_classes_with_bg, const float *input_scores,
                                float *output_boxes, float *output_classes, float *output_scores, float *output_num,
                                void (*)(const float *, int *, int, int), const DetectionPostProcessParameter *param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_DETECTION_POST_PROCESS_H_
