/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/ocr_ops_declare.h"
#include <vector>

namespace mindspore::transform {
INPUT_MAP(OCRDetectionPreHandle) = {{1, INPUT_DESC(img)}};
ATTR_MAP(OCRDetectionPreHandle) = {{"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(OCRDetectionPreHandle) = {
  {0, OUTPUT_DESC(resized_img)}, {1, OUTPUT_DESC(h_scale)}, {2, OUTPUT_DESC(w_scale)}};
REG_ADPT_DESC(OCRDetectionPreHandle, kNameOCRDetectionPreHandle, ADPT_DESC(OCRDetectionPreHandle))

INPUT_MAP(OCRFindContours) = {{1, INPUT_DESC(img)}};
ATTR_MAP(OCRFindContours) = {{"value_mode", ATTR_DESC(value_mode, AnyTraits<int64_t>())}};
OUTPUT_MAP(OCRFindContours) = {
  {0, OUTPUT_DESC(polys_data)}, {1, OUTPUT_DESC(polys_offset)}, {2, OUTPUT_DESC(polys_size)}};
REG_ADPT_DESC(OCRFindContours, kNameOCRFindContours, ADPT_DESC(OCRFindContours))

INPUT_MAP(BatchDilatePolys) = {{1, INPUT_DESC(polys_data)}, {2, INPUT_DESC(polys_offset)},
                               {3, INPUT_DESC(polys_size)}, {4, INPUT_DESC(score)},
                               {5, INPUT_DESC(min_border)}, {6, INPUT_DESC(min_area_thr)},
                               {7, INPUT_DESC(score_thr)},  {8, INPUT_DESC(expands_cale)}};
ATTR_MAP(BatchDilatePolys) = EMPTY_ATTR_MAP;
OUTPUT_MAP(BatchDilatePolys) = {
  {0, OUTPUT_DESC(dilated_polys_data)}, {1, OUTPUT_DESC(dilated_polys_offset)}, {2, OUTPUT_DESC(dilated_polys_size)}};
REG_ADPT_DESC(BatchDilatePolys, kNameBatchDilatePolys, ADPT_DESC(BatchDilatePolys))

INPUT_MAP(ResizeAndClipPolys) = {
  {1, INPUT_DESC(polys_data)}, {2, INPUT_DESC(polys_offset)}, {3, INPUT_DESC(polys_size)}, {4, INPUT_DESC(h_scale)},
  {5, INPUT_DESC(w_scale)},    {6, INPUT_DESC(img_h)},        {7, INPUT_DESC(img_w)}};
ATTR_MAP(ResizeAndClipPolys) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ResizeAndClipPolys) = {{0, OUTPUT_DESC(clipped_polys_data)},
                                  {1, OUTPUT_DESC(clipped_polys_offset)},
                                  {2, OUTPUT_DESC(clipped_polys_size)},
                                  {3, OUTPUT_DESC(clipped_polys_num)}};
REG_ADPT_DESC(ResizeAndClipPolys, kNameResizeAndClipPolys, ADPT_DESC(ResizeAndClipPolys))

INPUT_MAP(OCRDetectionPostHandle) = {
  {1, INPUT_DESC(img)}, {2, INPUT_DESC(polys_data)}, {3, INPUT_DESC(polys_offset)}, {4, INPUT_DESC(polys_size)}};
ATTR_MAP(OCRDetectionPostHandle) = {{"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(OCRDetectionPostHandle) = {{0, OUTPUT_DESC(imgs_data)},
                                      {1, OUTPUT_DESC(imgs_offset)},
                                      {2, OUTPUT_DESC(imgs_size)},
                                      {3, OUTPUT_DESC(rect_points)}};
REG_ADPT_DESC(OCRDetectionPostHandle, kNameOCRDetectionPostHandle, ADPT_DESC(OCRDetectionPostHandle))

INPUT_MAP(OCRIdentifyPreHandle) = {
  {1, INPUT_DESC(imgs_data)}, {2, INPUT_DESC(imgs_offset)}, {3, INPUT_DESC(imgs_size)}};
ATTR_MAP(OCRIdentifyPreHandle) = {{"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>())},
                                  {"format", ATTR_DESC(data_format, AnyTraits<std::string>())}};
OUTPUT_MAP(OCRIdentifyPreHandle) = {{0, OUTPUT_DESC(resized_imgs)}};
REG_ADPT_DESC(OCRIdentifyPreHandle, kNameOCRIdentifyPreHandle, ADPT_DESC(OCRIdentifyPreHandle))

INPUT_MAP(BatchEnqueue) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(queue_id)}};
ATTR_MAP(BatchEnqueue) = {{"batch_size", ATTR_DESC(batch_size, AnyTraits<int64_t>())},
                          {"queue_name", ATTR_DESC(queue_name, AnyTraits<std::string>())},
                          {"queue_depth", ATTR_DESC(queue_depth, AnyTraits<int64_t>())},
                          {"pad_mode", ATTR_DESC(pad_mode, AnyTraits<std::string>())}};
OUTPUT_MAP(BatchEnqueue) = {{0, OUTPUT_DESC(enqueue_count)}};
REG_ADPT_DESC(BatchEnqueue, kNameBatchEnqueue, ADPT_DESC(BatchEnqueue))

INPUT_MAP(Dequeue) = {{1, INPUT_DESC(queue_id)}};
ATTR_MAP(Dequeue) = {{"output_type", ATTR_DESC(output_type, AnyTraits<GEType>())},
                     {"output_shape", ATTR_DESC(output_shape, AnyTraits<std::vector<int64_t>>())},
                     {"queue_name", ATTR_DESC(queue_name, AnyTraits<std::string>())}};
OUTPUT_MAP(Dequeue) = {{0, OUTPUT_DESC(data)}};
REG_ADPT_DESC(Dequeue, kNameDequeue, ADPT_DESC(Dequeue))

INPUT_MAP(OCRRecognitionPreHandle) = {{1, INPUT_DESC(imgs_data)},
                                      {2, INPUT_DESC(imgs_offset)},
                                      {3, INPUT_DESC(imgs_size)},
                                      {4, INPUT_DESC(langs)},
                                      {5, INPUT_DESC(langs_score)}};
ATTR_MAP(OCRRecognitionPreHandle) = {{"batch_size", ATTR_DESC(batch_size, AnyTraits<int64_t>())},
                                     {"format", ATTR_DESC(data_format, AnyTraits<std::string>())},
                                     {"pad_mode", ATTR_DESC(pad_mode, AnyTraits<std::string>())}};
OUTPUT_MAP(OCRRecognitionPreHandle) = {{0, OUTPUT_DESC(imgs)},
                                       {1, OUTPUT_DESC(imgs_relation)},
                                       {2, OUTPUT_DESC(imgs_lang)},
                                       {3, OUTPUT_DESC(imgs_piece_fillers)}};
REG_ADPT_DESC(OCRRecognitionPreHandle, kNameOCRRecognitionPreHandle, ADPT_DESC(OCRRecognitionPreHandle))
}  // namespace mindspore::transform
