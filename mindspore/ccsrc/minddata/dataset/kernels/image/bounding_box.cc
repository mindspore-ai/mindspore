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

#include "minddata/dataset/kernels/image/bounding_box.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {

const uint8_t kNumOfCols = 4;

BoundingBox::BoundingBox(bbox_float x, bbox_float y, bbox_float width, bbox_float height)
    : x_(x), y_(y), width_(width), height_(height) {}

Status BoundingBox::ReadFromTensor(const TensorPtr &bbox_tensor, dsize_t index_of_bbox,
                                   std::shared_ptr<BoundingBox> *bbox_out) {
  CHECK_FAIL_RETURN_UNEXPECTED(bbox_tensor != nullptr, "BoundingBox: bbox_tensor is null.");
  bbox_float x;
  bbox_float y;
  bbox_float width;
  bbox_float height;
  RETURN_IF_NOT_OK(bbox_tensor->GetItemAt<bbox_float>(&x, {index_of_bbox, 0}));
  RETURN_IF_NOT_OK(bbox_tensor->GetItemAt<bbox_float>(&y, {index_of_bbox, 1}));
  RETURN_IF_NOT_OK(bbox_tensor->GetItemAt<bbox_float>(&width, {index_of_bbox, 2}));
  RETURN_IF_NOT_OK(bbox_tensor->GetItemAt<bbox_float>(&height, {index_of_bbox, 3}));
  *bbox_out = std::make_shared<BoundingBox>(x, y, width, height);
  return Status::OK();
}

Status BoundingBox::ValidateBoundingBoxes(const TensorRow &image_and_bbox) {
  constexpr int64_t input_size = 2;
  if (image_and_bbox.size() != input_size) {
    RETURN_STATUS_ERROR(StatusCode::kMDBoundingBoxInvalidShape,
                        "BoundingBox: invalid input, size of input data should be 2 "
                        "(including image and bounding box), but got: " +
                          std::to_string(image_and_bbox.size()));
  }
  if (image_and_bbox[1]->shape().Size() < 2) {
    RETURN_STATUS_ERROR(StatusCode::kMDBoundingBoxInvalidShape,
                        "BoundingBox: bounding boxes should have to be two-dimensional matrix at least, "
                        "but got " +
                          std::to_string(image_and_bbox[1]->shape().Size()) + " dimension.");
  }
  if (image_and_bbox[0]->shape().Size() < kMinImageRank) {
    RETURN_STATUS_UNEXPECTED("Invalid data, input image hasn't been decoded, you may need to perform Decode first.");
  }

  int64_t num_of_features = image_and_bbox[1]->shape()[1];
  if (num_of_features < kNumOfCols) {
    RETURN_STATUS_ERROR(
      StatusCode::kMDBoundingBoxInvalidShape,
      "BoundingBox: bounding boxes should be have at least 4 features, but got: " + std::to_string(num_of_features));
  }
  std::vector<std::shared_ptr<BoundingBox>> bbox_list;
  RETURN_IF_NOT_OK(GetListOfBoundingBoxes(image_and_bbox[1], &bbox_list));
  int64_t img_h = image_and_bbox[0]->shape()[0];
  int64_t img_w = image_and_bbox[0]->shape()[1];
  for (auto &bbox : bbox_list) {
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int64_t>::max() - bbox->x()) > bbox->width(),
                                 "BoundingBox: bbox width is too large as coordinate x bigger than max num of int64.");
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int64_t>::max() - bbox->y()) > bbox->height(),
                                 "BoundingBox: bbox height is too large as coordinate y bigger than max num of int64.");
    if ((static_cast<int64_t>(bbox->x() + bbox->width()) > img_w) ||
        (static_cast<int64_t>(bbox->y() + bbox->height()) > img_h)) {
      RETURN_STATUS_ERROR(
        StatusCode::kMDBoundingBoxOutOfBounds,
        "BoundingBox: bounding boxes is out of bounds of the image, as image width: " + std::to_string(img_w) +
          ", bbox width coordinate: " + std::to_string(bbox->x() + bbox->width()) + ", and image height: " +
          std::to_string(img_h) + ", bbox height coordinate: " + std::to_string(bbox->y() + bbox->height()));
    }
    if (static_cast<int>(bbox->x()) < 0 || static_cast<int>(bbox->y()) < 0) {
      RETURN_STATUS_ERROR(StatusCode::kMDBoundingBoxOutOfBounds,
                          "BoundingBox: the coordinates of the bounding boxes has negative value, got: (" +
                            std::to_string(bbox->x()) + "," + std::to_string(bbox->y()) + ").");
    }
  }
  return Status::OK();
}

Status BoundingBox::WriteToTensor(const TensorPtr &bbox_tensor, dsize_t index_of_bbox) {
  CHECK_FAIL_RETURN_UNEXPECTED(bbox_tensor != nullptr, "BoundingBox: bbox_tensor is null.");
  RETURN_IF_NOT_OK(bbox_tensor->SetItemAt<bbox_float>({index_of_bbox, 0}, x_));
  RETURN_IF_NOT_OK(bbox_tensor->SetItemAt<bbox_float>({index_of_bbox, 1}, y_));
  RETURN_IF_NOT_OK(bbox_tensor->SetItemAt<bbox_float>({index_of_bbox, 2}, width_));
  RETURN_IF_NOT_OK(bbox_tensor->SetItemAt<bbox_float>({index_of_bbox, 3}, height_));
  return Status::OK();
}

Status BoundingBox::GetListOfBoundingBoxes(const TensorPtr &bbox_tensor,
                                           std::vector<std::shared_ptr<BoundingBox>> *bbox_out) {
  CHECK_FAIL_RETURN_UNEXPECTED(bbox_tensor != nullptr, "BoundingBox: bbox_tensor is null.");
  dsize_t num_of_boxes = bbox_tensor->shape()[0];
  for (dsize_t i = 0; i < num_of_boxes; i++) {
    std::shared_ptr<BoundingBox> bbox;
    RETURN_IF_NOT_OK(ReadFromTensor(bbox_tensor, i, &bbox));
    bbox_out->push_back(bbox);
  }
  return Status::OK();
}

Status BoundingBox::CreateTensorFromBoundingBoxList(const std::vector<std::shared_ptr<BoundingBox>> &bboxes,
                                                    TensorPtr *tensor_out) {
  dsize_t num_of_boxes = bboxes.size();
  std::vector<bbox_float> bboxes_for_tensor;
  for (dsize_t i = 0; i < num_of_boxes; i++) {
    bbox_float b_data[kNumOfCols] = {bboxes[i]->x(), bboxes[i]->y(), bboxes[i]->width(), bboxes[i]->height()};
    bboxes_for_tensor.insert(bboxes_for_tensor.end(), b_data, b_data + kNumOfCols);
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(bboxes_for_tensor, TensorShape{num_of_boxes, kNumOfCols}, tensor_out));
  return Status::OK();
}

Status BoundingBox::PadBBoxes(const TensorPtr *bbox_list, size_t bbox_count, int32_t pad_top, int32_t pad_left) {
  CHECK_FAIL_RETURN_UNEXPECTED(bbox_list != nullptr, "BoundingBox: bbox_list ptr is null.");
  for (dsize_t i = 0; i < bbox_count; i++) {
    std::shared_ptr<BoundingBox> bbox;
    RETURN_IF_NOT_OK(ReadFromTensor(*bbox_list, i, &bbox));
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - bbox->x()) > pad_left,
                                 "BoundingBox: pad_left is too large as coordinate x bigger than max num of int64.");
    bbox->SetX(bbox->x() + pad_left);
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<int32_t>::max() - bbox->y()) > pad_top,
                                 "BoundingBox: pad_top is too large as coordinate y bigger than max num of int64.");
    bbox->SetY(bbox->y() + pad_top);
    RETURN_IF_NOT_OK(bbox->WriteToTensor(*bbox_list, i));
  }
  return Status::OK();
}

Status BoundingBox::UpdateBBoxesForCrop(TensorPtr *bbox_list, size_t *bbox_count, int32_t CB_Xmin, int32_t CB_Ymin,
                                        int32_t CB_Xmax, int32_t CB_Ymax) {
  CHECK_FAIL_RETURN_UNEXPECTED(bbox_list != nullptr, "BoundingBox: bbox_list ptr is null.");
  // PASS LIST, COUNT OF BOUNDING BOXES
  // Also PAss X/Y Min/Max of image cropped region - normally obtained from 'GetCropBox' functions
  std::vector<dsize_t> correct_ind;
  std::vector<bbox_float> copyVals;
  dsize_t bboxDim = (*bbox_list)->shape()[1];
  for (dsize_t i = 0; i < *bbox_count; i++) {
    std::shared_ptr<BoundingBox> bbox;
    RETURN_IF_NOT_OK(ReadFromTensor(*bbox_list, i, &bbox));
    bbox_float bb_Xmax = bbox->x() + bbox->width();
    bbox_float bb_Ymax = bbox->y() + bbox->height();
    // check for image / BB overlap
    if (((bbox->x() > CB_Xmax) || (bbox->y() > CB_Ymax)) || ((bb_Xmax < CB_Xmin) || (bb_Ymax < CB_Ymin))) {
      continue;  // no overlap found
    }
    // Update this bbox and select it to move to the final output tensor
    correct_ind.push_back(i);
    // adjust BBox corners by bringing into new CropBox if beyond
    // Also resetting/adjusting for boxes to lie within CropBox instead of Image - subtract CropBox Xmin/YMin

    bbox_float bb_Xmin = bbox->x() - std::min(static_cast<bbox_float>(0.0), (bbox->x() - CB_Xmin)) - CB_Xmin;
    bbox_float bb_Ymin = bbox->y() - std::min(static_cast<bbox_float>(0.0), (bbox->y() - CB_Ymin)) - CB_Ymin;
    bb_Xmax = bb_Xmax - std::max(static_cast<bbox_float>(0.0), (bb_Xmax - CB_Xmax)) - CB_Xmin;
    bb_Ymax = bb_Ymax - std::max(static_cast<bbox_float>(0.0), (bb_Ymax - CB_Ymax)) - CB_Ymin;

    // bound check for float values
    bb_Xmin = std::max(bb_Xmin, static_cast<bbox_float>(0));
    bb_Ymin = std::max(bb_Ymin, static_cast<bbox_float>(0));
    bb_Xmax = std::min(bb_Xmax, static_cast<bbox_float>(CB_Xmax - CB_Xmin));  // find max value relative to new image
    bb_Ymax = std::min(bb_Ymax, static_cast<bbox_float>(CB_Ymax - CB_Ymin));

    // reset min values and calculate width/height from Box corners
    bbox->SetX(bb_Xmin);
    bbox->SetY(bb_Ymin);
    bbox->SetWidth(bb_Xmax - bb_Xmin);
    bbox->SetHeight(bb_Ymax - bb_Ymin);
    RETURN_IF_NOT_OK(bbox->WriteToTensor(*bbox_list, i));
  }
  // create new tensor and copy over bboxes still valid to the image
  // bboxes outside of new cropped region are ignored - empty tensor returned in case of none
  *bbox_count = correct_ind.size();
  bbox_float temp = 0.0;
  for (auto slice : correct_ind) {  // for every index in the loop
    for (dsize_t ix = 0; ix < bboxDim; ix++) {
      RETURN_IF_NOT_OK((*bbox_list)->GetItemAt<bbox_float>(&temp, {slice, ix}));
      copyVals.push_back(temp);
    }
  }
  std::shared_ptr<Tensor> retV;
  RETURN_IF_NOT_OK(
    Tensor::CreateFromVector(copyVals, TensorShape({static_cast<dsize_t>(*bbox_count), bboxDim}), &retV));
  (*bbox_list) = retV;  // reset pointer
  return Status::OK();
}

Status BoundingBox::UpdateBBoxesForResize(const TensorPtr &bbox_list, size_t bbox_count, int32_t target_width,
                                          int32_t target_height, int32_t orig_width, int32_t orig_height) {
  CHECK_FAIL_RETURN_UNEXPECTED(bbox_list != nullptr, "BoundingBox: bbox_list ptr is null.");
  CHECK_FAIL_RETURN_UNEXPECTED(orig_width != 0, "BoundingBox: orig_width is zero.");
  CHECK_FAIL_RETURN_UNEXPECTED(orig_height != 0, "BoundingBox: orig_height is zero.");

  // cast to float to preserve fractional
  bbox_float W_aspRatio = (target_width * 1.0) / (orig_width * 1.0);
  bbox_float H_aspRatio = (target_height * 1.0) / (orig_height * 1.0);
  for (dsize_t i = 0; i < bbox_count; i++) {
    // for each bounding box
    std::shared_ptr<BoundingBox> bbox;
    RETURN_IF_NOT_OK(ReadFromTensor(bbox_list, i, &bbox));

    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / bbox->x()) > W_aspRatio,
                                 "BoundingBox: Width aspect Ratio is too large as got: " + std::to_string(W_aspRatio));
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / bbox->y()) > H_aspRatio,
                                 "BoundingBox: Height aspect Ratio is too large as got: " + std::to_string(H_aspRatio));
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / bbox->width()) > W_aspRatio,
                                 "BoundingBox: Width aspect Ratio is too large as got: " + std::to_string(W_aspRatio));
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() / bbox->height()) > H_aspRatio,
                                 "BoundingBox: Height aspect Ratio is too large as got: " + std::to_string(H_aspRatio));

    // update positions and widths
    bbox->SetX(bbox->x() * W_aspRatio);
    bbox->SetY(bbox->y() * H_aspRatio);
    bbox->SetWidth(bbox->width() * W_aspRatio);
    bbox->SetHeight(bbox->height() * H_aspRatio);
    // reset bounding box values
    RETURN_IF_NOT_OK(bbox->WriteToTensor(bbox_list, i));
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
