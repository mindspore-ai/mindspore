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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_BOUNDING_BOX_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_BOUNDING_BOX_H_

#include <memory>
#include <vector>
#include <string>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class BoundingBox {
 public:
  typedef float_t bbox_float;

  /// \brief Constructor for BoundingBox
  /// \param[in] x horizontal axis coordinate of bounding box
  /// \param[in] y vertical axis coordinate of bounding box
  /// \param[in] width width of bounding box on horizontal axis
  /// \param[in] height height of bounding box on vertical axis
  BoundingBox(bbox_float x, bbox_float y, bbox_float width, bbox_float height);

  ~BoundingBox() = default;

  /// \brief Provide stream operator for displaying a bounding box.
  friend std::ostream &operator<<(std::ostream &out, const BoundingBox &bbox) {
    out << "Bounding Box with (X,Y,W,H): (" << bbox.x_ << "," << bbox.y_ << "," << bbox.width_ << "," << bbox.height_
        << ")";
    return out;
  }

  /// Getters
  bbox_float x() { return x_; }
  bbox_float y() { return y_; }
  bbox_float width() { return width_; }
  bbox_float height() { return height_; }

  /// Setters
  void SetX(bbox_float x) { x_ = x; }
  void SetY(bbox_float y) { y_ = y; }
  void SetWidth(bbox_float w) { width_ = w; }
  void SetHeight(bbox_float h) { height_ = h; }

  /// \brief Set the bounding box data to bbox at a certain index in the tensor.
  /// \param[in] bbox_tensor tensor containing a list of bounding boxes of shape (m, n) where n >= 4
  ///    and first 4 items of each row are x,y,w,h of the bounding box
  /// \param[in] index_of_bbox index of bounding box to set to tensor
  /// \returns Status status of bounding box set
  Status WriteToTensor(const TensorPtr &bbox_tensor, dsize_t index_of_bbox = 0);

  /// \brief Create a bounding box object from an item at a certain index in a tensor.
  /// \param[in] bbox_tensor tensor containing a list of bounding boxes of shape (m, n) where n >= 4
  ///    and first 4 items of each row are x,y,w,h of the bounding box
  /// \param[in] index_of_bbox index of bounding box to fetch from the tensor
  /// \param[out] bbox_out output bounding box
  /// \returns Status status of bounding box fetch
  static Status ReadFromTensor(const TensorPtr &bbox_tensor, dsize_t index_of_bbox,
                               std::shared_ptr<BoundingBox> *bbox_out);

  /// \brief Validate a list of bounding boxes with respect to an image.
  /// \param[in] image_and_bbox tensor containing a list of bounding boxes of shape (m, n) where n >= 4
  ///    and first 4 items of each row are x,y,w,h of the bounding box and an image of shape (H, W, C) or (H, W)
  /// \returns Status status of bounding box fetch
  static Status ValidateBoundingBoxes(const TensorRow &image_and_bbox);

  /// \brief Get a list of bounding boxes from a tensor.
  /// \param[in] bbox_tensor tensor containing a list of bounding boxes of shape (m, n) where n >= 4
  ///    and first 4 items of each row are x,y,w,h of the bounding box
  /// \param[out] bbox_out output vector of bounding boxes
  /// \returns Status status of bounding box list fetch
  static Status GetListOfBoundingBoxes(const TensorPtr &bbox_tensor,
                                       std::vector<std::shared_ptr<BoundingBox>> *bbox_out);

  /// \brief Creates a tensor from a list of bounding boxes.
  /// \param[in] bboxes list of bounding boxes
  /// \param[out] tensor_out output tensor
  /// \returns Status status of tensor creation
  static Status CreateTensorFromBoundingBoxList(const std::vector<std::shared_ptr<BoundingBox>> &bboxes,
                                                TensorPtr *tensor_out);

  /// \brief Updates bounding boxes with required Top and Left padding
  /// \note Top and Left padding amounts required to adjust bboxs min X,Y values according to padding 'push'
  ///     Top/Left since images 0,0 coordinate is taken from top left
  /// \param bboxList: A tensor containing bounding box tensors
  /// \param bboxCount: total Number of bounding boxes - required within caller function to run update loop
  /// \param pad_top: Total amount of padding applied to image top
  /// \param pad_left: Total amount of padding applied to image left side
  static Status PadBBoxes(const TensorPtr *bbox_list, size_t bbox_count, int32_t pad_top, int32_t pad_left);

  /// \brief Updates and checks bounding boxes for new cropped region of image
  /// \param bbox_list: A tensor containing bounding box tensors
  /// \param bbox_count: total Number of bounding boxes - required within caller function to run update loop
  /// \param CB_Xmin: Image's CropBox Xmin coordinate
  /// \param CB_Xmin: Image's CropBox Ymin coordinate
  /// \param CB_Xmax: Image's CropBox Xmax coordinate - (Xmin + width)
  /// \param CB_Xmax: Image's CropBox Ymax coordinate - (Ymin + height)
  static Status UpdateBBoxesForCrop(TensorPtr *bbox_list, size_t *bbox_count, int32_t CB_Xmin, int32_t CB_Ymin,
                                    int32_t CB_Xmax, int32_t CB_Ymax);

  /// \brief Updates bounding boxes for an Image Resize Operation - Takes in set of valid BBoxes
  /// For e.g those that remain after a crop
  /// \param bbox_list: A tensor containing bounding box tensors
  /// \param bbox_count: total Number of bounding boxes - required within caller function to run update loop
  /// \param target_width: required width of image post resize
  /// \param target_height: required height of image post resize
  /// \param orig_width: current width of image pre resize
  /// \param orig_height: current height of image pre resize
  static Status UpdateBBoxesForResize(const TensorPtr &bbox_list, size_t bbox_count, int32_t target_width,
                                      int32_t target_height, int32_t orig_width, int32_t orig_height);

 private:
  bbox_float x_;
  bbox_float y_;
  bbox_float width_;
  bbox_float height_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_BOUNDING_BOX_H_
