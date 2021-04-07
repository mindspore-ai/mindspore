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
#ifndef TESTS_DATASET_UT_CORE_COMMON_DE_UT_BBOXOP_COMMON_H_
#define TESTS_DATASET_UT_CORE_COMMON_DE_UT_BBOXOP_COMMON_H_

#include "cvop_common.h"
#include "minddata/dataset/util/path.h"

namespace UT {
namespace CVOP {
namespace BBOXOP {

class BBoxOpCommon : public CVOpCommon {
 public:
  enum FileType {
    kExpected,
    kActual
  };

  BBoxOpCommon();

  ~BBoxOpCommon();

  /// \brief Sets up the class's variable, images_and_annotations
  void SetUp() override;

  /// \brief Get all images and annotations in images_and_annotations TensorTable from dir
  /// \param[in] dir directory containing images and annotation folders
  /// \param[in] num_of_samples number of rows to fetch (default = 1)
  void GetInputImagesAndAnnotations(const std::string &dir, std::size_t num_of_samples = 1);

  /// \brief Save the given tensor table
  /// \param[in] type type of images to be stored (e.g. Expected or Actual)
  /// \param[in] op_name name of op being tested
  /// \param[in] table rows of images and corresponding annotations
  void SaveImagesWithAnnotations(FileType type, const std::string &op_name, const TensorTable &table);

  /// \brief Compare actual and expected results. The images will have the bounding boxes on them
  ///    Log if images don't match
  /// \param[in] op_name name of op being tested
  void CompareActualAndExpected(const std::string &op_name);

  /// \brief Load BBox data from an XML file into a Tensor
  /// \param[in] path path to XML bbox data file
  /// \param[in, out] target_BBox pointer to a Tensor to load
  /// \return True if file loaded successfully, false if error -> logged to STD out
  bool LoadAnnotationFile(const std::string &path, std::shared_ptr<Tensor> *target_BBox);

  TensorTable images_and_annotations_;

 private:
  // directory of image_folder when the dataset/data gets transferred to build
  std::string image_folder_build_;
  // directory of image_folder in the source project (used to store expected results)
  std::string image_folder_src_;
};
}  // namespace BBOXOP
}  // namespace CVOP
}  // namespace UT

#endif  // TESTS_DATASET_UT_CORE_COMMON_DE_UT_BBOXOP_COMMON_H_
