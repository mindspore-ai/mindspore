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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_SERDES_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_SERDES_H_

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <unordered_set>
#include <utility>
#include <nlohmann/json.hpp>

#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/concat_node.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/project_node.h"
#include "minddata/dataset/engine/ir/datasetops/rename_node.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"
#include "minddata/dataset/engine/ir/datasetops/take_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"

#include "minddata/dataset/engine/ir/datasetops/source/album_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/flickr_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"

#include "minddata/dataset/engine/ir/datasetops/source/samplers/distributed_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/pk_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/prebuilt_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/random_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/sequential_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/subset_random_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/subset_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/weighted_random_sampler_ir.h"

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/iterator.h"
#include "minddata/dataset/include/dataset/samplers.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"

#include "minddata/dataset/kernels/py_func_op.h"
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"
#include "minddata/dataset/kernels/ir/vision/adjust_gamma_ir.h"
#include "minddata/dataset/kernels/ir/vision/affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/ascend_vision_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/bounding_box_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/center_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutmix_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutout_ir.h"
#include "minddata/dataset/kernels/ir/vision/decode_ir.h"
#include "minddata/dataset/kernels/ir/vision/equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/gaussian_blur_ir.h"
#include "minddata/dataset/kernels/ir/vision/horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/hwc_to_chw_ir.h"
#include "minddata/dataset/kernels/ir/vision/invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/mixup_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_adjust_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_decode_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_posterize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_rotation_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_select_subpolicy_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_solarize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/rescale_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_preserve_ar_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgba_to_bgr_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgba_to_rgb_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgb_to_bgr_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgb_to_gray_ir.h"
#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"
#include "minddata/dataset/kernels/ir/vision/slice_patches_ir.h"
#include "minddata/dataset/kernels/ir/vision/softdvpp_decode_random_crop_resize_jpeg_ir.h"
#include "minddata/dataset/kernels/ir/vision/softdvpp_decode_resize_jpeg_ir.h"
#include "minddata/dataset/kernels/ir/vision/swap_red_blue_ir.h"
#include "minddata/dataset/kernels/ir/vision/uniform_aug_ir.h"
#include "minddata/dataset/kernels/ir/vision/vertical_flip_ir.h"
#include "minddata/dataset/text/ir/kernels/text_ir.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// \brief The Serdes class is used to serialize an IR tree into JSON string and dump into file if file name
/// specified.
class Serdes {
 public:
  /// \brief Constructor
  Serdes() {}

  /// \brief default destructor
  ~Serdes() = default;

  /// \brief function to serialize IR tree into JSON string and/or JSON file
  /// \param[in] node IR node to be transferred
  /// \param[in] filename The file name. If specified, save the generated JSON string into the file
  /// \param[out] out_json The result json string
  /// \return Status The status code returned
  static Status SaveToJSON(std::shared_ptr<DatasetNode> node, const std::string &filename, nlohmann::json *out_json);

  /// \brief function to de-serialize JSON file to IR tree
  /// \param[in] json_filepath input path of json file
  /// \param[out] ds The deserialized dataset
  /// \return Status The status code returned
  static Status Deserialize(const std::string &json_filepath, std::shared_ptr<DatasetNode> *ds);

  /// \brief Helper function to construct IR tree, separate zip and other operations
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[out] ds Shared pointer of a DatasetNode object containing the deserialized IR tree
  /// \return Status The status code returned
  static Status ConstructPipeline(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds);

  /// \brief Helper functions for creating sampler, separate different samplers and call the related function
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[out] sampler Deserialized sampler
  /// \return Status The status code returned
  static Status ConstructSampler(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *sampler);

  /// \brief helper function to construct tensor operations
  /// \param[in] json_obj json object of operations to be deserilized
  /// \param[out] vector of tensor operation pointer
  /// \return Status The status code returned
  static Status ConstructTensorOps(nlohmann::json json_obj, std::vector<std::shared_ptr<TensorOperation>> *result);

  /// \brief helper function to load tensor operations from dataset JSON and construct Execute object.
  /// \param[in] dataset_json JSON string of dataset.
  /// \param[in] process_column Select all map operations which process this column.
  /// \param[out] data_graph Execute object contains tensor operations of map.
  /// \return Status The status code returned.
  static Status ParseMindIRPreprocess(const std::string &dataset_json, const std::string &process_column,
                                      std::vector<std::shared_ptr<mindspore::dataset::Execute>> *data_graph);

 protected:
  /// \brief Helper function to save JSON to a file
  /// \param[in] json_string The JSON string to be saved to the file
  /// \param[in] file_name The file name
  /// \return Status The status code returned
  static Status SaveJSONToFile(nlohmann::json json_string, const std::string &file_name);

  /// \brief Function to determine type of the node - dataset node if no dataset exists or operation node
  /// \param[in] child_ds children datasets that is already created
  /// \param[in] json_obj json object to read out type of the node
  /// \param[out] ds Shared pointer of a DatasetNode object containing the deserialized IR tree
  /// \return create new node based on the input dataset and type of the operation
  static Status CreateNode(const std::shared_ptr<DatasetNode> &child_ds, nlohmann::json json_obj,
                           std::shared_ptr<DatasetNode> *ds);

  /// \brief Helper functions for creating dataset nodes, separate different datasets and call the related function
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[in] op_type type of dataset
  /// \param[out] ds Shared pointer of a DatasetNode object containing the deserialized IR tree
  /// \return Status The status code returned
  static Status CreateDatasetNode(const nlohmann::json &json_obj, const std::string &op_type,
                                  std::shared_ptr<DatasetNode> *ds);

  /// \brief Helper functions for creating operation nodes, separate different operations and call the related function
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[in] op_type type of dataset
  /// \param[out] result Shared pointer of a DatasetNode object containing the deserialized IR tree
  /// \return Status The status code returned
  static Status CreateDatasetOperationNode(const std::shared_ptr<DatasetNode> &ds, const nlohmann::json &json_obj,
                                           const std::string &op_type, std::shared_ptr<DatasetNode> *result);

  /// \brief Helper function to map the function pointers
  /// \return map of key to function pointer
  static std::map<std::string, Status (*)(nlohmann::json json_obj, std::shared_ptr<TensorOperation> *operation)>
  InitializeFuncPtr();

 private:
  static std::map<std::string, Status (*)(nlohmann::json json_obj, std::shared_ptr<TensorOperation> *operation)>
    func_ptr_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_SERDES_H_
