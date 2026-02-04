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

#include "minddata/dataset/engine/operation/datasetops/batch_node.h"
#include "minddata/dataset/engine/operation/datasetops/concat_node.h"
#include "minddata/dataset/engine/operation/datasetops/dataset_node.h"
#include "minddata/dataset/engine/operation/datasetops/map_node.h"
#include "minddata/dataset/engine/operation/datasetops/project_node.h"
#include "minddata/dataset/engine/operation/datasetops/rename_node.h"
#include "minddata/dataset/engine/operation/datasetops/repeat_node.h"
#include "minddata/dataset/engine/operation/datasetops/shuffle_node.h"
#include "minddata/dataset/engine/operation/datasetops/skip_node.h"
#include "minddata/dataset/engine/operation/datasetops/data_queue_node.h"
#include "minddata/dataset/engine/operation/datasetops/take_node.h"
#include "minddata/dataset/engine/operation/datasetops/zip_node.h"

#include "minddata/dataset/data_source/operation/album_node.h"
#include "minddata/dataset/data_source/operation/celeba_node.h"
#include "minddata/dataset/data_source/operation/cifar10_node.h"
#include "minddata/dataset/data_source/operation/cifar100_node.h"
#include "minddata/dataset/data_source/operation/clue_node.h"
#include "minddata/dataset/data_source/operation/coco_node.h"
#include "minddata/dataset/data_source/operation/csv_node.h"
#include "minddata/dataset/data_source/operation/flickr_node.h"
#include "minddata/dataset/data_source/operation/image_folder_node.h"
#include "minddata/dataset/data_source/operation/kitti_node.h"
#include "minddata/dataset/data_source/operation/lj_speech_node.h"
#include "minddata/dataset/data_source/operation/manifest_node.h"
#include "minddata/dataset/data_source/operation/mnist_node.h"
#include "minddata/dataset/data_source/operation/sst2_node.h"
#include "minddata/dataset/data_source/operation/text_file_node.h"
#include "minddata/dataset/data_source/operation/tf_record_node.h"
#include "minddata/dataset/data_source/operation/voc_node.h"
#include "minddata/dataset/data_source/operation/wiki_text_node.h"

#include "minddata/dataset/data_source/operation/samplers/distributed_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/pk_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/prebuilt_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/random_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/samplers_ir.h"
#include "minddata/dataset/data_source/operation/samplers/sequential_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/skip_first_epoch_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/subset_random_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/subset_sampler_ir.h"
#include "minddata/dataset/data_source/operation/samplers/weighted_random_sampler_ir.h"

#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/iterator.h"
#include "minddata/dataset/include/dataset/samplers.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"

#include "minddata/dataset/kernels/py_func_op.h"
#include "minddata/dataset/general/transform/transforms_ir.h"
#include "minddata/dataset/vision/transform/adjust_gamma_ir.h"
#include "minddata/dataset/vision/transform/affine_ir.h"
#include "minddata/dataset/vision/transform/ascend_vision_ir.h"
#include "minddata/dataset/vision/transform/auto_contrast_ir.h"
#include "minddata/dataset/vision/transform/bounding_box_augment_ir.h"
#include "minddata/dataset/vision/transform/center_crop_ir.h"
#include "minddata/dataset/vision/transform/crop_ir.h"
#include "minddata/dataset/vision/transform/cutmix_batch_ir.h"
#include "minddata/dataset/vision/transform/cutout_ir.h"
#include "minddata/dataset/vision/transform/decode_ir.h"
#include "minddata/dataset/vision/transform/equalize_ir.h"
#include "minddata/dataset/vision/transform/gaussian_blur_ir.h"
#include "minddata/dataset/vision/transform/horizontal_flip_ir.h"
#include "minddata/dataset/vision/transform/hwc_to_chw_ir.h"
#include "minddata/dataset/vision/transform/invert_ir.h"
#include "minddata/dataset/vision/transform/mixup_batch_ir.h"
#include "minddata/dataset/vision/transform/normalize_ir.h"
#include "minddata/dataset/vision/transform/normalize_pad_ir.h"
#include "minddata/dataset/vision/transform/pad_ir.h"
#include "minddata/dataset/vision/transform/random_affine_ir.h"
#include "minddata/dataset/vision/transform/random_color_adjust_ir.h"
#include "minddata/dataset/vision/transform/random_color_ir.h"
#include "minddata/dataset/vision/transform/random_crop_decode_resize_ir.h"
#include "minddata/dataset/vision/transform/random_crop_ir.h"
#include "minddata/dataset/vision/transform/random_crop_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_horizontal_flip_ir.h"
#include "minddata/dataset/vision/transform/random_horizontal_flip_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_posterize_ir.h"
#include "minddata/dataset/vision/transform/random_resized_crop_ir.h"
#include "minddata/dataset/vision/transform/random_resized_crop_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_resize_ir.h"
#include "minddata/dataset/vision/transform/random_resize_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/random_rotation_ir.h"
#include "minddata/dataset/vision/transform/random_select_subpolicy_ir.h"
#include "minddata/dataset/vision/transform/random_sharpness_ir.h"
#include "minddata/dataset/vision/transform/random_solarize_ir.h"
#include "minddata/dataset/vision/transform/random_vertical_flip_ir.h"
#include "minddata/dataset/vision/transform/random_vertical_flip_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/rescale_ir.h"
#include "minddata/dataset/vision/transform/resize_ir.h"
#include "minddata/dataset/vision/transform/resize_preserve_ar_ir.h"
#include "minddata/dataset/vision/transform/resize_with_bbox_ir.h"
#include "minddata/dataset/vision/transform/rgba_to_bgr_ir.h"
#include "minddata/dataset/vision/transform/rgba_to_rgb_ir.h"
#include "minddata/dataset/vision/transform/rgb_to_bgr_ir.h"
#include "minddata/dataset/vision/transform/rgb_to_gray_ir.h"
#include "minddata/dataset/vision/transform/rotate_ir.h"
#include "minddata/dataset/vision/transform/slice_patches_ir.h"
#include "minddata/dataset/vision/transform/swap_red_blue_ir.h"
#include "minddata/dataset/vision/transform/to_tensor_ir.h"
#include "minddata/dataset/vision/transform/uniform_aug_ir.h"
#include "minddata/dataset/vision/transform/vertical_flip_ir.h"
#include "minddata/dataset/text/transform/text_ir.h"

namespace mindspore::dataset {
using FuncResult = std::map<std::string, Status (*)(nlohmann::json, std::shared_ptr<TensorOperation> *)>;

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

  /// \brief Function to update the parameters [num_parallel_workers, connector_queue_size] in the serialized JSON
  /// object of the optimized IR tree
  /// \param[in, out] serialized_json The optimized ir tree json node
  /// \param[in] op_map An ID to DatasetOp mapping
  static Status UpdateOptimizedIRTreeJSON(nlohmann::json *serialized_json,
                                          const std::map<int32_t, std::shared_ptr<DatasetOp>> &op_map);

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
  /// \param[in] map_json_string JSON string of dataset.
  /// \param[out] data_graph Execute object contains tensor operations of map.
  /// \return Status The status code returned.
  static Status ParseMindIRPreprocess(const std::vector<std::string> &map_json_string,
                                      std::vector<std::shared_ptr<mindspore::dataset::Execute>> *data_graph);

  /// \brief Helper function to save JSON to a file
  /// \param[in] json_string The JSON string to be saved to the file
  /// \param[in] file_name The file name
  /// \param[in] pretty Flag to control pretty printing of JSON string to the file
  /// \return Status The status code returned
  static Status SaveJSONToFile(const nlohmann::json &json_string, const std::string &file_name, bool pretty = false);

 protected:
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
  static FuncResult InitializeFuncPtr();

  /// \brief Helper function to perform recursive DFS on the optimized IR tree and to match each IR node with its
  /// corresponding dataset op
  /// \param [in, out] serialized_json The optimized ir tree json node
  /// \param [in, out] op_id The id in execution tree from where to continue the IR Node - DatasetOp matching search
  /// \param [in] op_map An ID to DatasetOp mapping
  static Status RecurseUpdateOptimizedIRTreeJSON(nlohmann::json *serialized_json, int32_t *op_id,
                                                 const std::map<int32_t, std::shared_ptr<DatasetOp>> &op_map);

 private:
  static FuncResult func_ptr_;
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_SERDES_H_
