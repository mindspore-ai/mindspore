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

#include "minddata/dataset/api/python/pybind_register.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/engine/datasetops/dataset_op.h"

#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/clue_op.h"
#include "minddata/dataset/engine/datasetops/source/csv_op.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(CifarOp, 1, ([](const py::module *m) {
                  (void)py::class_<CifarOp, DatasetOp, std::shared_ptr<CifarOp>>(*m, "CifarOp")
                    .def_static("get_num_rows", [](const std::string &dir, const std::string &usage, bool isCifar10) {
                      int64_t count = 0;
                      THROW_IF_ERROR(CifarOp::CountTotalRows(dir, usage, isCifar10, &count));
                      return count;
                    });
                }));

PYBIND_REGISTER(ClueOp, 1, ([](const py::module *m) {
                  (void)py::class_<ClueOp, DatasetOp, std::shared_ptr<ClueOp>>(*m, "ClueOp")
                    .def_static("get_num_rows", [](const py::list &files) {
                      int64_t count = 0;
                      std::vector<std::string> filenames;
                      for (auto file : files) {
                        file.is_none() ? (void)filenames.emplace_back("") : filenames.push_back(py::str(file));
                      }
                      THROW_IF_ERROR(ClueOp::CountAllFileRows(filenames, &count));
                      return count;
                    });
                }));

PYBIND_REGISTER(CsvOp, 1, ([](const py::module *m) {
                  (void)py::class_<CsvOp, DatasetOp, std::shared_ptr<CsvOp>>(*m, "CsvOp")
                    .def_static("get_num_rows", [](const py::list &files, bool csv_header) {
                      int64_t count = 0;
                      std::vector<std::string> filenames;
                      for (auto file : files) {
                        file.is_none() ? (void)filenames.emplace_back("") : filenames.push_back(py::str(file));
                      }
                      THROW_IF_ERROR(CsvOp::CountAllFileRows(filenames, csv_header, &count));
                      return count;
                    });
                }));
PYBIND_REGISTER(CocoOp, 1, ([](const py::module *m) {
                  (void)py::class_<CocoOp, DatasetOp, std::shared_ptr<CocoOp>>(*m, "CocoOp")
                    .def_static("get_class_indexing",
                                [](const std::string &dir, const std::string &file, const std::string &task) {
                                  std::vector<std::pair<std::string, std::vector<int32_t>>> output_class_indexing;
                                  THROW_IF_ERROR(CocoOp::GetClassIndexing(dir, file, task, &output_class_indexing));
                                  return output_class_indexing;
                                })
                    .def_static("get_num_rows",
                                [](const std::string &dir, const std::string &file, const std::string &task) {
                                  int64_t count = 0;
                                  THROW_IF_ERROR(CocoOp::CountTotalRows(dir, file, task, &count));
                                  return count;
                                });
                }));

PYBIND_REGISTER(ImageFolderOp, 1, ([](const py::module *m) {
                  (void)py::class_<ImageFolderOp, DatasetOp, std::shared_ptr<ImageFolderOp>>(*m, "ImageFolderOp")
                    .def_static("get_num_rows",
                                [](const std::string &path) {
                                  int64_t count = 0;
                                  THROW_IF_ERROR(ImageFolderOp::CountRowsAndClasses(path, {}, &count, nullptr, {}));
                                  return count;
                                })
                    .def_static("get_num_classes", [](const std::string &path,
                                                      const std::map<std::string, int32_t> class_index) {
                      int64_t num_classes = 0;
                      THROW_IF_ERROR(ImageFolderOp::CountRowsAndClasses(path, {}, nullptr, &num_classes, class_index));
                      return num_classes;
                    });
                }));

PYBIND_REGISTER(ManifestOp, 1, ([](const py::module *m) {
                  (void)py::class_<ManifestOp, DatasetOp, std::shared_ptr<ManifestOp>>(*m, "ManifestOp");
                }));
PYBIND_REGISTER(MindRecordOp, 1, ([](const py::module *m) {
                  (void)py::class_<MindRecordOp, DatasetOp, std::shared_ptr<MindRecordOp>>(*m, "MindRecordOp")
                    .def_static("get_num_rows", [](const std::vector<std::string> &paths, bool load_dataset,
                                                   const py::object &sampler, const int64_t num_padded) {
                      int64_t count = 0;
                      std::shared_ptr<mindrecord::ShardOperator> op;
                      if (py::hasattr(sampler, "create_for_minddataset")) {
                        auto create = sampler.attr("create_for_minddataset");
                        op = create().cast<std::shared_ptr<mindrecord::ShardOperator>>();
                      }
                      THROW_IF_ERROR(MindRecordOp::CountTotalRows(paths, load_dataset, op, &count, num_padded));
                      return count;
                    });
                }));

PYBIND_REGISTER(MnistOp, 1, ([](const py::module *m) {
                  (void)py::class_<MnistOp, DatasetOp, std::shared_ptr<MnistOp>>(*m, "MnistOp")
                    .def_static("get_num_rows", [](const std::string &dir, const std::string &usage) {
                      int64_t count = 0;
                      THROW_IF_ERROR(MnistOp::CountTotalRows(dir, usage, &count));
                      return count;
                    });
                }));

PYBIND_REGISTER(TextFileOp, 1, ([](const py::module *m) {
                  (void)py::class_<TextFileOp, DatasetOp, std::shared_ptr<TextFileOp>>(*m, "TextFileOp")
                    .def_static("get_num_rows", [](const py::list &files) {
                      int64_t count = 0;
                      std::vector<std::string> filenames;
                      for (auto file : files) {
                        !file.is_none() ? filenames.push_back(py::str(file)) : (void)filenames.emplace_back("");
                      }
                      THROW_IF_ERROR(TextFileOp::CountAllFileRows(filenames, &count));
                      return count;
                    });
                }));

PYBIND_REGISTER(TFReaderOp, 1, ([](const py::module *m) {
                  (void)py::class_<TFReaderOp, DatasetOp, std::shared_ptr<TFReaderOp>>(*m, "TFReaderOp")
                    .def_static(
                      "get_num_rows", [](const py::list &files, int64_t numParallelWorkers, bool estimate = false) {
                        int64_t count = 0;
                        std::vector<std::string> filenames;
                        for (auto l : files) {
                          !l.is_none() ? filenames.push_back(py::str(l)) : (void)filenames.emplace_back("");
                        }
                        THROW_IF_ERROR(TFReaderOp::CountTotalRows(&count, filenames, numParallelWorkers, estimate));
                        return count;
                      });
                }));

PYBIND_REGISTER(VOCOp, 1, ([](const py::module *m) {
                  (void)py::class_<VOCOp, DatasetOp, std::shared_ptr<VOCOp>>(*m, "VOCOp")
                    .def_static("get_class_indexing", [](const std::string &dir, const std::string &task_type,
                                                         const std::string &task_mode, const py::dict &dict) {
                      std::map<std::string, int32_t> output_class_indexing;
                      THROW_IF_ERROR(VOCOp::GetClassIndexing(dir, task_type, task_mode, dict, &output_class_indexing));
                      return output_class_indexing;
                    });
                }));

}  // namespace dataset
}  // namespace mindspore
