/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "pybind11/pybind11.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/api/python/python_mp.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/datasets.h"

// IR leaf nodes
#include "minddata/dataset/data_source/operation/ag_news_node.h"
#include "minddata/dataset/data_source/operation/amazon_review_node.h"
#include "minddata/dataset/data_source/operation/clue_node.h"
#include "minddata/dataset/data_source/operation/conll2000_node.h"
#include "minddata/dataset/data_source/operation/csv_node.h"
#include "minddata/dataset/data_source/operation/dbpedia_node.h"
#include "minddata/dataset/data_source/operation/en_wik9_node.h"
#include "minddata/dataset/data_source/operation/generator_node.h"
#include "minddata/dataset/data_source/operation/image_folder_node.h"
#include "minddata/dataset/data_source/operation/imdb_node.h"
#include "minddata/dataset/data_source/operation/iwslt2016_node.h"
#include "minddata/dataset/data_source/operation/iwslt2017_node.h"
#include "minddata/dataset/data_source/operation/penn_treebank_node.h"
#include "minddata/dataset/data_source/operation/random_node.h"
#include "minddata/dataset/data_source/operation/squad_node.h"
#include "minddata/dataset/data_source/operation/sst2_node.h"
#include "minddata/dataset/data_source/operation/text_file_node.h"
#include "minddata/dataset/data_source/operation/udpos_node.h"
#include "minddata/dataset/data_source/operation/wiki_text_node.h"
#include "minddata/dataset/data_source/operation/yahoo_answers_node.h"
#include "minddata/dataset/data_source/operation/yelp_review_node.h"

// IR leaf nodes disabled for android
#include "minddata/dataset/data_source/operation/manifest_node.h"
#include "minddata/dataset/data_source/operation/minddata_node.h"
#include "minddata/dataset/data_source/operation/multi30k_node.h"
#include "minddata/dataset/data_source/operation/sogou_news_node.h"
#include "minddata/dataset/data_source/operation/tf_record_node.h"

namespace mindspore {
namespace dataset {
// PYBIND FOR LEAF NODES
// (In alphabetical order)
PYBIND_REGISTER(AGNewsNode, 2, ([](const py::module *m) {
                  (void)py::class_<AGNewsNode, DatasetNode, std::shared_ptr<AGNewsNode>>(*m, "AGNewsNode",
                                                                                         "to create an AGNewsNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto ag_news = std::make_shared<AGNewsNode>(dataset_dir, num_samples, toShuffleMode(shuffle),
                                                                  usage, num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(ag_news->ValidateParams());
                      return ag_news;
                    }));
                }));

PYBIND_REGISTER(AmazonReviewNode, 2, ([](const py::module *m) {
                  (void)py::class_<AmazonReviewNode, DatasetNode, std::shared_ptr<AmazonReviewNode>>(
                    *m, "AmazonReviewNode", "to create an AmazonReviewNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<AmazonReviewNode> amazon_review = std::make_shared<AmazonReviewNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(amazon_review->ValidateParams());
                      return amazon_review;
                    }));
                }));

PYBIND_REGISTER(CLUENode, 2, ([](const py::module *m) {
                  (void)py::class_<CLUENode, DatasetNode, std::shared_ptr<CLUENode>>(*m, "CLUENode",
                                                                                     "to create a CLUENode")
                    .def(py::init([](const py::list &files, const std::string &task, const std::string &usage,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<CLUENode> clue_node =
                        std::make_shared<dataset::CLUENode>(toStringVector(files), task, usage, num_samples,
                                                            toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(clue_node->ValidateParams());
                      return clue_node;
                    }));
                }));

PYBIND_REGISTER(CoNLL2000Node, 2, ([](const py::module *m) {
                  (void)py::class_<CoNLL2000Node, DatasetNode, std::shared_ptr<CoNLL2000Node>>(
                    *m, "CoNLL2000Node", "to create a CoNLL2000Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<CoNLL2000Node> conll2000 = std::make_shared<CoNLL2000Node>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(conll2000->ValidateParams());
                      return conll2000;
                    }));
                }));

PYBIND_REGISTER(CSVNode, 2, ([](const py::module *m) {
                  (void)py::class_<CSVNode, DatasetNode, std::shared_ptr<CSVNode>>(*m, "CSVNode", "to create a CSVNode")
                    .def(py::init([](const std::vector<std::string> &csv_files, char field_delim,
                                     const py::list &column_defaults, const std::vector<std::string> &column_names,
                                     int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto csv =
                        std::make_shared<CSVNode>(csv_files, field_delim, toCSVBase(column_defaults), column_names,
                                                  num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(csv->ValidateParams());
                      return csv;
                    }));
                }));

PYBIND_REGISTER(DBpediaNode, 2, ([](const py::module *m) {
                  (void)py::class_<DBpediaNode, DatasetNode, std::shared_ptr<DBpediaNode>>(*m, "DBpediaNode",
                                                                                           "to create a DBpediaNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto dbpedia = std::make_shared<DBpediaNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(dbpedia->ValidateParams());
                      return dbpedia;
                    }));
                }));

PYBIND_REGISTER(EnWik9Node, 2, ([](const py::module *m) {
                  (void)py::class_<EnWik9Node, DatasetNode, std::shared_ptr<EnWik9Node>>(*m, "EnWik9Node",
                                                                                         "to create an EnWik9Node")
                    .def(py::init([](const std::string &dataset_dir, int32_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<EnWik9Node> en_wik9 = std::make_shared<EnWik9Node>(
                        dataset_dir, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(en_wik9->ValidateParams());
                      return en_wik9;
                    }));
                }));

PYBIND_REGISTER(GeneratorNode, 2, ([](const py::module *m) {
                  (void)py::class_<GeneratorNode, DatasetNode, std::shared_ptr<GeneratorNode>>(
                    *m, "GeneratorNode", "to create a GeneratorNode")
                    .def(
                      py::init([](const py::function &generator_function, const std::vector<std::string> &column_names,
                                  const std::vector<DataType> &column_types, int64_t dataset_len,
                                  const py::handle &sampler, uint32_t num_parallel_workers,
                                  std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime,
                                  bool has_batch_sampler) {
                        auto gen = std::make_shared<GeneratorNode>(
                          generator_function, column_names, column_types, dataset_len, toSamplerObj(sampler),
                          num_parallel_workers, std::move(python_multiprocessing_runtime), has_batch_sampler);
                        THROW_IF_ERROR(gen->ValidateParams());
                        return gen;
                      }))
                    .def(py::init([](const py::function &generator_function, const std::shared_ptr<SchemaObj> &schema,
                                     int64_t dataset_len, const py::handle &sampler, uint32_t num_parallel_workers,
                                     std::shared_ptr<PythonMultiprocessingRuntime> python_multiprocessing_runtime,
                                     bool has_batch_sampler) {
                      auto gen = std::make_shared<GeneratorNode>(
                        generator_function, schema, dataset_len, toSamplerObj(sampler), num_parallel_workers,
                        std::move(python_multiprocessing_runtime), has_batch_sampler);
                      THROW_IF_ERROR(gen->ValidateParams());
                      return gen;
                    }))
                    .def("Build", [](GeneratorNode &self) {
                      std::vector<std::shared_ptr<DatasetOp>> node_ops;
                      THROW_IF_ERROR(self.Build(&node_ops));
                      std::shared_ptr<GeneratorOp> generator_op = std::dynamic_pointer_cast<GeneratorOp>(node_ops[0]);
                      THROW_IF_ERROR(generator_op->CalculatedSampleId());
                      return generator_op;
                    });
                }));

PYBIND_REGISTER(GeneratorOp, 0, ([](const py::module *m) {
                  (void)py::class_<GeneratorOp, std::shared_ptr<GeneratorOp>>(*m, "GeneratorOp")
                    .def("GetMappedIndex", [](GeneratorOp &self, int64_t index) {
                      int64_t sample_id;
                      THROW_IF_ERROR(self.GetMappedIndex(index, &sample_id));
                      return sample_id;
                    });
                }));

PYBIND_REGISTER(ImageFolderNode, 2, ([](const py::module *m) {
                  (void)py::class_<ImageFolderNode, DatasetNode, std::shared_ptr<ImageFolderNode>>(
                    *m, "ImageFolderNode", "to create an ImageFolderNode")
                    .def(py::init([](const std::string &dataset_dir, bool decode, const py::handle &sampler,
                                     const py::list &extensions, const py::dict &class_indexing,
                                     const py::object &decrypt_obj) {
                      // Don't update recursive to true
                      bool recursive = false;  // Will be removed in future PR
                      py::function decrypt =
                        py::isinstance<py::function>(decrypt_obj) ? decrypt_obj.cast<py::function>() : py::function();
                      auto imagefolder = std::make_shared<ImageFolderNode>(
                        dataset_dir, decode, toSamplerObj(sampler), recursive, toStringSet(extensions),
                        toStringMap(class_indexing), nullptr, decrypt);
                      THROW_IF_ERROR(imagefolder->ValidateParams());
                      return imagefolder;
                    }));
                }));

PYBIND_REGISTER(IMDBNode, 2, ([](const py::module *m) {
                  (void)py::class_<IMDBNode, DatasetNode, std::shared_ptr<IMDBNode>>(*m, "IMDBNode",
                                                                                     "to create an IMDBNode")
                    .def(
                      py::init([](const std::string &dataset_dir, const std::string &usage, const py::handle &sampler) {
                        auto imdb = std::make_shared<IMDBNode>(dataset_dir, usage, toSamplerObj(sampler), nullptr);
                        THROW_IF_ERROR(imdb->ValidateParams());
                        return imdb;
                      }));
                }));

PYBIND_REGISTER(IWSLT2016Node, 2, ([](const py::module *m) {
                  (void)py::class_<IWSLT2016Node, DatasetNode, std::shared_ptr<IWSLT2016Node>>(
                    *m, "IWSLT2016Node", "to create an IWSLT2016Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::vector<std::string> &language_pair, const std::string &valid_set,
                                     const std::string &test_set, int64_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<IWSLT2016Node> iwslt2016 = std::make_shared<IWSLT2016Node>(
                        dataset_dir, usage, language_pair, valid_set, test_set, num_samples, toShuffleMode(shuffle),
                        num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(iwslt2016->ValidateParams());
                      return iwslt2016;
                    }));
                }));

PYBIND_REGISTER(IWSLT2017Node, 2, ([](const py::module *m) {
                  (void)py::class_<IWSLT2017Node, DatasetNode, std::shared_ptr<IWSLT2017Node>>(
                    *m, "IWSLT2017Node", "to create an IWSLT2017Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::vector<std::string> &language_pair, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<IWSLT2017Node> iwslt2017 =
                        std::make_shared<IWSLT2017Node>(dataset_dir, usage, language_pair, num_samples,
                                                        toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(iwslt2017->ValidateParams());
                      return iwslt2017;
                    }));
                }));

PYBIND_REGISTER(ManifestNode, 2, ([](const py::module *m) {
                  (void)py::class_<ManifestNode, DatasetNode, std::shared_ptr<ManifestNode>>(*m, "ManifestNode",
                                                                                             "to create a ManifestNode")
                    .def(py::init([](const std::string &dataset_file, const std::string &usage,
                                     const py::handle &sampler, const py::dict &class_indexing, bool decode) {
                      auto manifest = std::make_shared<ManifestNode>(dataset_file, usage, toSamplerObj(sampler),
                                                                     toStringMap(class_indexing), decode, nullptr);
                      THROW_IF_ERROR(manifest->ValidateParams());
                      return manifest;
                    }));
                }));

PYBIND_REGISTER(
  MindDataNode, 2, ([](const py::module *m) {
    (void)py::class_<MindDataNode, DatasetNode, std::shared_ptr<MindDataNode>>(*m, "MindDataNode",
                                                                               "to create a MindDataNode")
      .def(py::init([](const std::string &dataset_file, const py::list &columns_list, const py::handle &sampler,
                       const py::dict &padded_sample, int64_t num_padded, ShuffleMode shuffle_mode) {
        nlohmann::json padded_sample_json;
        std::map<std::string, std::string> sample_bytes;
        THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
        auto minddata =
          std::make_shared<MindDataNode>(dataset_file, toStringVector(columns_list), toSamplerObj(sampler, true),
                                         padded_sample_json, num_padded, shuffle_mode, nullptr);
        minddata->SetSampleBytes(&sample_bytes);
        THROW_IF_ERROR(minddata->ValidateParams());
        return minddata;
      }))
      .def(py::init([](const py::list &dataset_file, const py::list &columns_list, const py::handle &sampler,
                       const py::dict &padded_sample, int64_t num_padded, ShuffleMode shuffle_mode) {
        nlohmann::json padded_sample_json;
        std::map<std::string, std::string> sample_bytes;
        THROW_IF_ERROR(ToJson(padded_sample, &padded_sample_json, &sample_bytes));
        auto minddata = std::make_shared<MindDataNode>(toStringVector(dataset_file), toStringVector(columns_list),
                                                       toSamplerObj(sampler, true), padded_sample_json, num_padded,
                                                       shuffle_mode, nullptr);
        minddata->SetSampleBytes(&sample_bytes);
        THROW_IF_ERROR(minddata->ValidateParams());
        return minddata;
      }))
      .def("Build", [](MindDataNode &self) {
        std::vector<std::shared_ptr<DatasetOp>> node_ops;
        THROW_IF_ERROR(self.Build(&node_ops));
        auto mindrecord_op = std::dynamic_pointer_cast<MindRecordOp>(node_ops[0]);
        THROW_IF_ERROR(mindrecord_op->ShardReaderLaunch());
        return mindrecord_op;
      });
  }));

PYBIND_REGISTER(MindRecordOp, 0, ([](const py::module *m) {
                  (void)py::class_<MindRecordOp, std::shared_ptr<MindRecordOp>>(*m, "MindRecordOp")
                    .def("__getitem__", [](MindRecordOp &self, size_t index) {
                      auto out = self[index];
                      std::vector<TensorPtr> row;
                      row = out.getRow();
                      {
                        py::gil_scoped_acquire gil_acquire;
                        py::list result;
                        for (const auto &el : row) {
                          result.append(el);
                        }
                        return result;
                      }
                    });
                }));

PYBIND_REGISTER(Multi30kNode, 2, ([](const py::module *m) {
                  (void)py::class_<Multi30kNode, DatasetNode, std::shared_ptr<Multi30kNode>>(*m, "Multi30kNode",
                                                                                             "to create a Multi30kNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage,
                                     const std::vector<std::string> &language_pair, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<Multi30kNode> multi30k =
                        std::make_shared<Multi30kNode>(dataset_dir, usage, language_pair, num_samples,
                                                       toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(multi30k->ValidateParams());
                      return multi30k;
                    }));
                }));

PYBIND_REGISTER(PennTreebankNode, 2, ([](const py::module *m) {
                  (void)py::class_<PennTreebankNode, DatasetNode, std::shared_ptr<PennTreebankNode>>(
                    *m, "PennTreebankNode", "to create a PennTreebankNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int32_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto penn_treebank = std::make_shared<PennTreebankNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(penn_treebank->ValidateParams());
                      return penn_treebank;
                    }));
                }));

PYBIND_REGISTER(
  RandomNode, 2, ([](const py::module *m) {
    (void)py::class_<RandomNode, DatasetNode, std::shared_ptr<RandomNode>>(*m, "RandomNode", "to create a RandomNode")
      .def(py::init([](int32_t total_rows, const std::shared_ptr<SchemaObj> &schema, const py::list &columns_list) {
        auto random_node = std::make_shared<RandomNode>(total_rows, schema, toStringVector(columns_list), nullptr);
        THROW_IF_ERROR(random_node->ValidateParams());
        return random_node;
      }))
      .def(py::init([](int32_t total_rows, const std::string &schema, const py::list &columns_list) {
        auto random_node = std::make_shared<RandomNode>(total_rows, schema, toStringVector(columns_list), nullptr);
        THROW_IF_ERROR(random_node->ValidateParams());
        return random_node;
      }));
  }));

PYBIND_REGISTER(SogouNewsNode, 2, ([](const py::module *m) {
                  (void)py::class_<SogouNewsNode, DatasetNode, std::shared_ptr<SogouNewsNode>>(
                    *m, "SogouNewsNode", "to create a SogouNewsNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto sogou_news = std::make_shared<SogouNewsNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(sogou_news->ValidateParams());
                      return sogou_news;
                    }));
                }));

PYBIND_REGISTER(SQuADNode, 2, ([](const py::module *m) {
                  (void)py::class_<SQuADNode, DatasetNode, std::shared_ptr<SQuADNode>>(*m, "SQuADNode",
                                                                                       "to create a SQuADNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<SQuADNode> squad = std::make_shared<dataset::SQuADNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(squad->ValidateParams());
                      return squad;
                    }));
                }));

PYBIND_REGISTER(SST2Node, 2, ([](const py::module *m) {
                  (void)py::class_<SST2Node, DatasetNode, std::shared_ptr<SST2Node>>(*m, "SST2Node",
                                                                                     "to create a SST2Node")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto sst2 = std::make_shared<SST2Node>(dataset_dir, usage, num_samples, toShuffleMode(shuffle),
                                                             num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(sst2->ValidateParams());
                      return sst2;
                    }));
                }));

PYBIND_REGISTER(TextFileNode, 2, ([](const py::module *m) {
                  (void)py::class_<TextFileNode, DatasetNode, std::shared_ptr<TextFileNode>>(*m, "TextFileNode",
                                                                                             "to create a TextFileNode")
                    .def(py::init([](const py::list &dataset_files, int32_t num_samples, int32_t shuffle,
                                     int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<TextFileNode> textfile_node =
                        std::make_shared<TextFileNode>(toStringVector(dataset_files), num_samples,
                                                       toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(textfile_node->ValidateParams());
                      return textfile_node;
                    }));
                }));

PYBIND_REGISTER(
  TFRecordNode, 2, ([](const py::module *m) {
    (void)py::class_<TFRecordNode, DatasetNode, std::shared_ptr<TFRecordNode>>(*m, "TFRecordNode",
                                                                               "to create a TFRecordNode")
      .def(py::init([](const py::list &dataset_files, const std::shared_ptr<SchemaObj> &schema,
                       const py::list &columns_list, int64_t num_samples, int32_t shuffle, int32_t num_shards,
                       int32_t shard_id, bool shard_equal_rows, const std::string &compression_type) {
        std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
          toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples, toShuffleMode(shuffle),
          num_shards, shard_id, shard_equal_rows, nullptr, compression_type);
        THROW_IF_ERROR(tfrecord->ValidateParams());
        return tfrecord;
      }))
      .def(py::init([](const py::list &dataset_files, const std::string &schema, const py::list &columns_list,
                       int64_t num_samples, int32_t shuffle, int32_t num_shards, int32_t shard_id,
                       bool shard_equal_rows, const std::string &compression_type) {
        std::shared_ptr<TFRecordNode> tfrecord = std::make_shared<TFRecordNode>(
          toStringVector(dataset_files), schema, toStringVector(columns_list), num_samples, toShuffleMode(shuffle),
          num_shards, shard_id, shard_equal_rows, nullptr, compression_type);
        THROW_IF_ERROR(tfrecord->ValidateParams());
        return tfrecord;
      }));
  }));

PYBIND_REGISTER(UDPOSNode, 2, ([](const py::module *m) {
                  (void)py::class_<UDPOSNode, DatasetNode, std::shared_ptr<UDPOSNode>>(*m, "UDPOSNode",
                                                                                       "to create an UDPOSNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<UDPOSNode> udpos = std::make_shared<UDPOSNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(udpos->ValidateParams());
                      return udpos;
                    }));
                }));

PYBIND_REGISTER(WikiTextNode, 2, ([](const py::module *m) {
                  (void)py::class_<WikiTextNode, DatasetNode, std::shared_ptr<WikiTextNode>>(*m, "WikiTextNode",
                                                                                             "to create a WikiTextNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int32_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto wiki_text = std::make_shared<WikiTextNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(wiki_text->ValidateParams());
                      return wiki_text;
                    }));
                }));

PYBIND_REGISTER(YahooAnswersNode, 2, ([](const py::module *m) {
                  (void)py::class_<YahooAnswersNode, DatasetNode, std::shared_ptr<YahooAnswersNode>>(
                    *m, "YahooAnswersNode", "to create a YahooAnswersNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      auto yahoo_answers = std::make_shared<YahooAnswersNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(yahoo_answers->ValidateParams());
                      return yahoo_answers;
                    }));
                }));

PYBIND_REGISTER(YelpReviewNode, 2, ([](const py::module *m) {
                  (void)py::class_<YelpReviewNode, DatasetNode, std::shared_ptr<YelpReviewNode>>(
                    *m, "YelpReviewNode", "to create a YelpReviewNode")
                    .def(py::init([](const std::string &dataset_dir, const std::string &usage, int64_t num_samples,
                                     int32_t shuffle, int32_t num_shards, int32_t shard_id) {
                      std::shared_ptr<YelpReviewNode> yelp_review = std::make_shared<YelpReviewNode>(
                        dataset_dir, usage, num_samples, toShuffleMode(shuffle), num_shards, shard_id, nullptr);
                      THROW_IF_ERROR(yelp_review->ValidateParams());
                      return yelp_review;
                    }));
                }));

}  // namespace dataset
}  // namespace mindspore
