/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LIBRISPEECH_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LIBRISPEECH_OP_H_

extern "C"{
  #include <libavutil/frame.h>
  #include <libavutil/mem.h>
  #include <libavutil/file.h>
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libavformat/avio.h>
}

#include <memory>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>

#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {

struct LibriSpeechLabelTuple{
  std::shared_ptr<Tensor> waveform;
  uint32_t sample_rate;
  std::string utterance;
  uint32_t speaker_id;
  uint32_t chapter_id;
  uint32_t utterance_id;
};

struct flac_node {
  std::string file_link;
  std::string utterance;
  uint32_t speaker_id;
  uint32_t chapter_id;
  uint32_t utterance_id;
};

class LibriSpeechOp : public MappableLeafOp {
 public:
  // Constructor
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test' ,'valid'or 'all'
  // @param int32_t num_workers - number of workers reading audios in parallel
  // @param std::string folder_path - dir directory of LibriSppech
  // @param int32_t queue_size - connector queue size
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the LibriSppech dataset
  // @param td::unique_ptr<Sampler> sampler - sampler tells LibriSpeechOp what to read
  LibriSpeechOp(const std::string &usage, int32_t num_workers, std::string folder_path, int32_t queue_size,
          std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  // Destructor.
  ~LibriSpeechOp() = default;

  // Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  // @param (std::map<uint64_t, std::vector<uint64_t >> * map - key label, val all ids for this class
  // @return Status The status code returned
  Status GetClassIds(std::map<uint32_t, std::vector<int64_t>> *cls_ids) const ;

  // A print method typically used for debugging
  // @param out
  // @param show_all
  void Print(std::ostream &out, bool show_all) const override;

  // Function to count the number of samples in the LibriSppech dataset
  // @param dir path to the LibriSppech directory
  // @param count output arg that will hold the minimum of the actual dataset size and numSamples
  // @return

 static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  // Op name getter
  // @return Name of the current Op
 std::string Name() const override { return "LibriSpeechOp"; }

 private:
  Status DecodeFlac(AVCodecContext *dec_ctx, AVPacket *pkt, AVFrame *frame,std::vector<double> &arr);
  
  // Load a tensor row according to a pair
  // @param row_id_type row_id - id for this tensor row
  // @param ImageLabelPair pair - <audiofile,label>
  // @param TensorRow row - audio & label read into this tensor row
  // @return Status The status code returned
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  Status ReadAudio();

  Status ReadLabel();

  // Read all files in the directory
  // @return Status The status code returned
  Status WalkAllFiles();

  // Called first when function is called
  // @return Status The status code returned
  Status LaunchThreadsAndInitOp() override;

  // Private function for computing the assignment of the column name map.
  // @return - Status
  Status ComputeColMap() override;


  std::string folder_path_;  // directory of audio folder
  const std::string usage_;  // can only be either "train" or "test"
  
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<LibriSpeechLabelTuple> audio_label_tuple_;

  std::vector<std::string>label_files_;
  std::vector<std::pair<std::string, std::string>>flac_files_;
  std::vector<flac_node>flac_nodes_;
};



}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_LIBRISPEECH_OP_H_
