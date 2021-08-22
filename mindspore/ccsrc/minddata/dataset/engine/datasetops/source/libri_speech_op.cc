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

#include "minddata/dataset/engine/datasetops/source/libri_speech_op.h"


#include <fstream>
#include <iomanip>
#include <set>
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {

const uint32_t kAudioBufferSize = 20480;
const uint32_t kAudioRefillThresh = 4096;

LibriSpeechOp::LibriSpeechOp(const std::string &usage, int32_t num_workers, std::string folder_path, int32_t queue_size,
               std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      usage_(usage),
      folder_path_(folder_path),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers, queue_size);
}

Status LibriSpeechOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  LibriSpeechLabelTuple audio_tuple = audio_label_tuple_[row_id];
  std::shared_ptr <Tensor> waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id;

  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(audio_tuple.waveform, &waveform));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.sample_rate, &sample_rate));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.utterance, &utterance));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.speaker_id, &speaker_id));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.chapter_id, &chapter_id));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.utterance_id, &utterance_id));

  (*trow) = TensorRow(row_id,
            {std::move(waveform), std::move(sample_rate), std::move(utterance), std::move(speaker_id),
             std::move(chapter_id), std::move(utterance_id)});
  trow->setPath({flac_nodes_[row_id].file_link});
  return Status::OK();
}

void LibriSpeechOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  }
  else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nLibriSpeech Directory: " << folder_path_ << "\n\n";
  }
}

// Derived from RandomAccessOp
Status LibriSpeechOp::GetClassIds(std::map<uint32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || audio_label_tuple_.empty()) {
    if (audio_label_tuple_.empty()) {
      RETURN_STATUS_UNEXPECTED("No audio found in dataset, please check if Op read images successfully or not.");
    }
    else {
      RETURN_STATUS_UNEXPECTED(
          "Map for storaging image-index pair is nullptr or has been set in other place,"
          "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < audio_label_tuple_.size(); ++i) {
    (*cls_ids)[audio_label_tuple_[i].utterance_id].push_back(i);//
  }
  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}


Status LibriSpeechOp::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  // the logic of counting the number of samples is copied from ParseMnistData() and uses CheckReader()
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();

  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT64), TensorImpl::kCv, 1)));
  TensorShape scalar_rate = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("sample_rate", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0,
                      &scalar_rate)));
  TensorShape scalar_utterance = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("utterance", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0,
                      &scalar_utterance)));
  TensorShape scalar_speaker_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("speaker_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0,
                      &scalar_speaker_id)));
  TensorShape scalar_chapter_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("chapter_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0,
                      &scalar_chapter_id)));
  TensorShape scalar_utterance_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("utterance_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0,
                      &scalar_utterance_id)));

  std::shared_ptr <ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op = std::make_shared<LibriSpeechOp>(usage, num_workers, dir, op_connect_size, std::move(schema),
                        std::move(sampler));
  RETURN_IF_NOT_OK(op->WalkAllFiles());
  *count = op->flac_files_.size();
  return Status::OK();
}



Status LibriSpeechOp::DecodeFlac(AVCodecContext *dec_ctx, AVPacket *pkt, AVFrame *frame,std::vector<double> &arr) {
  int32_t i, ch;
  int32_t ret, data_size;

  ret = avcodec_send_packet(dec_ctx, pkt);
  if (ret < 0) {
    RETURN_STATUS_UNEXPECTED("Error submitting the packet to the decoder!");
  }

  while (ret >= 0) {
    ret = avcodec_receive_frame(dec_ctx, frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      return Status::OK();
    }
    else if (ret < 0) {
      RETURN_STATUS_UNEXPECTED("Error during decoding!");
    }
    data_size = av_get_bytes_per_sample(dec_ctx->sample_fmt);
    if (data_size < 0) {
      RETURN_STATUS_UNEXPECTED("Failed to calculate data size!");
    }
    for (i = 0; i < frame->nb_samples; i++)
      for (ch = 0; ch < dec_ctx->channels; ch++)
        arr.push_back((*(short *) (frame->data[ch] + data_size * i)) / 32768.0);
  }
  return Status::OK();
}


Status LibriSpeechOp::ComputeColMap() {
  // set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->column(i).name()] = i;
    }
  }
  else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status LibriSpeechOp::ReadLabel() {
  char buff[2048];
  for (auto u:label_files_) {
    std::ifstream in(u);
    while (!in.eof()) {
      in.getline(buff, 2048);
      if (buff[0] < '0' || buff[0] > '9')
        break;

      uint32_t blank[3] = {0};
      uint32_t cur = 0;
      uint32_t start = 0;
      for (uint32_t i = 0; i < 2048; i++) {
        if (buff[i] == '-')
          blank[cur++] = i;
        if (buff[i] == ' ') {
          start = i + 1;
          break;
        }
      }
      if (cur != 2)
        RETURN_STATUS_UNEXPECTED("Label file error!");
      uint32_t speaker_id = 0;
      uint32_t chapter_id = 0;
      uint32_t utterance_id = 0;
      for (uint32_t i = 0; i < blank[0]; i++)
        speaker_id = speaker_id * 10 + buff[i] - '0';
      for (uint32_t i = blank[0] + 1; i < blank[1]; i++)
        chapter_id = chapter_id * 10 + buff[i] - '0';
      for (uint32_t i = blank[1] + 1; i < start - 1; i++)
        utterance_id = utterance_id * 10 + buff[i] - '0';
      buff[start - 1] = 0;
      flac_nodes_.push_back({std::string(buff), std::string(buff + start), speaker_id, chapter_id, utterance_id});
    }
  }

  std::sort(flac_files_.begin(), flac_files_.end());
  std::sort(flac_nodes_.begin(), flac_nodes_.end(),
        [&](flac_node a, flac_node b) { return a.file_link < b.file_link; });
  for (uint32_t i = 0; i < flac_files_.size(); i++) {
    if (flac_nodes_[i].file_link != flac_files_[i].first) {
      RETURN_STATUS_UNEXPECTED("An error occurred between the label and the file content!");
    }
    flac_nodes_[i].file_link = flac_files_[i].second;
  }
  return Status::OK();
}

Status LibriSpeechOp::ReadAudio() {

  for (flac_node u:flac_nodes_) {
    std::vector<double> arr;
    char *filename = u.file_link.data();
    const AVCodec *codec;

    AVCodecContext *c = NULL;
    AVCodecParserContext *parser = NULL;
    AVPacket *pkt;
    AVFrame *decoded_frame = NULL;
    FILE *f;

    int32_t len, ret;
    uint8_t inbuf[kAudioBufferSize + AV_INPUT_BUFFER_PADDING_SIZE];
    uint8_t *data;
    size_t data_size;

    pkt = av_packet_alloc();
    codec = avcodec_find_decoder(AV_CODEC_ID_FLAC);
    if (!codec) {
      RETURN_STATUS_UNEXPECTED("Codec not found!");
    }
    parser = av_parser_init(codec->id);
    if (!parser) {
      RETURN_STATUS_UNEXPECTED("Parser not found!");
    }
    c = avcodec_alloc_context3(codec);
    if (!c) {
      RETURN_STATUS_UNEXPECTED("Could not allocate audio codec context!");
    }
    if (avcodec_open2(c, codec, NULL) < 0) {
      RETURN_STATUS_UNEXPECTED("Could not open codec!");
    }

    f = fopen(filename, "rb");
    if (!f) {
      RETURN_STATUS_UNEXPECTED(std::string("Could not open ") + filename);
    }

    data = inbuf;
    data_size = fread(inbuf, 1, kAudioBufferSize, f);

    decoded_frame = av_frame_alloc();
    while (true) {
      pkt->size = 0;
      pkt->data = nullptr;
      ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size,
                   data, data_size,
                   AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);

      if (pkt->size == 0 && data_size == 0)
        break;
      if (ret < 0) {
        RETURN_STATUS_UNEXPECTED("Error while parsing");
      }
      data += ret;
      data_size -= ret;
      if (pkt->size) {
        RETURN_IF_NOT_OK(DecodeFlac(c, pkt, decoded_frame, arr));
      }

      if (data_size < kAudioRefillThresh) {
        memmove(inbuf, data, data_size);
        data = inbuf;
        len = fread(data + data_size, 1,
              kAudioBufferSize - data_size, f);
        if (len > 0)
          data_size += len;
      }
    }

    pkt->size = 0;
    pkt->data = nullptr;
    RETURN_IF_NOT_OK(DecodeFlac(c, pkt, decoded_frame, arr));
    uint32_t rate = c->sample_rate;
    fclose(f);
    avcodec_free_context(&c);
    av_parser_close(parser);
    av_frame_free(&decoded_frame);
    av_packet_free(&pkt);
    std::shared_ptr <Tensor> audio;
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(arr, &audio));
    audio_label_tuple_.push_back({audio, rate, u.utterance, u.speaker_id, u.speaker_id, u.utterance_id});
  }
  num_rows_ = audio_label_tuple_.size();
  return Status::OK();
}

Status LibriSpeechOp::WalkAllFiles() {
  Path dir(folder_path_);
  Path fullDir = dir + usage_;
  auto dirIt = Path::DirIterator::OpenDirectory(&fullDir);
  if (dirIt != nullptr) {
    while (dirIt->hasNext()) {
      Path file = dirIt->next();

      auto subDirIt = Path::DirIterator::OpenDirectory(&file);
      if (subDirIt != nullptr) {
        while (subDirIt->hasNext()) {
          Path subFile = subDirIt->next();

          auto leafDirIt = Path::DirIterator::OpenDirectory(&subFile);
          if (leafDirIt != nullptr) {
            while (leafDirIt->hasNext()) {
              Path actFile = leafDirIt->next();
              std::string p = actFile.toString();
              size_t pos = p.size() - 3;
              size_t len = actFile.Basename().size() - 5;
              if (pos < 0 || len < 0)
                RETURN_STATUS_UNEXPECTED("File name parsing error!");
              std::string t = p.substr(pos);
              if (t == "lac") {
                flac_files_.push_back({actFile.Basename().substr(0, len), p});
              }
              else if (t == "txt") {
                label_files_.push_back(p);
              }
              else {
                MS_LOG(WARNING) << "File name format error :" << actFile.toString() << ".";
              }
            }
          }//leafDirIt

        }
      }//subDirIt

    }
  }//DirIt
  else {
    MS_LOG(WARNING) << "Unable to open directory " << fullDir.toString() << ".";
  }
  return Status::OK();
}

Status LibriSpeechOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
      tree_->LaunchWorkers(num_workers_, std::bind(&LibriSpeechOp::WorkerEntry, this, std::placeholders::_1), "",
                 id()));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->WalkAllFiles());
  RETURN_IF_NOT_OK(this->ReadLabel());
  RETURN_IF_NOT_OK(this->ReadAudio());
  RETURN_IF_NOT_OK(this->InitSampler());  // handle shake with sampler
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
