/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/video_utils.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
};
#endif

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "utils/file_utils.h"

const int32_t MAX_AVIO_BUFFER_SIZE = 1073741824;

namespace mindspore {
namespace dataset {
struct MediaContainer {
  int channels = 1;

  AVCodecContext *codec_context = nullptr;

  AVFrame *frame = nullptr;
  int64_t frame_number = 0;
  float frame_rate = 1.0;
  int64_t samples_count = 0;

  AVStream *stream = nullptr;
  int stream_index = -1;

  float time_base = 1.0;
  int timestamp_increment = 1;

  // start_frame_number: the frame number corresponding to the start_pts
  int64_t start_frame_number = 0;
  int64_t start_pts = 0;

  // end_frame_number: the frame number corresponding to the end_pts
  int64_t end_frame_number = 0;
  int64_t end_pts = 0;

  int64_t current_pts = 0;

  // for the calculation of current_pts
  int adjust_pts_flag = 0;

  // for the conversion of frames
  int conversion_flags = 0;

  // whether find any error in this container
  bool find_error = false;

  ~MediaContainer() {
    if (codec_context) {
      avcodec_free_context(&codec_context);
      codec_context = nullptr;
    }

    if (frame) {
      av_frame_free(&frame);
      frame = nullptr;
    }
  }
};

struct TensorData {
  const uint8_t *ptr;
  // size left in the tensor
  size_t size;
};

struct AudioVisual {
  AVFormatContext *avformat = nullptr;
  AVPacket *packet = nullptr;

  // for audio
  struct MediaContainer audio;
  AVSampleFormat sample_format = AV_SAMPLE_FMT_NONE;
  int sample_rate = 1;
  int64_t nb_samples = 0;
  SwrContext *audio_conversion = nullptr;

  // for visual
  struct MediaContainer visual;
  int image_height = 1;
  int image_width = 1;
  std::map<int64_t, std::shared_ptr<Tensor>> map_visual_tensor;
  SwsContext *visual_conversion = nullptr;
  uint8_t *visual_aligned_buffer[kMaxImageChannel] = {nullptr, nullptr, nullptr, nullptr};
  int visual_aligned_linesize[kMaxImageChannel] = {0, 0, 0, 0};
  uint8_t *visual_unaligned_buffer = nullptr;

  // for DecodeVideo op
  struct TensorData tensor_data = {nullptr, 0};
  uint8_t *avio_context_buffer = nullptr;
  AVIOContext *avio_context = nullptr;

  // for debug and the log_level of FFMPEG
  int av_log_level = AV_LOG_FATAL;

  ~AudioVisual() {
    if (avio_context) {
      av_freep(&avio_context->buffer);
      avio_context_free(&avio_context);
    }

    if (packet) {
      av_packet_free(&packet);
      packet = nullptr;
    }

    if (audio_conversion) {
      swr_free(&audio_conversion);
      audio_conversion = nullptr;
    }

    // Free the allocated memory by swap to an empty.
    std::map<int64_t, std::shared_ptr<Tensor>>().swap(map_visual_tensor);

    if (visual_unaligned_buffer) {
      free(visual_unaligned_buffer);
      visual_unaligned_buffer = nullptr;
    }

    if (visual_aligned_buffer[0]) {
      av_freep(visual_aligned_buffer);
      visual_aligned_buffer[0] = nullptr;
    }
    if (visual_conversion) {
      sws_freeContext(visual_conversion);
      visual_conversion = nullptr;
    }

    if (avformat) {
      avformat_close_input(&avformat);
      avformat = nullptr;
    }
  }
};

int avio_read_buffer(void *opaque, uint8_t *buf, int buf_size) {
  struct TensorData *tensor_data = (struct TensorData *)opaque;
  buf_size = FFMIN(buf_size, tensor_data->size);
  if (!buf_size) {
    return AVERROR_EOF;
  }
  memcpy_s(buf, buf_size, tensor_data->ptr, buf_size);
  tensor_data->ptr += buf_size;
  tensor_data->size -= buf_size;
  return buf_size;
}

Status AVOpenMemoryRead(const TensorRow &input, struct AudioVisual *avinfo) {
  std::string err_msg;
  if (input.size() != 1) {
    err_msg = "The input has invalid size " + std::to_string(input.size());
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (input[0]->type() != DataType::DE_UINT8) {
    err_msg = "The type of the elements of input data should be UINT8, but got " + input[0]->type().ToString() + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  int64_t data_size = input[0]->Size();
  if (data_size >= kDeMaxDim || data_size <= 0) {
    err_msg = "The input[0] has invalid size " + std::to_string(data_size);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  const uint8_t *data_buffer;
  data_buffer = (const uint8_t *)input[0]->GetBuffer();
  avinfo->tensor_data.ptr = data_buffer;
  avinfo->tensor_data.size = data_size;

  // Allocate the avformat
  avinfo->avformat = avformat_alloc_context();
  if (avinfo->avformat == nullptr) {
    err_msg = "Failed to call avformat_alloc_context().";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  // Set the flags so that it will discard corrupt frames
  avinfo->avformat->flags |= AVFMT_FLAG_DISCARD_CORRUPT;

  int avio_buffer_size = FFMIN(data_size, MAX_AVIO_BUFFER_SIZE);
  avinfo->avio_context_buffer = reinterpret_cast<uint8_t *>(av_malloc(avio_buffer_size));
  if (avinfo->avio_context_buffer == nullptr) {
    err_msg = "Failed to allocate buffer for avio_context";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  avinfo->avio_context = avio_alloc_context(avinfo->avio_context_buffer, avio_buffer_size, 0, &avinfo->tensor_data,
                                            &avio_read_buffer, NULL, NULL);
  if (avinfo->avio_context == nullptr) {
    err_msg = "Failed to allocate avio_context";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  avinfo->avformat->pb = avinfo->avio_context;

  // Open the input from memory
  if (avformat_open_input(&(avinfo->avformat), nullptr, nullptr, nullptr) < 0) {
    err_msg = "Failed to open the input.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Find all the streams
  if (avformat_find_stream_info(avinfo->avformat, nullptr) < 0) {
    err_msg = "Failed to find stream information.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

// Check the filename and open it by FFMPEG
Status AVOpenFile(const std::string &filename, struct AudioVisual *avinfo) {
  std::string err_msg;
  // check the input parameter: filename
  auto realpath = FileUtils::GetRealPath(filename.c_str());
  if (!realpath.has_value()) {
    err_msg = "Invalid file path, " + filename + " does not exist.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  struct stat sb;
  stat(realpath.value().c_str(), &sb);
  if (S_ISREG(sb.st_mode) == 0) {
    err_msg = "Invalid file path, " + filename + " is not a regular file.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (sb.st_size <= 0) {
    err_msg = filename + " has invalid file size " + std::to_string(sb.st_size);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Allocate the avformat
  avinfo->avformat = avformat_alloc_context();
  if (avinfo->avformat == nullptr) {
    err_msg = "Failed to call avformat_alloc_context().";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  // Set the flags so that it will discard corrupt frames
  avinfo->avformat->flags |= AVFMT_FLAG_DISCARD_CORRUPT;

  // Open the video file.
  if (avformat_open_input(&(avinfo->avformat), realpath.value().c_str(), nullptr, nullptr) < 0) {
    err_msg = "Failed to open the file " + filename + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Find all the streams
  if (avformat_find_stream_info(avinfo->avformat, nullptr) < 0) {
    err_msg = "Failed to find stream information.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

// Find one audio stream
int AVFindAudioStream(struct AudioVisual *avinfo) {
  int stream_index = av_find_best_stream(avinfo->avformat, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
  if (stream_index >= 0 && stream_index < avinfo->avformat->nb_streams) {
    avinfo->audio.stream_index = stream_index;
    avinfo->audio.stream = avinfo->avformat->streams[stream_index];
  }
  return stream_index;
}

// Find one visual stream
int AVFindVisualStream(struct AudioVisual *avinfo) {
  int stream_index = av_find_best_stream(avinfo->avformat, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (stream_index >= 0 && stream_index < avinfo->avformat->nb_streams) {
    avinfo->visual.stream_index = stream_index;
    avinfo->visual.stream = avinfo->avformat->streams[stream_index];
  }
  return stream_index;
}

// Check and convert AVRational to float for the time_base
Status AVCalculateAudioTimeBase(struct AudioVisual *avinfo) {
  // Use the time_base of codec_context
  // The time_base of stream is different from that of codec_context
  // The stream->time_base.den is channels * bits * codec_context->time_base.den
  if (avinfo->audio.codec_context->time_base.den > 0) {
    avinfo->audio.time_base = static_cast<float>(av_q2d(avinfo->audio.codec_context->time_base));
    return Status::OK();
  }
  RETURN_STATUS_UNEXPECTED("Failed to calculate the audio time base.");
}

// Check and calculate the frame rate
Status AVCalculateAudioFrameRate(struct AudioVisual *avinfo) {
  AVStream *stream = avinfo->audio.stream;
  if (stream->avg_frame_rate.den > 0) {
    avinfo->audio.frame_rate = static_cast<float>(av_q2d(stream->avg_frame_rate));
  } else {
    if (stream->time_base.den > 0 && stream->time_base.num > 0) {
      // use the time_base to calculate the frame_rate
      // frame_rate = 1.0 / time_base = time_base.den / time_base.num
      stream->avg_frame_rate.num = stream->time_base.den;
      stream->avg_frame_rate.den = stream->time_base.num;
      avinfo->audio.frame_rate = static_cast<float>(stream->time_base.den / stream->time_base.num);
    } else {
      RETURN_STATUS_UNEXPECTED("Failed to calculate the audio frame rate.");
    }
  }
  return Status::OK();
}

// Check and convert AVRational to float for the time_base
Status AVCalculateVisualTimeBase(struct AudioVisual *avinfo) {
  if (avinfo->visual.stream->time_base.den > 0) {
    avinfo->visual.time_base = static_cast<float>(av_q2d(avinfo->visual.stream->time_base));
    return Status::OK();
  }
  RETURN_STATUS_UNEXPECTED("Failed to calculate the visual time base.");
}

// Check and calculate the frame rate
Status AVCalculateVisualFrameRate(struct AudioVisual *avinfo) {
  AVStream *stream = avinfo->visual.stream;
  if (stream->avg_frame_rate.den > 0) {
    avinfo->visual.frame_rate = static_cast<float>(av_q2d(stream->avg_frame_rate));
  } else {
    if (stream->time_base.den > 0 && stream->time_base.num > 0) {
      // use the time_base to calculate the frame_rate
      // frame_rate = 1.0 / time_base = time_base.den / time_base.num
      stream->avg_frame_rate.num = stream->time_base.den;
      stream->avg_frame_rate.den = stream->time_base.num;
      avinfo->visual.frame_rate = static_cast<float>(stream->time_base.den / stream->time_base.num);
    } else {
      RETURN_STATUS_UNEXPECTED("Failed to calculate the visual frame rate.");
    }
  }
  return Status::OK();
}

// Check whether the extradata contains search_string
bool FindStringInCodecContextExtradata(const AVCodecContext *codec_context, const char *search_string) {
  int extradata_size = codec_context->extradata_size;
  if (extradata_size <= 0) {
    return false;
  }

  int i;
  int len = strlen(search_string);

  // Locate the start position of the last string in extradata
  for (i = extradata_size - len; i > 0; i--) {
    if (codec_context->extradata[i - 1] == 0) {
      break;
    }
  }

  if (strstr((const char *)(&(codec_context->extradata[i])), search_string)) {
    return true;
  }
  return false;
}

// Open the coder or decoder for the stream
Status AVOpenStreamCodecContext(struct MediaContainer *container, bool enable_thread_frame, bool enable_fast) {
  AVStream *stream = container->stream;
  AVCodecID codec_id = stream->codecpar->codec_id;
  std::string err_msg;
  if (codec_id == AV_CODEC_ID_NONE) {
    err_msg = "Failed to support AV_CODEC_ID_NONE.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  const AVCodec *codec = avcodec_find_decoder(codec_id);
  if (codec == nullptr) {
    err_msg = "Failed to find a proper codec for " + std::string(avcodec_get_name(codec_id));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  AVCodecContext *codec_context = avcodec_alloc_context3(codec);
  if (codec_context == nullptr) {
    err_msg = "Failed to allocate the " + std::string(codec->name);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Set the codec's parameters according to the stream
  if (stream->codecpar == nullptr) {
    err_msg = "The stream->codepar is nullptr";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (avcodec_parameters_to_context(codec_context, stream->codecpar) < 0) {
    err_msg = "Failed to set the parameters of " + std::string(codec->name);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  container->codec_context = codec_context;

  // thread_count = 0 means automatically.
  codec_context->thread_count = 0;
  codec_context->thread_type = FF_THREAD_SLICE;
  if (enable_thread_frame == true && (codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) > 0) {
    if (stream->nb_frames > 1) {
      codec_context->thread_type |= FF_THREAD_FRAME;
    }
  }

  // Some decoders need unaligned memory access
  codec_context->flags |= AV_CODEC_FLAG_UNALIGNED;
  if (enable_fast == true) {
    codec_context->flags2 |= AV_CODEC_FLAG2_FAST;
  }

  // Open the codec
  codec_context->pkt_timebase = codec_context->time_base;
  if (avcodec_open2(codec_context, codec, nullptr) < 0) {
    err_msg = "Failed to open the codec " + std::string(codec->name);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

// Allocate a packet and check it is not nullptr
Status AVPacketAllocate(struct AudioVisual *avinfo) {
  if (avinfo->packet == nullptr) {
    avinfo->packet = av_packet_alloc();
    if (avinfo->packet == nullptr) {
      RETURN_STATUS_UNEXPECTED("Failed to allocate packet.");
    }
  }
  return Status::OK();
}

// Allocate a frame and check it is not nullptr
Status AVFrameAllocate(struct AudioVisual *avinfo, struct MediaContainer *container) {
  if (container->frame == nullptr) {
    container->frame = av_frame_alloc();
    if (container->frame == nullptr) {
      RETURN_STATUS_UNEXPECTED("Failed to allocate visual frame.");
    }
  }
  return Status::OK();
}

// Read the presentation time stamps of a visual stream by decoding each frame
Status AVReadVisualPtsByFrame(struct AudioVisual *avinfo, struct MediaContainer *container,
                              std::vector<int64_t> *pts_int64_vector) {
  int status = 0;
  int64_t frame_number = 0;
  int64_t packet_number = 0;
  AVFormatContext *avformat = avinfo->avformat;
  AVPacket *packet = avinfo->packet;
  int stream_index = container->stream_index;
  AVCodecContext *decoder_context = container->codec_context;
  AVFrame *frame = container->frame;
  int timestamp_increment = container->timestamp_increment;
  char err_buf[AV_ERROR_MAX_STRING_SIZE];
  int64_t pts_number = 0;
  int adjust_pts_flag = 0;
  std::string err_msg;

  // Read a packet
  status = av_read_frame(avformat, packet);
  while (status >= 0) {
    if (packet->stream_index == stream_index) {
      // Send it to decoder
      status = avcodec_send_packet(decoder_context, packet);
      if (status < 0) {
        // The decoder failed to receive the packet
        av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, status);
        err_msg = "Failed to receive packet " + std::to_string(packet_number) + ". ";
        err_msg += std::string(err_buf);
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      while (status >= 0) {
        // Read a frame from the decoder
        status = avcodec_receive_frame(decoder_context, frame);
        if (status < 0) {
          if (status != AVERROR_EOF && status != AVERROR(EAGAIN)) {
            // Failed to receive frame from the decoder
            av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, status);
            err_msg = "Failed to receive frame " + std::to_string(frame_number) + ". ";
            err_msg += std::string(err_buf);
            RETURN_STATUS_UNEXPECTED(err_msg);
          }
          break;
        }
        // frame_number = 0 is the first frame
        // frame_number = 1 is the second frame
        // check whether the pts_number should be adjusted
        // for example
        // the frame_number       :  0, 1, 2, 3, 4, ...
        // the frame->pkt_dts     :  2, 3, 4, 5, 6, ...
        // should be adjusted to  :  1, 2, 3, 4, 5, ... when frame->pkt_dts == frame->coded_picture_number
        //                        :  1, 3, 4, 5, 6, ... when frame->pkt_dts != frame->coded_picture_number
        // the adjust method is let it - 1 from the second frame
        // we check the condition on the second frame, the first frame will be adjusted directly
        // The condition can be described as:
        // frame_number == 1 : this is the second frame
        // timestamp_increment == 1 : the timestamp is incremented by 1
        // frame->pts == AV_NOPTS_VALUE : there is no valid pts value
        // frame->pkt_dts == frame->coded_picture_number : the frame->pkt_dts is same as frame->coded_picture_number
        // frame->pkt_dts -1 == frame_number + 1 : when the frame->pkt_dts is 3 for the second frame
        if (frame_number == 1 && timestamp_increment == 1 && frame->pts == AV_NOPTS_VALUE &&
            frame->pkt_dts == frame->coded_picture_number && frame->pkt_dts - 1 == frame_number + 1) {
          adjust_pts_flag = 1;
        }

        pts_number = frame->best_effort_timestamp;
        if (pts_number == AV_NOPTS_VALUE) {
          pts_number = frame_number * timestamp_increment;
        } else {
          if (timestamp_increment == 1) {
            if (frame_number == 0) {
              if (pts_number > 1) {
                pts_number = 1;
              }
            } else {
              if (adjust_pts_flag) {
                pts_number -= 1;
              }
            }
          }
        }
        pts_int64_vector->push_back(pts_number);
        frame_number++;
        av_frame_unref(frame);
      }
    }
    packet_number++;
    av_packet_unref(packet);
    status = av_read_frame(avformat, packet);
  }
  avinfo->visual.current_pts = pts_number;
  avinfo->visual.adjust_pts_flag = adjust_pts_flag;

  return Status::OK();
}

Status AVReadVisualPtsByPacket(struct AudioVisual *avinfo, struct MediaContainer *container,
                               std::vector<int64_t> *pts_int64_vector) {
  int status = 0;
  int64_t packet_number = 0;
  int64_t pts_number = 0;
  std::string err_msg;

  AVFormatContext *avformat = avinfo->avformat;
  AVPacket *packet = avinfo->packet;
  int stream_index = container->stream_index;

  status = av_read_frame(avformat, packet);
  while (status >= 0) {
    if (packet->stream_index == stream_index) {
      pts_number = packet->pts;
      // Check the pts_number, make sure it is a valid number
      if (pts_number == AV_NOPTS_VALUE) {
        err_msg = "Failed to skip frame because there is no pts value for packet " + std::to_string(packet_number);
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      pts_int64_vector->push_back(pts_number);
    }
    av_packet_unref(packet);
    packet_number++;
    status = av_read_frame(avformat, packet);
  }
  avinfo->visual.current_pts = pts_number;
  return Status::OK();
}

// Flush the decoder and read it's pts
Status AVReadVisualPtsFlushDecoder(struct AudioVisual *avinfo, struct MediaContainer *container,
                                   std::vector<int64_t> *pts_int64_vector) {
  int status;
  std::string err_msg;

  int64_t *pts_number = &(avinfo->visual.current_pts);

  AVCodecContext *decoder_context = container->codec_context;
  AVFrame *frame = container->frame;
  int timestamp_increment = container->timestamp_increment;

  status = avcodec_send_packet(decoder_context, nullptr);

  while (status >= 0) {
    status = avcodec_receive_frame(decoder_context, frame);
    if (status < 0) {
      if (status != AVERROR_EOF && status != AVERROR(EAGAIN)) {
        err_msg = "Failed to receive frame during the flushing of the decoders.";
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      break;
    }
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
      *pts_number = frame->best_effort_timestamp;
    } else {
      *pts_number += timestamp_increment;
    }
    pts_int64_vector->push_back(*pts_number);
    av_frame_unref(frame);
  }
  return Status::OK();
}

Status AVReadVisualPts(struct AudioVisual *avinfo, std::vector<int64_t> *pts_int64_vector, float *video_fps,
                       float *time_base) {
  // Calculate the time_base
  RETURN_IF_NOT_OK(AVCalculateVisualTimeBase(avinfo));
  *time_base = avinfo->visual.time_base;

  // Calculate the frame_rate
  RETURN_IF_NOT_OK(AVCalculateVisualFrameRate(avinfo));
  *video_fps = avinfo->visual.frame_rate;

  // Calculate the timestampe_increment
  AVStream *stream = avinfo->visual.stream;
  int timestamp_increment =
    (stream->time_base.den * stream->avg_frame_rate.den) / (stream->time_base.num * stream->avg_frame_rate.num);
  avinfo->visual.timestamp_increment = timestamp_increment;

  bool skip_frame = FindStringInCodecContextExtradata(avinfo->visual.codec_context, "Lavc");
  if (skip_frame) {
    RETURN_IF_NOT_OK(AVReadVisualPtsByPacket(avinfo, &(avinfo->visual), pts_int64_vector));
  } else {
    RETURN_IF_NOT_OK(AVReadVisualPtsByFrame(avinfo, &(avinfo->visual), pts_int64_vector));
  }

  // Flush the decoders
  return AVReadVisualPtsFlushDecoder(avinfo, &(avinfo->visual), pts_int64_vector);
}

Status AVDecodeVisualFrame(struct AudioVisual *avinfo, std::shared_ptr<Tensor> *output) {
  int status = 0;
  int channels = avinfo->visual.channels;
  AVFrame *frame = avinfo->visual.frame;

  int width = avinfo->image_width;
  int height = avinfo->image_height;

  std::string err_msg;
  if (frame->height != height || frame->width != width) {
    err_msg = "image->height = " + std::to_string(height);
    err_msg += " image->width = " + std::to_string(width);
    err_msg += ". They are changed at visual frame " + std::to_string(avinfo->visual.frame_number) + ".";
    err_msg += " frame->hegiht = " + std::to_string(frame->height);
    err_msg += " frame->width = " + std::to_string(frame->width);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  SwsContext *conversion = avinfo->visual_conversion;
  status = sws_scale(conversion, frame->data, frame->linesize, 0, height, avinfo->visual_aligned_buffer,
                     avinfo->visual_aligned_linesize);
  if (status < 1) {
    err_msg = "Failed to call sws_scale at visual frame " + std::to_string(avinfo->visual.frame_number);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  int linesize = avinfo->visual_aligned_linesize[0];
  // Use the aligned buffer to create Tensor.
  if (linesize == width * channels) {
    return Tensor::CreateFromMemory(TensorShape({height, width, channels}), (const DataType)DataType::DE_UINT8,
                                    (const uchar *)avinfo->visual_aligned_buffer[0], output);
  }

  // Use the unaligned buffer to create Tensor.
  // Ignore the unused padding bytes for each line.
  int i;
  uint8_t *src_address = avinfo->visual_aligned_buffer[0];
  uint8_t *target_address = avinfo->visual_unaligned_buffer;
  int src_offset = linesize;
  int target_offset = width * channels;
  for (i = 0; i < height; i++) {
    if (target_offset < SECUREC_MEM_MAX_LEN) {
      int ret_code = memcpy_s(target_address, target_offset, src_address, target_offset);
      CHECK_FAIL_RETURN_UNEXPECTED(
        ret_code == EOK, "Failed to copy data into tensor, memcpy_s errorno: " + std::to_string(ret_code) + ".");
    } else {
      auto ret_code = std::memcpy(target_address, src_address, target_offset);
      CHECK_FAIL_RETURN_UNEXPECTED(ret_code == target_address, "Failed to copy data into tensor.");
    }
    src_address += src_offset;
    target_address += target_offset;
  }
  return Tensor::CreateFromMemory(TensorShape({height, width, channels}), (const DataType)DataType::DE_UINT8,
                                  (const uchar *)avinfo->visual_unaligned_buffer, output);
}

template <typename T>
void AVDecodeAudioFramePacket(struct AudioVisual *avinfo, std::vector<std::vector<T>> *audio_vector) {
  AVFrame *frame = avinfo->audio.frame;

  int channels = frame->channels;
  int64_t nb_samples = frame->nb_samples;
  int out_buffer_size = nb_samples * sizeof(T);

  const uint8_t **out_data = (const uint8_t **)(frame->extended_data);

  if (audio_vector->size() == 0) {
    audio_vector->resize(avinfo->audio.channels, std::vector<T>());
  }

  const T *p_data = nullptr;
  for (int channel = 0; channel < channels; channel++) {
    for (int i = channel * sizeof(T); i < out_buffer_size + channel * sizeof(T); i += sizeof(T)) {
      p_data = (const T *)(out_data[0] + i);
      (*audio_vector)[channel].push_back(*p_data);
    }
  }
}

template <typename T>
void AVDecodeAudioFramePlanar(struct AudioVisual *avinfo, std::vector<std::vector<T>> *audio_vector) {
  AVFrame *frame = avinfo->audio.frame;

  int channels = frame->channels;
  int64_t nb_samples = frame->nb_samples;

  const uint8_t **out_data = (const uint8_t **)(frame->extended_data);

  if (audio_vector->size() == 0) {
    audio_vector->resize(avinfo->audio.channels, std::vector<T>());
  }

  for (int channel = 0; channel < channels; channel++) {
    const T *p_data = reinterpret_cast<const T *>(out_data[channel]);
    (*audio_vector)[channel].insert((*audio_vector)[channel].end(), p_data, p_data + nb_samples);
  }
}

Status AVAllocateAudioConversion(struct AudioVisual *avinfo, AVSampleFormat out_sample_format) {
  std::string err_msg;
  if (avinfo->audio_conversion != nullptr) {
    err_msg = "audio_conversion should be freed firstly.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  AVFrame *frame = avinfo->audio.frame;
  int sample_rate = frame->sample_rate;
  int64_t channel_layout = frame->channel_layout;

  AVSampleFormat in_sample_format = (AVSampleFormat)(frame->format);

  SwrContext *swr_context = swr_alloc();
  if (swr_context == nullptr) {
    err_msg = "Failed to get audio convert context.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  av_opt_set_int(swr_context, "in_channel_layout", channel_layout, 0);
  av_opt_set_int(swr_context, "out_channel_layout", channel_layout, 0);
  av_opt_set_int(swr_context, "in_sample_rate", sample_rate, 0);
  av_opt_set_int(swr_context, "out_sample_rate", sample_rate, 0);
  av_opt_set_sample_fmt(swr_context, "in_sample_fmt", in_sample_format, 0);
  av_opt_set_sample_fmt(swr_context, "out_sample_fmt", out_sample_format, 0);

  int result = swr_init(swr_context);
  if (result < 0) {
    err_msg = "Failed to initialize the resampling context.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  avinfo->audio_conversion = swr_context;
  return Status::OK();
}

template <typename T>
Status AVDecodeAudioFrameByConversion(struct AudioVisual *avinfo, std::vector<std::vector<T>> *audio_vector) {
  AVFrame *frame = avinfo->audio.frame;

  if (audio_vector->size() == 0) {
    audio_vector->resize(avinfo->audio.channels, std::vector<T>());
  }

  std::string err_msg;

  struct SwrContext *swr_context = avinfo->audio_conversion;
  AVSampleFormat out_sample_format = AV_SAMPLE_FMT_FLTP;
  if (swr_context == nullptr) {
    RETURN_IF_NOT_OK(AVAllocateAudioConversion(avinfo, out_sample_format));
    swr_context = avinfo->audio_conversion;
  }

  int64_t nb_samples = frame->nb_samples;
  int sample_rate = frame->sample_rate;
  int64_t out_nb_samples = av_rescale_rnd(nb_samples, sample_rate, sample_rate, AV_ROUND_UP);

  uint8_t **out_data = nullptr;
  int linesize[2] = {0, 0};
  int channels = frame->channels;
  int status = av_samples_alloc_array_and_samples(&out_data, linesize, channels, out_nb_samples, out_sample_format, 0);
  if (status < 0) {
    err_msg = "Failed to call av_samples_alloc_array_and_samples at audio frame ";
    err_msg += std::to_string(avinfo->audio.frame_number);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  status = swr_convert(swr_context, out_data, out_nb_samples, (const uint8_t **)(frame->extended_data), nb_samples);
  if (status < 0) {
    err_msg = "Failed to call swr_convert at audio frame " + std::to_string(avinfo->audio.frame_number);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  for (int channel = 0; channel < channels; channel++) {
    T *p_data = reinterpret_cast<T *>(out_data[channel]);
    (*audio_vector)[channel].insert((*audio_vector)[channel].end(), p_data, p_data + nb_samples);
  }

  if (out_data) {
    av_freep(out_data);
  }
  return Status::OK();
}

template <typename T>
Status AVDecodeAudioFrame(struct AudioVisual *avinfo, std::vector<std::vector<T>> *audio_vector) {
  AVFrame *frame = avinfo->audio.frame;
  AVSampleFormat in_sample_format = (AVSampleFormat)(frame->format);

  std::string err_msg;
  if (avinfo->sample_format == AV_SAMPLE_FMT_NONE) {
    avinfo->sample_format = in_sample_format;
  } else {
    if (avinfo->sample_format != in_sample_format) {
      err_msg = "Failed to support the change of sample format at audio frame ";
      err_msg += std::to_string(avinfo->audio.frame_number);
      avinfo->audio.find_error = true;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  if (frame->channels != avinfo->audio.channels) {
    err_msg = "The frame->channels is " + std::to_string(frame->channels) + ", it should be ";
    err_msg += std::to_string(avinfo->audio.channels) + " at audio frame ";
    err_msg += std::to_string(avinfo->audio.frame_number);
    if (avinfo->audio.frame_number > 0) {
      avinfo->audio.find_error = true;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    MS_LOG(WARNING) << err_msg;
    avinfo->audio.channels = frame->channels;
  }

  switch (in_sample_format) {
    case AV_SAMPLE_FMT_DBL:
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_S64:
    case AV_SAMPLE_FMT_U8:
      AVDecodeAudioFramePacket(avinfo, audio_vector);
      break;
    case AV_SAMPLE_FMT_DBLP:
    case AV_SAMPLE_FMT_FLTP:
    case AV_SAMPLE_FMT_S16P:
    case AV_SAMPLE_FMT_S32P:
    case AV_SAMPLE_FMT_S64P:
    case AV_SAMPLE_FMT_U8P:
      AVDecodeAudioFramePlanar(avinfo, audio_vector);
      break;
    default:
      return AVDecodeAudioFrameByConversion(avinfo, audio_vector);
      break;
  }
  return Status::OK();
}

template <typename T>
Status AVAlignAudioFrames(std::vector<std::vector<T>> *audio_vector, int audio_channels,
                          std::shared_ptr<Tensor> *audio_output) {
  TensorShape shape({audio_channels, (int64_t)(*audio_vector)[0].size()});
  DataType type = DataType::FromCType<T>();
  if (Tensor::CreateEmpty(shape, type, audio_output) != Status::OK()) {
    RETURN_STATUS_UNEXPECTED("Failed to call Tensor::CreateEmpty.");
  }
  for (int channel = 0; channel < audio_channels; channel++) {
    std::shared_ptr<Tensor> single_audio;
    if (Tensor::CreateFromVector((*audio_vector)[channel], &single_audio) != Status::OK()) {
      RETURN_STATUS_UNEXPECTED("Failed to call Tensor::CreateFromVector.");
    }
    (*audio_output)->InsertTensor({channel}, single_audio);
  }
  return Status::OK();
}

void GetVisualCurrentPts(struct AudioVisual *avinfo) {
  AVFrame *frame = avinfo->visual.frame;
  int64_t frame_number = avinfo->visual.frame_number;
  int timestamp_increment = avinfo->visual.timestamp_increment;

  // frame_number = 0 is the first frame
  // frame_number = 1 is the second frame
  // check whether the pts_number should be adjusted
  // for example
  // the frame_number       :  0, 1, 2, 3, 4, ...
  //  the frame->pkt_dts    :  2, 3, 4, 5, 6, ...
  // should be adjusted to  :  1, 2, 3, 4, 5, ... when frame->pkt_dts == frame->coded_picture_number
  //                        :  1, 3, 4, 5, 6, ... when frame->pkt_dts != frame->coded_picture_number
  // the adjust method is let it - 1 from the second frame
  // we check the condition on the second frame, the first frame will be adjusted directly
  // The condition can be described as:
  // frame_number == 1 : this is the second frame
  // timestamp_increment == 1 : the timestamp is incremented by 1
  // frame->pts == AV_NOPTS_VALUE : there is no valid pts value
  // frame->pkt_dts == frame->coded_picture_number : the frame->pkt_dts is same as frame->coded_picture_number
  // frame->pkt_dts -1 == frame_number + 1 : when the frame->pkt_dts is 3 for the second frame

  int *adjust_pts_flag = &(avinfo->visual.adjust_pts_flag);
  if (frame_number == 1 && timestamp_increment == 1 && frame->pts == AV_NOPTS_VALUE &&
      frame->pkt_dts == frame->coded_picture_number && frame->pkt_dts - 1 == frame_number + 1) {
    *adjust_pts_flag = 1;
  }

  int64_t *current_pts = &(avinfo->visual.current_pts);
  *current_pts = frame->best_effort_timestamp;
  if (*current_pts == AV_NOPTS_VALUE) {
    *current_pts = frame_number * timestamp_increment;
  } else {
    if (timestamp_increment == 1) {
      if (frame_number == 0) {
        if (*current_pts > 1) {
          *current_pts = 1;
        }
      } else {
        if (*adjust_pts_flag) {
          *current_pts -= 1;
        }
      }
    }
  }
}

Status AVDecodePacketVisual(struct AudioVisual *avinfo, AVPacket *packet) {
  int status = 0;
  AVCodecContext *codec_context = avinfo->visual.codec_context;
  AVFrame *frame = avinfo->visual.frame;

  std::map<int64_t, std::shared_ptr<Tensor>> *output = &(avinfo->map_visual_tensor);
  int64_t *frame_number = &(avinfo->visual.frame_number);
  int64_t *current_pts = &(avinfo->visual.current_pts);
  int64_t start_pts = avinfo->visual.start_pts;
  int64_t end_pts = avinfo->visual.end_pts;
  int64_t *start_frame_number = &(avinfo->visual.start_frame_number);
  int64_t *end_frame_number = &(avinfo->visual.end_frame_number);

  std::string err_msg;

  status = avcodec_send_packet(codec_context, packet);
  if (status < 0) {
    err_msg = "Failed to receive packet for visual frame " + std::to_string(*frame_number);
    if (packet == nullptr) {
      err_msg += " when flushing the codec_context";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  while (status >= 0) {
    std::shared_ptr<Tensor> frame_tensor;
    status = avcodec_receive_frame(codec_context, frame);
    if (status < 0) {
      if (status == AVERROR_EOF || status == AVERROR(EAGAIN)) {
        return Status::OK();
      }
      err_msg = "Failed to receive visual frame " + std::to_string(*frame_number) + ".";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    GetVisualCurrentPts(avinfo);

    // adjust the start_pts when the *current_pts is 1 , *frame_number is 0, and start_pts is 1
    if (*current_pts == 1 && *frame_number == 0 && start_pts == 1) {
      start_pts = 0;
      avinfo->visual.start_pts = 0;
    }

    if (*current_pts >= start_pts && *current_pts <= end_pts) {
      RETURN_IF_NOT_OK(AVDecodeVisualFrame(avinfo, &frame_tensor));
    }

    av_frame_unref(frame);

    int64_t best_frame_number;

    if (frame->display_picture_number > 0) {
      best_frame_number = frame->display_picture_number;
    } else {
      best_frame_number = *frame_number;
    }
    if (*current_pts >= start_pts && *current_pts <= end_pts) {
      if (start_pts > 0) {
        // Record the frame number corresponding to the start_pts
        if (*start_frame_number == 0) {
          *start_frame_number = best_frame_number;
        }
      }
      if (best_frame_number + 1 > *end_frame_number) {
        *end_frame_number = best_frame_number + 1;
      }
      output->insert(std::pair<int64_t, std::shared_ptr<Tensor>>(best_frame_number, frame_tensor));
    }

    if (best_frame_number + 1 > *frame_number) {
      *frame_number = best_frame_number + 1;
    }
  }

  return Status::OK();
}

template <typename T>
Status AVDecodePacketAudio(struct AudioVisual *avinfo, std::vector<std::vector<T>> *audio_vector, AVPacket *packet) {
  AVCodecContext *codec_context = avinfo->audio.codec_context;
  AVFrame *frame = avinfo->audio.frame;

  int64_t *frame_number = &(avinfo->audio.frame_number);

  std::string err_msg;
  int status = avcodec_send_packet(codec_context, packet);
  if (status < 0) {
    err_msg = "Failed to receive packet for audio frame " + std::to_string(*frame_number);
    if (packet == nullptr) {
      err_msg += " when flushing the codec_context";
    }
    char err_buf[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, status);
    err_msg += ". " + std::string(err_buf);
    MS_LOG(WARNING) << err_msg;
    avinfo->audio.find_error = true;
    return Status::OK();
  }
  while (status >= 0) {
    status = avcodec_receive_frame(codec_context, frame);
    if (status < 0) {
      if (status == AVERROR_EOF || status == AVERROR(EAGAIN)) {
        return Status::OK();
      }
      err_msg = "Failed to receive audio frame " + std::to_string(*frame_number) + ".";
      MS_LOG(WARNING) << err_msg;
      avinfo->audio.find_error = true;
      return Status::OK();
    }

    RETURN_IF_NOT_OK(AVDecodeAudioFrame(avinfo, audio_vector));

    *frame_number += 1;
    avinfo->nb_samples += frame->nb_samples;

    av_frame_unref(frame);
  }
  return Status::OK();
}

Status AVOpenAudioStream(struct AudioVisual *avinfo) {
  int stream_index = AVFindAudioStream(avinfo);
  if (stream_index < 0) {
    return Status::OK();
  }

  RETURN_IF_NOT_OK(AVOpenStreamCodecContext(&avinfo->audio, false, false));

  avinfo->sample_format = avinfo->audio.codec_context->sample_fmt;
  avinfo->audio.channels = avinfo->audio.codec_context->channels;
  // Calculate the time_base
  return AVCalculateAudioTimeBase(avinfo);
}

Status AVOpenVisualStream(struct AudioVisual *avinfo) {
  int stream_index = AVFindVisualStream(avinfo);
  if (stream_index < 0) {
    return Status::OK();
  }

  RETURN_IF_NOT_OK(AVOpenStreamCodecContext(&avinfo->visual, true, false));

  std::string err_msg;
  // check the pixel format
  AVStream *stream = avinfo->visual.stream;
  if (stream->codecpar->format == AV_PIX_FMT_NONE) {
    err_msg = "The pixel format is AV_PIX_FMT_NONE.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  avinfo->image_height = stream->codecpar->height;
  avinfo->image_width = stream->codecpar->width;

  RETURN_IF_NOT_OK(AVCalculateVisualFrameRate(avinfo));

  // Calculate the time_base
  RETURN_IF_NOT_OK(AVCalculateVisualTimeBase(avinfo));

  AVRational time_base = stream->time_base;
  AVRational avg_frame_rate = stream->avg_frame_rate;
  avinfo->visual.timestamp_increment = (time_base.den * avg_frame_rate.den) / (time_base.num * avg_frame_rate.num);

  int width = avinfo->image_width;
  int height = avinfo->image_height;

  AVCodecContext *codec_context = avinfo->visual.codec_context;
  AVPixelFormat from_format = codec_context->pix_fmt;
  AVPixelFormat target_format = AV_PIX_FMT_RGB24;

  int channels = kDefaultImageChannel;
  avinfo->visual.channels = channels;

  int align = kMaxImageChannel * kMaxImageChannel;
  if (avinfo->visual_aligned_buffer[0] == nullptr) {
    // adjust the width, height according to the codec_context
    int adjusted_width = width;
    int adjusted_height = height;
    avcodec_align_dimensions(codec_context, &adjusted_width, &adjusted_height);
    // Some codecs need extra 16 bytes, then add 1 extra line.
    adjusted_height += 1;

    av_image_alloc(avinfo->visual_aligned_buffer, avinfo->visual_aligned_linesize, adjusted_width, adjusted_height,
                   target_format, align);
  }
  if (avinfo->visual_aligned_buffer[0] == nullptr) {
    err_msg = "Failed to allocate visual_aligned_buffer.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  if (avinfo->visual_unaligned_buffer == nullptr) {
    avinfo->visual_unaligned_buffer = reinterpret_cast<uint8_t *>(calloc((height + 1) * width * channels, 1));
  }
  if (avinfo->visual_unaligned_buffer == nullptr) {
    err_msg = "Failed to allocate visual_unaligned_buffer.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  SwsContext *conversion = sws_getContext(width, height, from_format, width, height, target_format,
                                          avinfo->visual.conversion_flags, nullptr, nullptr, nullptr);
  if (conversion == nullptr) {
    err_msg = "Failed to get visual convert context.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  avinfo->visual_conversion = conversion;
  return Status::OK();
}

template <typename T>
Status AVGenereateAudioTensor(std::vector<std::vector<T>> *audio_vector, int audio_channels,
                              std::shared_ptr<Tensor> *audio_output) {
  if (audio_vector->size() > 0) {
    return AVAlignAudioFrames(audio_vector, audio_channels, audio_output);
  }

  TensorShape audio_shape = TensorShape({1, 0});
  if (Tensor::CreateEmpty(audio_shape, DataType(DataType::DE_FLOAT32), audio_output) != Status::OK()) {
    RETURN_STATUS_UNEXPECTED("Failed to call Tensor::CreateEmpty.");
  }
  return Status::OK();
}

Status AVGenereateVisualTensor(struct AudioVisual *avinfo, std::shared_ptr<Tensor> *visual_output) {
  int tensor_size = static_cast<int32_t>(avinfo->map_visual_tensor.size());
  int image_height = avinfo->image_height;
  int image_width = avinfo->image_width;
  int64_t start_frame_number = avinfo->visual.start_frame_number;

  TensorShape shape({tensor_size, image_height, image_width, kDefaultImageRank});
  if (Tensor::CreateEmpty(shape, DataType(DataType::DE_UINT8), visual_output) != Status::OK()) {
    RETURN_STATUS_UNEXPECTED("Failed to call Tensor::CreateEmpty.");
  }
  for (auto &visual_frame : avinfo->map_visual_tensor) {
    if (visual_frame.first == 0 && start_frame_number != 0) {
      MS_LOG(WARNING) << "visual_frame.first = " + std::to_string(visual_frame.first);
      MS_LOG(WARNING) << "start_frame_number = " + std::to_string(start_frame_number);

      start_frame_number = 0;
      avinfo->visual.start_frame_number = 0;

      MS_LOG(WARNING) << "The bug is fixed by let start_frame_number = " + std::to_string(start_frame_number);
    }
    (*visual_output)->InsertTensor({visual_frame.first - start_frame_number}, visual_frame.second);
  }
  return Status::OK();
}

template <typename T>
void AudioRemovePoint(struct AudioVisual *avinfo, std::vector<std::vector<T>> *audio_vector) {
  int k;

  int64_t start_pts = avinfo->audio.start_pts;
  int64_t end_pts = avinfo->audio.end_pts;
  int64_t point_number = avinfo->nb_samples;
  int channels = avinfo->audio.channels;

  for (k = 0; k < channels; k++) {
    auto it_begin = (*audio_vector)[k].begin();
    (*audio_vector)[k].erase(it_begin + end_pts, it_begin + point_number);
    (*audio_vector)[k].erase(it_begin, it_begin + start_pts);
  }
}

template <typename T>
void AudioStartEnd(struct AudioVisual *avinfo, std::vector<std::vector<T>> *audio_vector) {
  if (avinfo->audio.stream_index < 0 || avinfo->visual.stream_index < 0) {
    return;
  }

  int64_t start_frame_number = avinfo->visual.start_frame_number;
  int64_t frame_count = avinfo->visual.frame_number;
  int64_t tensor_count = avinfo->map_visual_tensor.size();
  int64_t point_count = avinfo->nb_samples;

  if (point_count < 1 || frame_count < 1) {
    return;
  }

  float points_per_frame = static_cast<float>(point_count) / static_cast<float>(frame_count);
  int64_t start_point = round(start_frame_number * points_per_frame);
  int64_t end_point = start_point + round(tensor_count * points_per_frame);
  if (start_point < 1 && end_point >= point_count) {
    return;
  }

  avinfo->audio.start_pts = start_point;
  avinfo->audio.end_pts = end_point;

  AudioRemovePoint(avinfo, audio_vector);
}

template <typename T>
Status AVReadPackets(struct AudioVisual *avinfo, std::shared_ptr<Tensor> *visual_output,
                     std::shared_ptr<Tensor> *audio_output, std::vector<std::vector<T>> *audio_vector) {
  AVFormatContext *avformat = avinfo->avformat;
  AVPacket *packet = avinfo->packet;

  while (av_read_frame(avformat, packet) >= 0) {
    if (packet->stream_index == avinfo->audio.stream_index && avinfo->audio.find_error == false) {
      RETURN_IF_NOT_OK(AVDecodePacketAudio(avinfo, audio_vector, packet));
    }
    if (packet->stream_index == avinfo->visual.stream_index) {
      RETURN_IF_NOT_OK(AVDecodePacketVisual(avinfo, packet));
    }
    av_packet_unref(packet);
  }
  if (avinfo->audio.stream_index >= 0 && avinfo->audio.find_error == false) {
    // Flush the audio codec with a null packet
    RETURN_IF_NOT_OK(AVDecodePacketAudio(avinfo, audio_vector, nullptr));
  }
  if (avinfo->visual.stream_index >= 0) {
    // Flush the visual codec with a null packet
    RETURN_IF_NOT_OK(AVDecodePacketVisual(avinfo, nullptr));
  }

  RETURN_IF_NOT_OK(AVGenereateVisualTensor(avinfo, visual_output));

  AudioStartEnd(avinfo, audio_vector);

  return AVGenereateAudioTensor(audio_vector, avinfo->audio.channels, audio_output);
}

Status VisualStartEnd(struct AudioVisual *avinfo, float start_pts, float end_pts, const std::string &pts_unit) {
  if (avinfo->visual.stream_index >= 0) {
    if (pts_unit == "pts") {
      avinfo->visual.start_pts = round(start_pts);
      avinfo->visual.end_pts = round(end_pts);
    } else {
      if (avinfo->visual.time_base < 1.0e-9) {
        RETURN_STATUS_UNEXPECTED("The time_base is too small. It is " + std::to_string(avinfo->visual.time_base));
      }
      avinfo->visual.start_pts = round(start_pts / avinfo->visual.time_base);
      avinfo->visual.end_pts = round(end_pts / avinfo->visual.time_base);
    }
  }
  return Status::OK();
}

Status AVGenerateDefaultOutput(std::shared_ptr<Tensor> *video_output, std::shared_ptr<Tensor> *audio_output) {
  TensorShape video_shape({0, 1, 1, kDefaultImageRank});
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(video_shape, DataType(DataType::DE_UINT8), video_output));

  TensorShape audio_shape({1, 0});
  return Tensor::CreateEmpty(audio_shape, DataType(DataType::DE_FLOAT32), audio_output);
}

Status PrepareRead(struct AudioVisual *avinfo, float start_pts, float end_pts, const std::string &pts_unit) {
  // Allocate the packet
  RETURN_IF_NOT_OK(AVPacketAllocate(avinfo));

  // Allocate the frames
  RETURN_IF_NOT_OK(AVFrameAllocate(avinfo, &avinfo->audio));
  RETURN_IF_NOT_OK(AVFrameAllocate(avinfo, &avinfo->visual));

  // Open the video stream and decoder
  RETURN_IF_NOT_OK(AVOpenVisualStream(avinfo));

  // Open the audio stream and decoder
  RETURN_IF_NOT_OK(AVOpenAudioStream(avinfo));

  if (avinfo->visual.stream_index < 0 && avinfo->audio.stream_index < 0) {
    RETURN_STATUS_UNEXPECTED("Neither audio nor visual is found.");
  }

  return VisualStartEnd(avinfo, start_pts, end_pts, pts_unit);
}

Status AVReadVisualAudio(struct AudioVisual *avinfo, std::shared_ptr<Tensor> *visual_output,
                         std::shared_ptr<Tensor> *audio_output) {
  if (avinfo->audio.stream_index < 0) {
    std::vector<std::vector<float>> audio_float_vector;
    return AVReadPackets(avinfo, visual_output, audio_output, &audio_float_vector);
  }
  switch (avinfo->sample_format) {
    case AV_SAMPLE_FMT_DBL:
    case AV_SAMPLE_FMT_DBLP: {
      std::vector<std::vector<double>> audio_double_vector;
      return AVReadPackets(avinfo, visual_output, audio_output, &audio_double_vector);
    }
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP: {
      std::vector<std::vector<float>> audio_float_vector;
      return AVReadPackets(avinfo, visual_output, audio_output, &audio_float_vector);
    }
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S16P: {
      std::vector<std::vector<int16_t>> audio_int16_vector;
      return AVReadPackets(avinfo, visual_output, audio_output, &audio_int16_vector);
    }
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_S32P: {
      std::vector<std::vector<int32_t>> audio_int32_vector;
      return AVReadPackets(avinfo, visual_output, audio_output, &audio_int32_vector);
    }
    case AV_SAMPLE_FMT_S64:
    case AV_SAMPLE_FMT_S64P: {
      std::vector<std::vector<int64_t>> audio_int64_vector;
      return AVReadPackets(avinfo, visual_output, audio_output, &audio_int64_vector);
    }
    case AV_SAMPLE_FMT_U8:
    case AV_SAMPLE_FMT_U8P: {
      std::vector<std::vector<uint8_t>> audio_uint8_vector;
      return AVReadPackets(avinfo, visual_output, audio_output, &audio_uint8_vector);
    }
    default: {
      std::vector<std::vector<float>> audio_float_vector;
      return AVReadPackets(avinfo, visual_output, audio_output, &audio_float_vector);
    }
  }
  return Status::OK();
}

Status DecodeVideo(const TensorRow &input, TensorRow *output) {
  std::shared_ptr<Tensor> visual_output;
  std::shared_ptr<Tensor> audio_output;

  struct AudioVisual avinfo;
  av_log_set_level(avinfo.av_log_level);
  RETURN_IF_NOT_OK(AVOpenMemoryRead(input, &avinfo));

  RETURN_IF_NOT_OK(PrepareRead(&avinfo, 0, INT_MAX, "pts"));

  // Read the packets
  RETURN_IF_NOT_OK(AVReadVisualAudio(&avinfo, &visual_output, &audio_output));

  output->emplace_back(visual_output);
  output->emplace_back(audio_output);
  return Status::OK();
}

Status ReadVideo(const std::string &filename, std::shared_ptr<Tensor> *video_output,
                 std::shared_ptr<Tensor> *audio_output, std::map<std::string, std::string> *metadata_output,
                 float start_pts, float end_pts, const std::string &pts_unit) {
  // Check the output parameters
  RETURN_UNEXPECTED_IF_NULL(video_output);
  RETURN_UNEXPECTED_IF_NULL(audio_output);
  RETURN_UNEXPECTED_IF_NULL(metadata_output);

  std::string err_msg;

  // Check the input parameter: start_pts
  if (start_pts < 0) {
    err_msg = "ReadVideo: Not supported start_pts for " + std::to_string(start_pts) + ". It should be >= 0.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  // Check the input parameter: end_pts
  if (end_pts < start_pts) {
    err_msg = "ReadVideo: Not supported end_pts for " + std::to_string(end_pts) + ".";
    err_msg += " The start_pts = " + std::to_string(start_pts) + ". The end_pts should be >= start_pts.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  // Check the input parameter: pts_unit
  if (pts_unit != "pts" && pts_unit != "sec") {
    RETURN_STATUS_UNEXPECTED("ReadVideo: Not supported pts_unit for " + pts_unit + ".");
  }

  struct AudioVisual avinfo;
  av_log_set_level(avinfo.av_log_level);
  // check the input parameter filename and open it
  RETURN_IF_NOT_OK(AVOpenFile(filename, &avinfo));

  RETURN_IF_NOT_OK(PrepareRead(&avinfo, start_pts, end_pts, pts_unit));
  if (avinfo.visual.stream_index >= 0) {
    (*metadata_output)["video_fps"] = std::to_string(avinfo.visual.frame_rate);
  }
  if (avinfo.audio.stream_index >= 0) {
    (*metadata_output)["audio_fps"] = std::to_string(avinfo.audio.codec_context->sample_rate);
  }

  // Read the packets
  return AVReadVisualAudio(&avinfo, video_output, audio_output);
}

Status ReadVideoTimestamps(const std::string &filename, std::vector<int64_t> *pts_int64_vector, float *video_fps,
                           float *time_base, const std::string &pts_unit) {
  // check the output parameters
  RETURN_UNEXPECTED_IF_NULL(pts_int64_vector);
  RETURN_UNEXPECTED_IF_NULL(video_fps);
  RETURN_UNEXPECTED_IF_NULL(time_base);

  // check the input parameter: pts_unit
  if (pts_unit != "pts" && pts_unit != "sec") {
    std::string err_msg = "ReadVideoTimestamps: Not supported pts_unit for " + pts_unit;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // set default outputs
  // when there is not any video stream, assume the video_fps and time_base are both 1.0
  *video_fps = 1.0;
  *time_base = 1.0;

  struct AudioVisual avinfo;
  av_log_set_level(avinfo.av_log_level);
  // check the input parameter filename and open it
  RETURN_IF_NOT_OK(AVOpenFile(filename, &avinfo));

  int video_stream_index = AVFindVisualStream(&avinfo);
  if (video_stream_index < 0) {
    return Status::OK();
  }

  // Allocate the packet
  RETURN_IF_NOT_OK(AVPacketAllocate(&avinfo));

  // Allocate the frames
  RETURN_IF_NOT_OK(AVFrameAllocate(&avinfo, &avinfo.visual));

  // Open the decoder for the visual
  RETURN_IF_NOT_OK(AVOpenStreamCodecContext(&avinfo.visual, false, true));

  return AVReadVisualPts(&avinfo, pts_int64_vector, video_fps, time_base);
}
}  // namespace dataset
}  // namespace mindspore
