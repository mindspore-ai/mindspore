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
#include "debug/debugger/grpc_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/pkcs12.h>
#include <openssl/x509.h>
#include <openssl/evp.h>

#include <thread>
#include <vector>
#include "utils/log_adapter.h"

using debugger::Chunk;
using debugger::EventListener;
using debugger::EventReply;
using debugger::EventReply_Status_FAILED;
using debugger::GraphProto;
using debugger::Metadata;
using debugger::TensorProto;
using debugger::WatchpointHit;

#define CHUNK_SIZE 1024 * 1024 * 3

namespace mindspore {
GrpcClient::GrpcClient(const std::string &host, const std::string &port, const bool &ssl_certificate,
                       const std::string &certificate_dir, const std::string &certificate_passphrase)
    : stub_(nullptr) {
  Init(host, port, ssl_certificate, certificate_dir, certificate_passphrase);
}

void GrpcClient::Init(const std::string &host, const std::string &port, const bool &ssl_certificate,
                      const std::string &certificate_dir, const std::string &certificate_passphrase) {
  std::string target_str = host + ":" + port;
  MS_LOG(INFO) << "GrpcClient connecting to: " << target_str;

  std::shared_ptr<grpc::Channel> channel;
  if (ssl_certificate) {
    FILE *fp;
    EVP_PKEY *pkey = NULL;
    X509 *cert = NULL;
    STACK_OF(X509) *ca = NULL;
    PKCS12 *p12 = NULL;

    if ((fp = fopen(certificate_dir.c_str(), "rb")) == NULL) {
      MS_LOG(ERROR) << "Error opening file: " << certificate_dir;
      exit(EXIT_FAILURE);
    }
    p12 = d2i_PKCS12_fp(fp, NULL);
    fclose(fp);
    if (p12 == NULL) {
      MS_LOG(ERROR) << "Error reading PKCS#12 file";
      X509_free(cert);
      EVP_PKEY_free(pkey);
      sk_X509_pop_free(ca, X509_free);
      exit(EXIT_FAILURE);
    }
    if (!PKCS12_parse(p12, certificate_passphrase.c_str(), &pkey, &cert, &ca)) {
      MS_LOG(ERROR) << "Error parsing PKCS#12 file";
      X509_free(cert);
      EVP_PKEY_free(pkey);
      sk_X509_pop_free(ca, X509_free);
      exit(EXIT_FAILURE);
    }
    std::string strca;
    std::string strcert;
    std::string strkey;

    if (pkey == NULL || cert == NULL || ca == NULL) {
      MS_LOG(ERROR) << "Error private key or cert or CA certificate.";
      X509_free(cert);
      EVP_PKEY_free(pkey);
      sk_X509_pop_free(ca, X509_free);
      exit(EXIT_FAILURE);
    } else {
      ASN1_TIME *validtime = X509_getm_notAfter(cert);
      if (X509_cmp_current_time(validtime) < 0) {
        MS_LOG(ERROR) << "This certificate is over its valid time, please use a new certificate.";
        X509_free(cert);
        EVP_PKEY_free(pkey);
        sk_X509_pop_free(ca, X509_free);
        exit(EXIT_FAILURE);
      }
      int nid = X509_get_signature_nid(cert);
      int keybit = EVP_PKEY_bits(pkey);
      if (nid == NID_sha1) {
        MS_LOG(WARNING) << "Signature algrithm is sha1, which maybe not secure enough.";
      } else if (keybit < 2048) {
        MS_LOG(WARNING) << "The private key bits is: " << keybit << ", which maybe not secure enough.";
      }
      int dwPriKeyLen = i2d_PrivateKey(pkey, NULL);  // get the length of private key
      unsigned char *pribuf = (unsigned char *)malloc(sizeof(unsigned char) * dwPriKeyLen);
      i2d_PrivateKey(pkey, &pribuf);  // PrivateKey DER code
      strkey = std::string(reinterpret_cast<char const *>(pribuf), dwPriKeyLen);

      int dwcertLen = i2d_X509(cert, NULL);  // get the length of private key
      unsigned char *certbuf = (unsigned char *)malloc(sizeof(unsigned char) * dwcertLen);
      i2d_X509(cert, &certbuf);  // PrivateKey DER code
      strcert = std::string(reinterpret_cast<char const *>(certbuf), dwcertLen);

      int dwcaLen = i2d_X509(sk_X509_value(ca, 0), NULL);  // get the length of private key
      unsigned char *cabuf = (unsigned char *)malloc(sizeof(unsigned char) * dwcaLen);
      i2d_X509(sk_X509_value(ca, 0), &cabuf);  // PrivateKey DER code
      strca = std::string(reinterpret_cast<char const *>(cabuf), dwcaLen);

      free(pribuf);
      free(certbuf);
      free(cabuf);
    }

    grpc::SslCredentialsOptions opts = {strca, strkey, strcert};
    channel = grpc::CreateChannel(target_str, grpc::SslCredentials(opts));
  } else {
    channel = grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials());
  }
  stub_ = EventListener::NewStub(channel);
}

void GrpcClient::Reset() { stub_ = nullptr; }

EventReply GrpcClient::WaitForCommand(const Metadata &metadata) {
  EventReply reply;
  grpc::ClientContext context;
  grpc::Status status = stub_->WaitCMD(&context, metadata, &reply);

  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: WaitForCommand";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendMetadata(const Metadata &metadata) {
  EventReply reply;
  grpc::ClientContext context;
  grpc::Status status = stub_->SendMetadata(&context, metadata, &reply);

  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendMetadata";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

std::vector<std::string> ChunkString(std::string str, int graph_size) {
  std::vector<std::string> buf;
  int size_iter = 0;
  while (size_iter < graph_size) {
    int chunk_size = CHUNK_SIZE;
    if (graph_size - size_iter < CHUNK_SIZE) {
      chunk_size = graph_size - size_iter;
    }
    std::string buffer;
    buffer.resize(chunk_size);
    memcpy(reinterpret_cast<char *>(buffer.data()), str.data() + size_iter, chunk_size);
    buf.push_back(buffer);
    size_iter += CHUNK_SIZE;
  }
  return buf;
}

EventReply GrpcClient::SendGraph(const GraphProto &graph) {
  EventReply reply;
  grpc::ClientContext context;
  Chunk chunk;

  std::unique_ptr<grpc::ClientWriter<Chunk> > writer(stub_->SendGraph(&context, &reply));
  std::string str = graph.SerializeAsString();
  int graph_size = graph.ByteSize();
  auto buf = ChunkString(str, graph_size);

  for (unsigned int i = 0; i < buf.size(); i++) {
    MS_LOG(INFO) << "RPC:sending the " << i << "chunk in graph";
    chunk.set_buffer(buf[i]);
    if (!writer->Write(chunk)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();

  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendGraph";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendTensors(const std::list<TensorProto> &tensors) {
  EventReply reply;
  grpc::ClientContext context;

  std::unique_ptr<grpc::ClientWriter<TensorProto> > writer(stub_->SendTensors(&context, &reply));
  for (const auto &tensor : tensors) {
    if (!writer->Write(tensor)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();

  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendTensors";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}

EventReply GrpcClient::SendWatchpointHits(const std::list<WatchpointHit> &watchpoints) {
  EventReply reply;
  grpc::ClientContext context;

  std::unique_ptr<grpc::ClientWriter<WatchpointHit> > writer(stub_->SendWatchpointHits(&context, &reply));
  for (const auto &watchpoint : watchpoints) {
    if (!writer->Write(watchpoint)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  writer->WritesDone();
  grpc::Status status = writer->Finish();

  if (!status.ok()) {
    MS_LOG(ERROR) << "RPC failed: SendWatchpointHits";
    MS_LOG(ERROR) << status.error_code() << ": " << status.error_message();
    reply.set_status(EventReply_Status_FAILED);
  }
  return reply;
}
}  // namespace mindspore
