#pragma once
#include <string>

#include "av_raii.hpp"
#include "frame_types.hpp"

// Decodes the best video stream of a file into tightly packed YUV420P frames.
class VideoDecoder {
public:
    explicit VideoDecoder(const std::string& path);

    // Returns false at end of stream (after the decoder has been flushed).
    bool read(PlanarYUV420& out);

    int width() const { return ctx_->width; }
    int height() const { return ctx_->height; }
    AVRational frame_rate() const { return fps_; }

private:
    void convert_to_packed(const AVFrame* f, PlanarYUV420& out);

    AVInputFormatPtr fmt_;
    AVCodecCtxPtr ctx_;
    AVFramePtr frame_;
    AVFramePtr yuv_frame_;  // only used when input is not YUV420P
    AVPacketPtr pkt_;
    SwsCtxPtr sws_;
    int stream_index_ = -1;
    bool eof_sent_ = false;
    AVRational fps_{25, 1};
};
