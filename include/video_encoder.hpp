#pragma once
#include <cstdint>
#include <string>

#include "av_raii.hpp"
#include "frame_types.hpp"

struct EncoderConfig {
    std::string codec = "libx264";  // or libx265
    int crf = 23;
    std::string preset = "medium";
};

// Encodes YUV420P frames into an MP4 (or any container chosen by extension).
class VideoEncoder {
public:
    VideoEncoder(const std::string& path, int w, int h, AVRational fps, const EncoderConfig& cfg);
    ~VideoEncoder();

    void write(const PlanarYUV420& frame);
    void finish();  // flush delayed frames + write trailer (called by dtor if needed)

private:
    void drain(bool flushing);

    AVOutputFormatPtr fmt_;
    AVCodecCtxPtr ctx_;
    AVFramePtr frame_;
    AVPacketPtr pkt_;
    AVStream* stream_ = nullptr;
    int64_t next_pts_ = 0;
    bool finished_ = false;
};
