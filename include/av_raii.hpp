#pragma once
// RAII wrappers for FFmpeg objects: every allocation is freed automatically
// when the owning unique_ptr goes out of scope, even if an exception is thrown.

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <memory>
#include <stdexcept>
#include <string>

struct AVInputFormatCtxDeleter {
    void operator()(AVFormatContext* p) const { avformat_close_input(&p); }
};
struct AVOutputFormatCtxDeleter {
    void operator()(AVFormatContext* p) const {
        if (p && p->oformat && !(p->oformat->flags & AVFMT_NOFILE)) avio_closep(&p->pb);
        avformat_free_context(p);
    }
};
struct AVCodecCtxDeleter {
    void operator()(AVCodecContext* p) const { avcodec_free_context(&p); }
};
struct AVFrameDeleter {
    void operator()(AVFrame* p) const { av_frame_free(&p); }
};
struct AVPacketDeleter {
    void operator()(AVPacket* p) const { av_packet_free(&p); }
};
struct SwsCtxDeleter {
    void operator()(SwsContext* p) const { sws_freeContext(p); }
};

using AVInputFormatPtr  = std::unique_ptr<AVFormatContext, AVInputFormatCtxDeleter>;
using AVOutputFormatPtr = std::unique_ptr<AVFormatContext, AVOutputFormatCtxDeleter>;
using AVCodecCtxPtr     = std::unique_ptr<AVCodecContext, AVCodecCtxDeleter>;
using AVFramePtr        = std::unique_ptr<AVFrame, AVFrameDeleter>;
using AVPacketPtr       = std::unique_ptr<AVPacket, AVPacketDeleter>;
using SwsCtxPtr         = std::unique_ptr<SwsContext, SwsCtxDeleter>;

// av_err2str() is a C macro using a compound literal, which C++ does not
// allow, so we wrap av_strerror() instead.
inline std::string av_error_string(int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, buf, sizeof(buf));
    return buf;
}

// Throw with a readable message if an FFmpeg call returned a negative code.
inline void av_check(int ret, const char* what) {
    if (ret < 0) throw std::runtime_error(std::string(what) + ": " + av_error_string(ret));
}
