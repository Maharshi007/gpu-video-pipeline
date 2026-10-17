#include "video_decoder.hpp"

#include <cstring>

VideoDecoder::VideoDecoder(const std::string& path) {
    AVFormatContext* raw = nullptr;
    av_check(avformat_open_input(&raw, path.c_str(), nullptr, nullptr), ("open " + path).c_str());
    fmt_.reset(raw);
    av_check(avformat_find_stream_info(fmt_.get(), nullptr), "find_stream_info");

    const AVCodec* codec = nullptr;
    stream_index_ = av_find_best_stream(fmt_.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    av_check(stream_index_, "no video stream");
    AVStream* st = fmt_->streams[stream_index_];

    ctx_.reset(avcodec_alloc_context3(codec));
    if (!ctx_) throw std::runtime_error("avcodec_alloc_context3 failed");
    av_check(avcodec_parameters_to_context(ctx_.get(), st->codecpar), "parameters_to_context");
    ctx_->thread_count = 0;  // let FFmpeg pick (frame/slice threads)
    av_check(avcodec_open2(ctx_.get(), codec, nullptr), "avcodec_open2 (decoder)");

    fps_ = av_guess_frame_rate(fmt_.get(), st, nullptr);
    if (fps_.num <= 0 || fps_.den <= 0) fps_ = AVRational{25, 1};

    frame_.reset(av_frame_alloc());
    yuv_frame_.reset(av_frame_alloc());
    pkt_.reset(av_packet_alloc());
    if (!frame_ || !yuv_frame_ || !pkt_) throw std::runtime_error("FFmpeg alloc failed");
}

// Standard send/receive decode loop:
//   receive_frame -> EAGAIN means "feed me more packets",
//   at end of file we send a NULL packet to flush frames still in the decoder
//   (B-frame reordering / frame threading keep a few frames buffered).
bool VideoDecoder::read(PlanarYUV420& out) {
    for (;;) {
        int ret = avcodec_receive_frame(ctx_.get(), frame_.get());
        if (ret == 0) {
            convert_to_packed(frame_.get(), out);
            av_frame_unref(frame_.get());
            return true;
        }
        if (ret == AVERROR_EOF) return false;
        if (ret != AVERROR(EAGAIN)) av_check(ret, "receive_frame");

        if (eof_sent_) return false;
        ret = av_read_frame(fmt_.get(), pkt_.get());
        if (ret < 0) {  // end of file (or read error): flush the decoder
            av_check(avcodec_send_packet(ctx_.get(), nullptr), "send_packet(flush)");
            eof_sent_ = true;
            continue;
        }
        if (pkt_->stream_index == stream_index_)
            av_check(avcodec_send_packet(ctx_.get(), pkt_.get()), "send_packet");
        av_packet_unref(pkt_.get());
    }
}

// Copy an AVFrame into our tightly packed buffer. FFmpeg rows are padded for
// SIMD alignment (linesize >= width), so we must copy row by row using
// linesize as the source stride - a single memcpy would skew the image.
void VideoDecoder::convert_to_packed(const AVFrame* f, PlanarYUV420& out) {
    const AVFrame* src = f;
    if (f->format != AV_PIX_FMT_YUV420P) {
        // Any other pixel format (yuv422p, 10-bit, nv12...) -> yuv420p.
        sws_.reset(sws_getCachedContext(sws_.release(), f->width, f->height,
                                        (AVPixelFormat)f->format, f->width, f->height,
                                        AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr,
                                        nullptr));
        if (!sws_) throw std::runtime_error("sws_getCachedContext failed");
        if (yuv_frame_->width != f->width || yuv_frame_->height != f->height) {
            av_frame_unref(yuv_frame_.get());
            yuv_frame_->format = AV_PIX_FMT_YUV420P;
            yuv_frame_->width = f->width;
            yuv_frame_->height = f->height;
            av_check(av_frame_get_buffer(yuv_frame_.get(), 0), "frame_get_buffer");
        }
        sws_scale(sws_.get(), f->data, f->linesize, 0, f->height, yuv_frame_->data,
                  yuv_frame_->linesize);
        src = yuv_frame_.get();
    }

    out.resize(src->width, src->height);
    uint8_t* dst[3] = {out.Y(), out.U(), out.V()};
    int pw[3] = {out.w, out.w / 2, out.w / 2};
    int ph[3] = {out.h, out.h / 2, out.h / 2};
    for (int p = 0; p < 3; ++p)
        for (int y = 0; y < ph[p]; ++y)
            std::memcpy(dst[p] + size_t(y) * pw[p], src->data[p] + size_t(y) * src->linesize[p], pw[p]);
}
