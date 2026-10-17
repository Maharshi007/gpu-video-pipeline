#include "video_encoder.hpp"

#include <cstring>
#include <iostream>

VideoEncoder::VideoEncoder(const std::string& path, int w, int h, AVRational fps,
                           const EncoderConfig& cfg) {
    AVFormatContext* raw = nullptr;
    av_check(avformat_alloc_output_context2(&raw, nullptr, nullptr, path.c_str()), "alloc_output");
    fmt_.reset(raw);

    const AVCodec* codec = avcodec_find_encoder_by_name(cfg.codec.c_str());
    if (!codec) throw std::runtime_error("encoder not found: " + cfg.codec);

    stream_ = avformat_new_stream(fmt_.get(), nullptr);
    if (!stream_) throw std::runtime_error("avformat_new_stream failed");

    ctx_.reset(avcodec_alloc_context3(codec));
    ctx_->width = w;
    ctx_->height = h;
    ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx_->time_base = av_inv_q(fps);  // 1 tick = 1 frame, so pts = frame index
    ctx_->framerate = fps;
    // MP4 stores SPS/PPS once in the container header, not in every keyframe.
    if (fmt_->oformat->flags & AVFMT_GLOBALHEADER) ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // x264/x265 private options: CRF = constant quality (bitrate varies with
    // content), preset = speed vs compression-efficiency trade-off.
    av_opt_set(ctx_->priv_data, "preset", cfg.preset.c_str(), 0);
    av_opt_set(ctx_->priv_data, "crf", std::to_string(cfg.crf).c_str(), 0);
    if (cfg.codec == "libx265") av_opt_set(ctx_->priv_data, "x265-params", "log-level=error", 0);

    av_check(avcodec_open2(ctx_.get(), codec, nullptr), "avcodec_open2 (encoder)");
    av_check(avcodec_parameters_from_context(stream_->codecpar, ctx_.get()), "parameters_from_context");
    stream_->time_base = ctx_->time_base;

    if (!(fmt_->oformat->flags & AVFMT_NOFILE))
        av_check(avio_open(&fmt_->pb, path.c_str(), AVIO_FLAG_WRITE), ("open " + path).c_str());
    av_check(avformat_write_header(fmt_.get(), nullptr), "write_header");

    frame_.reset(av_frame_alloc());
    frame_->format = AV_PIX_FMT_YUV420P;
    frame_->width = w;
    frame_->height = h;
    av_check(av_frame_get_buffer(frame_.get(), 0), "frame_get_buffer");
    pkt_.reset(av_packet_alloc());
}

VideoEncoder::~VideoEncoder() {
    try {
        finish();
    } catch (const std::exception& e) {
        std::cerr << "encoder finish failed: " << e.what() << "\n";
    }
}

void VideoEncoder::write(const PlanarYUV420& in) {
    // The encoder may still hold a reference to the previous frame's buffer.
    av_check(av_frame_make_writable(frame_.get()), "frame_make_writable");
    const uint8_t* src[3] = {in.Y(), in.U(), in.V()};
    int pw[3] = {in.w, in.w / 2, in.w / 2};
    int ph[3] = {in.h, in.h / 2, in.h / 2};
    for (int p = 0; p < 3; ++p)  // packed -> padded rows (linesize)
        for (int y = 0; y < ph[p]; ++y)
            std::memcpy(frame_->data[p] + size_t(y) * frame_->linesize[p], src[p] + size_t(y) * pw[p], pw[p]);
    frame_->pts = next_pts_++;
    av_check(avcodec_send_frame(ctx_.get(), frame_.get()), "send_frame");
    drain(false);
}

// Pull every packet the encoder has ready and mux it into the file.
void VideoEncoder::drain(bool flushing) {
    for (;;) {
        int ret = avcodec_receive_packet(ctx_.get(), pkt_.get());
        if (ret == AVERROR(EAGAIN) && !flushing) return;
        if (ret == AVERROR_EOF) return;
        av_check(ret, "receive_packet");
        av_packet_rescale_ts(pkt_.get(), ctx_->time_base, stream_->time_base);
        pkt_->stream_index = stream_->index;
        av_check(av_interleaved_write_frame(fmt_.get(), pkt_.get()), "write_frame");
    }
}

void VideoEncoder::finish() {
    if (finished_) return;
    finished_ = true;
    // NULL frame = "no more input": x264/x265 then emit their look-ahead frames.
    av_check(avcodec_send_frame(ctx_.get(), nullptr), "send_frame(flush)");
    drain(true);
    av_check(av_write_trailer(fmt_.get()), "write_trailer");
}
