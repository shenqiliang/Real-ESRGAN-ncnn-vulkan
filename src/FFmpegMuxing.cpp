#include "FFmpegMuxing.h"





void LogPacket(const AVFormatContext* FormatCtx, const AVPacket* Pkt)
{
 //    AVRational* TimeBase = &FormatCtx->streams[Pkt->stream_index]->time_base;
	// AV_TS2STR(Pkt->pts);
 //
	//
 //    FFmpegLog("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
 //           AV_TS2STR(Pkt->pts), AV_TS2TIME_STR(Pkt->pts, TimeBase),
 //           AV_TS2STR(Pkt->dts), AV_TS2TIME_STR(Pkt->dts, TimeBase),
 //           AV_TS2STR(Pkt->duration), AV_TS2TIME_STR(Pkt->duration, TimeBase),
 //           Pkt->stream_index);
}

int WriteFrame(AVFormatContext* FormatCtx, AVCodecContext* Codec, const AVStream* Stream, const AVFrame* Frame)
{
	// send the frame to the encoder
	int Ret = avcodec_send_frame(Codec, Frame);
	if (Ret < 0) {
		FFmpegLog("Error sending a frame to the encoder: %s\n",
				AV_ERR2STR(Ret));
		throw;
	}

	while (Ret >= 0) {
		AVPacket Pkt = { nullptr };

		Ret = avcodec_receive_packet(Codec, &Pkt);
		if (Ret == AVERROR(EAGAIN) || Ret == AVERROR_EOF)
		{
			break;
		}
		if (Ret < 0) {
			FFmpegLog("Error encoding a frame: %s\n", AV_ERR2STR(Ret));
			throw;
		}

		/* rescale output packet timestamp values from codec to stream timebase */
		av_packet_rescale_ts(&Pkt, Codec->time_base, Stream->time_base);
		Pkt.stream_index = Stream->index;

		/* Write the compressed frame to the media file. */
		LogPacket(FormatCtx, &Pkt);
		Ret = av_interleaved_write_frame(FormatCtx, &Pkt);
		av_packet_unref(&Pkt);
		if (Ret < 0) {
			FFmpegLog("Error while writing output packet: %s\n", AV_ERR2STR(Ret));
			throw;
		}
	}

	return Ret == AVERROR_EOF ? 1 : 0;
}

int FlushFrames(AVFormatContext* FormatCtx, AVCodecContext* Codec, const AVStream* Stream, AVPacket* Pkt)
{

	FFmpegLog("write last frame");

	// send the frame to the encoder
	int Ret = avcodec_send_frame(Codec, nullptr);
	if (Ret < 0) {
		FFmpegLog("Error sending a frame to the encoder: %s\n",
				AV_ERR2STR(Ret));
		throw;
	}

	while (Ret >= 0) {
		Ret = avcodec_receive_packet(Codec, Pkt);
		if (Ret == AVERROR_EOF)
		{
			break;
		}
		if (Ret < 0) {
			FFmpegLog("Error encoding a frame: %s\n", AV_ERR2STR(Ret));
			throw;
		}

		/* rescale output packet timestamp values from codec to stream timebase */
		av_packet_rescale_ts(Pkt, Codec->time_base, Stream->time_base);
		Pkt->stream_index = Stream->index;

		/* Write the compressed frame to the media file. */
		LogPacket(FormatCtx, Pkt);
		Ret = av_interleaved_write_frame(FormatCtx, Pkt);
		/* pkt is now blank (av_interleaved_write_frame() takes ownership of
		 * its contents and resets pkt), so that no unreferencing is necessary.
		 * This would be different if one used av_write_frame(). */
		if (Ret < 0) {
			FFmpegLog("Error while writing output packet: %s\n", AV_ERR2STR(Ret));
			throw;
		}
	}

	return Ret == AVERROR_EOF ? 1 : 0;
}

/* Add an output stream. */
void AddStream(FOutputStream* Ost, AVFormatContext* FormatCtx, AVCodec **Codec, const AVCodecID CodecId, const int Width, const int Height, const int UseHardware, const int Bitrate)
{
	int i;
	*Codec = avcodec_find_encoder(CodecId);
	
	if ((*Codec)->type == AVMEDIA_TYPE_VIDEO)
	{
		/* find the encoder */
		AVCodec* Hevc_Cuvid = avcodec_find_encoder_by_name("hevc_cuvid");
		AVCodec* H264_Cuvid = avcodec_find_encoder_by_name("h264_cuvid");
		AVCodec* Hevc_Nvenc = avcodec_find_encoder_by_name("hevc_nvenc");
		AVCodec* H264_Nvenc = avcodec_find_encoder_by_name("h264_nvenc");
		if (UseHardware && Hevc_Nvenc != nullptr && CodecId == AV_CODEC_ID_HEVC && avcodec_get_hw_config(Hevc_Nvenc, 0) != nullptr)
		{
			FFmpegLog("[Encoder] use hevc_nvenc.");
			*Codec = Hevc_Nvenc;
		}
		else if (UseHardware && Hevc_Cuvid != nullptr && CodecId == AV_CODEC_ID_HEVC && avcodec_get_hw_config(Hevc_Cuvid, 0) != nullptr)
		{
			FFmpegLog("[Encoder] use hevc_cuvid.");
			*Codec = Hevc_Cuvid;
		}
		else if (UseHardware && H264_Nvenc != nullptr && CodecId == AV_CODEC_ID_H264 && avcodec_get_hw_config(H264_Nvenc, 0) != nullptr)
		{
			FFmpegLog("[Encoder] use h264_nvenc.");
			*Codec = H264_Nvenc;
		}
		else if (UseHardware && H264_Cuvid != nullptr && CodecId == AV_CODEC_ID_H264 && avcodec_get_hw_config(H264_Cuvid, 0) != nullptr)
		{
			FFmpegLog("[Encoder] use h264_cuvid.");
			*Codec = H264_Cuvid;
		}
		else
		{
			FFmpegLog("[Encoder] fallback to use h264 soft decoder.");
		}
	
	}

    if (!(*Codec)) {
        FFmpegLog("Could not find encoder for '%s'\n",
                avcodec_get_name(CodecId));
        throw;
    }
	
    Ost->TmpPkt = av_packet_alloc();
    if (!Ost->TmpPkt) {
        FFmpegLog("Could not allocate AVPacket\n");
        throw;
    }

    Ost->Stream = avformat_new_stream(FormatCtx, nullptr);
    if (!Ost->Stream) {
        FFmpegLog("Could not allocate stream\n");
        throw;
    }
    Ost->Stream->id = FormatCtx->nb_streams-1;

    AVCodecContext* c = avcodec_alloc_context3(*Codec);
    if (!c) {
        FFmpegLog("Could not alloc an encoding context\n");
        throw;
    }

    Ost->EncodeCtx = c;
	if (((*Codec)->capabilities & AV_CODEC_CAP_SLICE_THREADS) != 0)
	{
		c->thread_count = 16;
		c->thread_type |= FF_THREAD_SLICE;
	}


    switch ((*Codec)->type) {
        case AVMEDIA_TYPE_AUDIO:
        	c->sample_fmt  = (*Codec)->sample_fmts ?
        		(*Codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
    		c->bit_rate    = 64000;
    		c->sample_rate = 44100;
    		if ((*Codec)->supported_samplerates) {
    			c->sample_rate = (*Codec)->supported_samplerates[0];
    			for (i = 0; (*Codec)->supported_samplerates[i]; i++) {
    				if ((*Codec)->supported_samplerates[i] == 44100)
    					c->sample_rate = 44100;
    			}
    		}
    		c->channels        = av_get_channel_layout_nb_channels(c->channel_layout);
    		c->channel_layout = AUDIO_ENC_CHANNEL_LAYOUT;
    		if ((*Codec)->channel_layouts) {
    			c->channel_layout = (*Codec)->channel_layouts[0];
    			for (i = 0; (*Codec)->channel_layouts[i]; i++) {
    				if ((*Codec)->channel_layouts[i] == AUDIO_ENC_CHANNEL_LAYOUT)
    					c->channel_layout = AUDIO_ENC_CHANNEL_LAYOUT;
    			}
    		}
    		c->channels        = av_get_channel_layout_nb_channels(c->channel_layout);
    		Ost->Stream->time_base = AVRational { 1, c->sample_rate };
    		break;
    	case AVMEDIA_TYPE_VIDEO:
	        c->codec_id = CodecId;
	        if (Bitrate > 0)
	        {
	            c->bit_rate = static_cast<int64_t>(Bitrate) * 1000000;
	        }
	        /* Resolution must be a multiple of two. */
	        c->width    = Width;
	        c->height   = Height;
	        /* timebase: This is the fundamental unit of time (in seconds) in terms
	         * of which frame timestamps are represented. For fixed-fps content,
	         * timebase should be 1/framerate and timestamp increments should be
	         * identical to 1. */
	        Ost->Stream->time_base = AVRational{ 1, 1000 };
	        c->time_base       = Ost->Stream->time_base;
	        c->framerate = AVRational{ STREAM_FRAME_RATE,  1};

	        c->gop_size      = 12; /* emit one intra frame every twelve frames at most */
	        c->pix_fmt       = STREAM_PIX_FMT;
	        if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
	            /* just for testing, we also add B-frames */
	            c->max_b_frames = 2;
	        }
	        if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
	            /* Needed to avoid using macroblocks in which some coeffs overflow.
	             * This does not happen with normal video, it just happens here as
	             * the motion of the chroma plane does not match the luma plane. */
	            c->mb_decision = 2;
	        }
			break;

	    default:
	        break;
    }

    /* Some formats want stream headers to be separate. */
    if (FormatCtx->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

	c->flags |= AV_CODEC_FLAG_LOW_DELAY;

}


/**************************************************************/
/* video output */

AVFrame* AllocPicture(const AVPixelFormat PixelFormat, const int Width, const int Height)
{
	AVFrame* Picture = av_frame_alloc();
    if (!Picture)
        return nullptr;

    Picture->format = PixelFormat;
    Picture->width  = Width;
    Picture->height = Height;

    /* allocate the buffers for the frame data */
	const int Ret = av_frame_get_buffer(Picture, 0);
    if (Ret < 0) {
        FFmpegLog("Could not allocate frame data.\n");
        throw;
    }

    return Picture;
}

void OpenVideo(const AVCodec* Codec, FOutputStream* Ost, const AVDictionary* Option)
{
	AVCodecContext* CodecContext = Ost->EncodeCtx;
    AVDictionary* Opt = nullptr;

    av_dict_copy(&Opt, Option, 0);

    /* open the codec */
    int Ret = avcodec_open2(CodecContext, Codec, &Opt);
    av_dict_free(&Opt);
    if (Ret < 0) {
        FFmpegLog("Could not open video codec: %s\n", AV_ERR2STR(Ret));
        throw;
    }

    /* allocate and init a re-usable frame */
    Ost->Frame = AllocPicture(CodecContext->pix_fmt, CodecContext->width, CodecContext->height);
    if (!Ost->Frame) {
        FFmpegLog("Could not allocate video frame\n");
        throw;
    }

    /* If the output format is not YUV420P, then a temporary YUV420P
     * picture is needed too. It is then converted to the required
     * output format. */
    Ost->TmpFrame = nullptr;
    if (CodecContext->pix_fmt != AV_PIX_FMT_YUV420P) {
        Ost->TmpFrame = AllocPicture(AV_PIX_FMT_YUV420P, CodecContext->width, CodecContext->height);
        if (!Ost->TmpFrame) {
            FFmpegLog("Could not allocate temporary picture\n");
            throw;
        }
    }

    /* copy the stream parameters to the muxer */
    Ret = avcodec_parameters_from_context(Ost->Stream->codecpar, CodecContext);
    if (Ret < 0) {
        FFmpegLog("Could not copy the stream parameters\n");
        throw;
    }
}

AVFrame* AllocAudioFrame(
	const AVSampleFormat SampleFormat,
	const uint64_t ChannelLayout,
	const int SampleRate,
	const int NumSamples
	)
{
	AVFrame* Frame = av_frame_alloc();

	if (!Frame) {
		FFmpegLog("Error allocating an audio frame\n");
		throw;
	}

	Frame->format = SampleFormat;
	Frame->channel_layout = ChannelLayout;
	Frame->sample_rate = SampleRate;
	Frame->nb_samples = NumSamples;

	if (NumSamples) {
		const int Ret = av_frame_get_buffer(Frame, 0);
		if (Ret < 0) {
			FFmpegLog("Error allocating an audio buffer\n");
			throw;
		}
	}

	return Frame;
}

void OpenAudio(const AVCodec* Codec, FOutputStream* Ost, const AVDictionary* OptArg)
{
	int NumSamples;
	AVDictionary* Opt = nullptr;

    AVCodecContext* CodecContext = Ost->EncodeCtx;

    /* open it */
    av_dict_copy(&Opt, OptArg, 0);
    int Ret = avcodec_open2(CodecContext, Codec, &Opt);
    av_dict_free(&Opt);
    if (Ret < 0) {
        FFmpegLog("Could not open audio codec: %s\n", AV_ERR2STR(Ret));
        throw;
    }
	
    if (CodecContext->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
        NumSamples = 10000;
    else
        NumSamples = CodecContext->frame_size;

	if (CodecContext->channels != AUDIO_ENC_CHANNELS)
	{
		FFmpegLog("Not supported AUDIO_ENC_CHANNELS\n");
		throw;
	}
	if (CodecContext->channel_layout != AUDIO_ENC_CHANNEL_LAYOUT)
	{
		FFmpegLog("Not supported AUDIO_ENC_CHANNEL_LAYOUT\n");
		throw;
	}


    Ost->Frame     = AllocAudioFrame(CodecContext->sample_fmt, CodecContext->channel_layout,
                                       CodecContext->sample_rate, NumSamples);
    Ost->TmpFrame = AllocAudioFrame(AV_SAMPLE_FMT_S16, CodecContext->channel_layout,
                                       CodecContext->sample_rate, NumSamples);

    /* copy the stream parameters to the muxer */
    Ret = avcodec_parameters_from_context(Ost->Stream->codecpar, CodecContext);
    if (Ret < 0) {
        FFmpegLog("Could not copy the stream parameters\n");
        throw;
    }

    /* create resampler context */
    Ost->SwrCtx = swr_alloc();
    if (!Ost->SwrCtx) {
        FFmpegLog("Could not allocate resampler context\n");
        throw;
    }

    /* set options */
    av_opt_set_int       (Ost->SwrCtx, "in_channel_count",   CodecContext->channels,       0);
    av_opt_set_int       (Ost->SwrCtx, "in_sample_rate",     CodecContext->sample_rate,    0);
    av_opt_set_sample_fmt(Ost->SwrCtx, "in_sample_fmt",      AV_SAMPLE_FMT_S16, 0);
    av_opt_set_int       (Ost->SwrCtx, "out_channel_count",  CodecContext->channels,       0);
    av_opt_set_int       (Ost->SwrCtx, "out_sample_rate",    CodecContext->sample_rate,    0);
    av_opt_set_sample_fmt(Ost->SwrCtx, "out_sample_fmt",     CodecContext->sample_fmt,     0);

    /* initialize the resampling context */
    if ((Ret = swr_init(Ost->SwrCtx)) < 0) {
        FFmpegLog("Failed to initialize the resampling context\n");
        throw;
    }
}

AVFrame* GetAudioFrame(FOutputStream* Ost, const void* Buffer, size_t Size)
{
	AVFrame* Frame = Ost->TmpFrame;
	void* q = Frame->data[0];
	memcpy(q, Buffer, Size);

	Frame->pts = Ost->NextPts;
	Ost->NextPts  += Frame->nb_samples;

	return Frame;
}

int WriteAudioFrame(AVFormatContext* FormatCtx, FOutputStream* Ost, const void* Buffer, size_t Size)
{
	AVCodecContext* CodecContext = Ost->EncodeCtx;
	
	AVFrame* Frame = GetAudioFrame(Ost, Buffer, Size);

	if (Frame) {
		/* convert samples from native format to destination codec format, using the resampler */
		/* compute destination number of samples */
		const int DestNumSamples = av_rescale_rnd(swr_get_delay(Ost->SwrCtx, CodecContext->sample_rate) + Frame->nb_samples,
			CodecContext->sample_rate, CodecContext->sample_rate, AV_ROUND_UP);
		av_assert0(DestNumSamples == Frame->nb_samples);

		/* when we pass a frame to the encoder, it may keep a reference to it
		 * internally;
		 * make sure we do not overwrite it here
		 */
		int Ret = av_frame_make_writable(Ost->Frame);
		if (Ret < 0)
			throw;

		/* convert to destination format */
		Ret = swr_convert(Ost->SwrCtx,
						  Ost->Frame->data, DestNumSamples,
						  const_cast<const uint8_t**>(Frame->data), Frame->nb_samples);
		if (Ret < 0) {
			FFmpegLog("Error while converting\n");
			throw;
		}
		Frame = Ost->Frame;

		Frame->pts = av_rescale_q(Ost->SampleNum, AVRational{1, CodecContext->sample_rate}, CodecContext->time_base);
		Ost->SampleNum += DestNumSamples;
		return WriteFrame(FormatCtx, CodecContext, Ost->Stream, Frame);
	}
	else
	{
		return AVERROR(EAGAIN);
	}

}

AVFrame* GetVideoFrame(FOutputStream* Ost, const char* Buffer, const int Width, const int Height, double Time)
{
	const AVCodecContext* CodecContext = Ost->EncodeCtx;

    // /* check if we want to generate more frames */
    // if (av_compare_ts(ost->next_pts, c->time_base,
    //                   STREAM_DURATION, AVRational{ 1, 1 }) > 0)
    //     return NULL;

    /* when we pass a frame to the encoder, it may keep a reference to it
     * internally; make sure we do not overwrite it here */
    if (av_frame_make_writable(Ost->Frame) < 0)
        throw;

	if (!Ost->SwsCtx) {
		Ost->SwsCtx = sws_getContext(Width, Height,
									  AV_PIX_FMT_RGB24,
									  CodecContext->width, CodecContext->height,
									  CodecContext->pix_fmt,
									  SCALE_FLAGS, nullptr, nullptr, nullptr);
		if (!Ost->SwsCtx) {
			fprintf(stderr,
					"Could not initialize the conversion context\n");
			throw;
		}
	}
	const int LineSize[2] = {3 * (Width), 0};
	sws_scale(Ost->SwsCtx, reinterpret_cast<const uint8_t*const*>(&Buffer),
			  LineSize, 0, Height, Ost->Frame->data,
			  Ost->Frame->linesize);
	if (Time >= 0.0)
	{
		Ost->Frame->pts = static_cast<int64_t>(Time/av_q2d(CodecContext->time_base));
		Ost->NextPts++;
	}
	else
	{
		Ost->Frame->pts = Ost->NextPts++;
	}
	return Ost->Frame;

}

/*
 * encode one video frame and send it to the muxer
 * return 1 when encoding is finished, 0 otherwise
 */
int WriteVideoFrame(AVFormatContext* FormatCtx, FOutputStream* Ost, const void* Buffer, const int Width, const int Height, double Time)
{
    return WriteFrame(FormatCtx, Ost->EncodeCtx, Ost->Stream, GetVideoFrame(Ost, static_cast<const char*>(Buffer), Width, Height, Time));
}

int FlushOutputStream(AVFormatContext* FormatCtx, FOutputStream* Ost)
{
	return FlushFrames(FormatCtx, Ost->EncodeCtx, Ost->Stream, Ost->TmpPkt);
}


void CloseStream(FOutputStream* Ost)
{
    avcodec_free_context(&Ost->EncodeCtx);
    av_frame_free(&Ost->Frame);
    av_frame_free(&Ost->TmpFrame);
    av_packet_free(&Ost->TmpPkt);
    sws_freeContext(Ost->SwsCtx);
    swr_free(&Ost->SwrCtx);
}
