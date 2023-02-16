#include "FFmpegVideoDecoder.h"

extern "C" {
#include <libavutil/opt.h>
}

#include "FFmpegMuxing.h"


#if PLATFORM_WINDOWS
#pragma warning(disable: 4576)
#endif




int FFmpegVideoDecoder::OutputAudioFrame(const AVFrame* Frame)
{
	if (SwrCtx == nullptr)
	{
		SwrCtx = swr_alloc();
		if (!SwrCtx) {
			FFmpegLog("Could not allocate resampler context\n");
			throw;
		}

		 swr_alloc_set_opts(SwrCtx, AUDIO_ENC_CHANNEL_LAYOUT, AUDIO_ENC_FORMAT, AUDIO_SAMPLE_RATE,
		 	AudioDecodeCtx->channel_layout, AudioDecodeCtx->sample_fmt, AudioDecodeCtx->sample_rate, 0, nullptr);


		/* initialize the resampling context */
		if ((ErrorCode = swr_init(SwrCtx)) < 0) {
			FFmpegLog("Failed to initialize the resampling context\n");
			throw;
		}

	}

	const int OutNumSamples = swr_get_out_samples(SwrCtx, Frame->nb_samples);
	const size_t DestSize = OutNumSamples * av_get_bytes_per_sample(AUDIO_ENC_FORMAT) * AUDIO_ENC_CHANNELS;
	DestMediaFrame->Buffer = static_cast<uint8*>(realloc(DestMediaFrame->Buffer, DestSize));
	DestMediaFrame->Size = DestSize;
	uint8_t* OutBuf = DestMediaFrame->Buffer;
	const int Count = swr_convert(SwrCtx, &OutBuf, OutNumSamples, const_cast<const uint8_t**>(Frame->data), Frame->nb_samples);
	if (Count >= 0)
	{
		DestMediaFrame->Time = static_cast<float>(Frame->pts) * av_q2d(AudioStream->time_base);
		DestMediaFrame->Index = AudioFrameNum++;
		DestMediaFrame->Pts = Frame->pts;

	}
	else
	{
		memset(DestMediaFrame->Buffer, 0, DestSize);
		DestMediaFrame->Time = static_cast<float>(Frame->pts) * av_q2d(AudioStream->time_base);
		DestMediaFrame->Index = AudioFrameNum++;
		DestMediaFrame->Pts = Frame->pts;
	}

	bDecodeFinished = true;
	return 0;
}

int FFmpegVideoDecoder::OutputVideoFrame(const AVFrame* Frame)
{
	if (Frame->width != VideoWidth || Frame->height != VideoHeight || Frame->format != PixelFormat) {
		/* To handle this change, one could call av_image_alloc again and
		 * decode the following frames into another raw video file. */
		FFmpegLog("Error: Width, height and pixel format have to be "
				"constant in a raw video file, but the width, height or "
				"pixel format of the input video changed:\n"
				"old: width = %d, height = %d, format = %s\n"
				"new: width = %d, height = %d, format = %s\n",
				VideoWidth, VideoHeight, av_get_pix_fmt_name(PixelFormat),
				Frame->width, Frame->height,
				av_get_pix_fmt_name(static_cast<AVPixelFormat>(Frame->format)));
		return static_cast<int>(EFFmpegResult::FormatNotMatch);
	}
	
	SwsCtx = sws_getCachedContext(
		SwsCtx,
		Frame->width,
		Frame->height,
		static_cast<AVPixelFormat>(Frame->format),
		VideoWidth,
		VideoHeight,
		AV_PIX_FMT_RGB24,
		SWS_FAST_BILINEAR,
		nullptr,
		nullptr,
		nullptr);
	
	uint8_t* Data[AV_NUM_DATA_POINTERS] = {nullptr};
	DestMediaFrame->Index = VideoFrameNum++;
	DestMediaFrame->Width = VideoWidth;
	DestMediaFrame->Height = VideoHeight;
	DestMediaFrame->Buffer = static_cast<uint8*>(malloc(VideoWidth * VideoHeight * 4));
	DestMediaFrame->Size = VideoWidth * VideoHeight * 3;
	Data[0] = (uint8_t *)DestMediaFrame->Buffer;
	int Lines[AV_NUM_DATA_POINTERS] = {0};
	Lines[0] = VideoWidth * 3;

	sws_scale(SwsCtx,
					  (const uint8_t **)Frame->data,
					  Frame->linesize,
					  0,
					  Frame->height,
					  Data,
					  Lines);

	DestMediaFrame->Time = static_cast<float>(Frame->pts) * av_q2d(VideoStream->time_base);
	bDecodeFinished = true;

	return 0;
}

int FFmpegVideoDecoder::DecodePacket(AVCodecContext* DecodeCtx, const AVPacket* Pkt)
{
	// submit the packet to the decoder
	ErrorCode = avcodec_send_packet(DecodeCtx, Pkt);
	if (ErrorCode < 0) {
		FFmpegLog("Error submitting a packet for decoding (%s)\n", AV_ERR2STR(ErrorCode));
		return ErrorCode;
	}

	// get all the available frames from the decoder
	while (ErrorCode >= 0) {
		ErrorCode = avcodec_receive_frame(DecodeCtx, DecodingFrame);
		if (ErrorCode < 0) {
			// those two return values are special and mean there is no output
			// frame available, but there were no errors during decoding
			if (ErrorCode == AVERROR_EOF || ErrorCode == AVERROR(EAGAIN))
				return 0;

			FFmpegLog("Error during decoding (%s)\n", AV_ERR2STR(ErrorCode));
			return ErrorCode;
		}

		// write the frame data to output file
		if (DecodeCtx->codec->type == AVMEDIA_TYPE_VIDEO)
			ErrorCode = OutputVideoFrame(DecodingFrame);
		else if (DecodeCtx->codec->type == AVMEDIA_TYPE_AUDIO)
			ErrorCode = OutputAudioFrame(DecodingFrame);

		av_frame_unref(DecodingFrame);
		if (ErrorCode < 0)
			return ErrorCode;
	}

	return 0;
}

int FFmpegVideoDecoder::OpenCodecContext(int* StreamIndex, AVCodecContext** DecCtxPtr, AVFormatContext* FormatCtx,
	AVMediaType const Type)
{
	const int Ret = av_find_best_stream(FormatCtx, Type, -1, -1, nullptr, 0);
	if (Ret < 0) {
		FFmpegLog("Could not find %s stream in input file '%s'\n",
				av_get_media_type_string(Type), VideoFilePath);
		ErrorCode = Ret;
		return Ret;
	} else {
		const int Stream_Index = Ret;
		const AVStream* Stream = FormatCtx->streams[Stream_Index];

		/* find decoder for the stream */
		const AVCodec* Dec = avcodec_find_decoder(Stream->codecpar->codec_id);
		if (!Dec) {
			FFmpegLog("Failed to find %s codec\n",
					av_get_media_type_string(Type));
			return AVERROR(EINVAL);
		}

		/* Allocate a codec context for the decoder */
		*DecCtxPtr = avcodec_alloc_context3(Dec);
		if (!*DecCtxPtr) {
			FFmpegLog("Failed to allocate the %s codec context\n",
					av_get_media_type_string(Type));
			return AVERROR(ENOMEM);
		}

		/* Copy codec parameters from input stream to output codec context */
		if ((ErrorCode = avcodec_parameters_to_context(*DecCtxPtr, Stream->codecpar)) < 0) {
			FFmpegLog("Failed to copy %s codec parameters to decoder context\n",
					av_get_media_type_string(Type));
			return ErrorCode;
		}

		/* Init the decoders */
		if ((ErrorCode = avcodec_open2(*DecCtxPtr, Dec, nullptr)) < 0) {
			FFmpegLog("Failed to open %s codec\n",
					av_get_media_type_string(Type));
			return ErrorCode;
		}
		*StreamIndex = Stream_Index;
	}

	return 0;
}



FFmpegVideoDecoder::FFmpegVideoDecoder(const std::string FilePath)
{
	VideoFilePath = CopyCString(FilePath.c_str());
	if (avformat_open_input(&DecodeFormatCtx, VideoFilePath, nullptr, nullptr) < 0) {
        FFmpegLog("Could not open source file %s\n", VideoFilePath);
		bInitFailed = true;
        return;
    }

    /* retrieve stream information */
    if (avformat_find_stream_info(DecodeFormatCtx, nullptr) < 0) {
        FFmpegLog("Could not find stream information\n");
    	bInitFailed = true;
		return;
    }

    if (OpenCodecContext(&VideoStreamIndex, &VideoDecodeCtx, DecodeFormatCtx, AVMEDIA_TYPE_VIDEO) >= 0) {
        VideoStream = DecodeFormatCtx->streams[VideoStreamIndex];
    	
        /* allocate image where the decoded image will be put */
        VideoWidth = VideoDecodeCtx->width;
        VideoHeight = VideoDecodeCtx->height;
        PixelFormat = VideoDecodeCtx->pix_fmt;
    }

	if (OpenCodecContext(&AudioStreamIndex, &AudioDecodeCtx, DecodeFormatCtx, AVMEDIA_TYPE_AUDIO) >= 0) {
		AudioStream = DecodeFormatCtx->streams[AudioStreamIndex];
	}
    /* dump input information to stderr */
    av_dump_format(DecodeFormatCtx, 0, VideoFilePath, 0);

    if (!VideoStream) {
        FFmpegLog("Could not find audio or video stream in the input, aborting\n");
        ErrorCode = 1;
        return;
    }

    DecodingFrame = av_frame_alloc();
    if (!DecodingFrame) {
        FFmpegLog("Could not allocate frame\n");
        ErrorCode = AVERROR(ENOMEM);
        return;
    }

    DecodingPacket = av_packet_alloc();
    if (!DecodingPacket) {
        FFmpegLog("Could not allocate packet\n");
        ErrorCode = AVERROR(ENOMEM);
        return;
    }


}

FFmpegVideoDecoder::~FFmpegVideoDecoder()
{
	/* flush the decoders */
	if (VideoDecodeCtx)
		DecodePacket(VideoDecodeCtx, nullptr);

	FFmpegLog("Demuxing succeeded.\n");


	avcodec_free_context(&VideoDecodeCtx);
	avformat_close_input(&DecodeFormatCtx);
	av_packet_free(&DecodingPacket);
	av_frame_free(&DecodingFrame);

	free(VideoFilePath);


}

int FFmpegVideoDecoder::GetWidth() const
{
	return VideoWidth;
}

int FFmpegVideoDecoder::GetHeight() const
{
	return VideoHeight;
}

bool FFmpegVideoDecoder::Seek(const float Time) const
{
	return av_seek_frame(DecodeFormatCtx, -1, Time * AV_TIME_BASE, AVSEEK_FLAG_ANY) >= 0;
}

int FFmpegVideoDecoder::DecodeMedia(FMediaFrame &MediaFrame)
{
	if (bInitFailed)
	{
		return -1;
	}
	int result = 0;
	DestMediaFrame = &MediaFrame;
	MediaFrame.Type = FFrameType::None;
	MediaFrame.Buffer = nullptr;
	MediaFrame.Size = 0;
	MediaFrame.Height = 0;
	MediaFrame.Width = 0;
	MediaFrame.Time = 0;
	MediaFrame.Pts = 0;
	bDecodeFinished = false;
	/* read frames from the file */
	while (av_read_frame(DecodeFormatCtx, DecodingPacket) >= 0) {
		// check if the packet belongs to a stream we are interested in, otherwise
		// skip it
		if (DecodingPacket->stream_index == VideoStreamIndex)
		{
			MediaFrame.Type = FFrameType::Video;
			result = DecodePacket(VideoDecodeCtx, DecodingPacket);
		}
		else if (DecodingPacket->stream_index == AudioStreamIndex)
		{
			MediaFrame.Type = FFrameType::Audio;
			result = DecodePacket(AudioDecodeCtx, DecodingPacket);
		}

		av_packet_unref(DecodingPacket);

		if (bDecodeFinished)
		{
			return 0;
		}
		
		if (result < 0)
		{
			return result;
		}
	}

	return AVERROR_EOF;

}
