#include "FFmpegVideoEncoder.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>


FFmpegVideoEncoder::FFmpegVideoEncoder(int Width, int Height, std::string FilePath)
{
	AVDictionary* Opt = nullptr;

	const char* FileName = FilePath.c_str();

	/* allocate the output media context */
	avformat_alloc_output_context2(&FormatContext, nullptr, nullptr, FileName);
	if (!FormatContext) {
		printf("Could not deduce output format from file extension: using MPEG.\n");
		avformat_alloc_output_context2(&FormatContext, nullptr, "mpeg", FileName);
	}
	if (!FormatContext)
		throw;

	OutputFormat = FormatContext->oformat;

	/* Add the audio and video streams using the default format codecs
	 * and initialize the codecs. */
	if (OutputFormat->video_codec != AV_CODEC_ID_NONE) {
		int Bitrate = 20;
		AddStream(&VideoStream, FormatContext, &VideoCodec, OutputFormat->video_codec, Width, Height, false, Bitrate);
		bHasVideo = true;
		EncodeVideo = 1;
	}
	if (OutputFormat->audio_codec != AV_CODEC_ID_NONE) {
		AddStream(&AudioStream, FormatContext, &AudioCodec, OutputFormat->audio_codec, 0, 0, 0, 0);
		bHasAudio = true;
		EncodeAudio = 1;
	}


	av_dict_set(&Opt, "tune", "zerolatency", 0);
	
	/* Now that all the parameters are set, we can open the audio and
	 * video codecs and allocate the necessary encode buffers. */
	if (bHasVideo)
		OpenVideo(VideoCodec, &VideoStream, Opt);
	if (bHasAudio)
		OpenAudio(AudioCodec, &AudioStream, Opt);


	av_dump_format(FormatContext, 0, FileName, 1);

	/* open the output file, if needed */
	if (!(OutputFormat->flags & AVFMT_NOFILE)) {
		Ret = avio_open(&FormatContext->pb, FileName, AVIO_FLAG_WRITE);
		if (Ret < 0) {
			FFmpegLog("Could not open '%s': %s\n", FileName,
					AV_ERR2STR(Ret));
			throw;
		}
	}

	/* Write the stream header, if any. */
	Ret = avformat_write_header(FormatContext, &Opt);
	if (Ret < 0) {
		FFmpegLog("Error occurred when opening output file: %s\n",
		 		AV_ERR2STR(Ret));
		throw;
	}

	if (AudioEncodeBuffer == nullptr)
	{
		TotalAudioBufferSize = AudioStream.TmpFrame->nb_samples * av_get_bytes_per_sample(AUDIO_ENC_FORMAT) * AUDIO_ENC_CHANNELS;
		AudioEncodeBuffer = static_cast<char*>(malloc(TotalAudioBufferSize));
	}

}


void FFmpegVideoEncoder::AddVideoBuffer(const char* Buffer, int Width, int Height, double Time)
{

	FFmpegLog("write video time: %f", Time);

	if (EncodeVideo)
	{
		EncodeVideo = !WriteVideoFrame(FormatContext, &VideoStream, Buffer, Width, Height, Time);
	}
	else
	{
		FFmpegLog("fail encode");
	}
 
}

void FFmpegVideoEncoder::AddAudioBuffer(const void* Buffer, const size_t Size)
{
	ReceivedAudioBufferSize += Size;
	
	if (AudioEncodeBuffer == nullptr)
	{
		return;
	}
	if (EncodeAudio)
	{
		const char* Cursor = static_cast<const char*>(Buffer);
		size_t buffer_left = Size;
		while (buffer_left > 0)
		{
			if (UsedAudioBufferSize + buffer_left >= TotalAudioBufferSize)
			{
				const int WriteSize = TotalAudioBufferSize - UsedAudioBufferSize;
				memcpy(AudioEncodeBuffer + UsedAudioBufferSize, Cursor, WriteSize);
				EncodeAudio = !WriteAudioFrame(FormatContext, &AudioStream, AudioEncodeBuffer, TotalAudioBufferSize);
				buffer_left -= WriteSize;
				Cursor += WriteSize;
				UsedAudioBufferSize = 0;
			}
			else
			{
				const int WriteSize = buffer_left;
				memcpy(AudioEncodeBuffer + UsedAudioBufferSize, Cursor, WriteSize);
				buffer_left = 0;
				UsedAudioBufferSize += WriteSize;
			}
		}
		
	}
}

void FFmpegVideoEncoder::Flush()
{
}

void FFmpegVideoEncoder::EndEncoder()
{

	free(AudioEncodeBuffer);
	
	UsedAudioBufferSize = 0;

	FlushOutputStream(FormatContext, &VideoStream);
	FlushOutputStream(FormatContext, &AudioStream);

	av_write_trailer(FormatContext);

	avio_flush(FormatContext->pb);

	/* Close each codec. */
	if (bHasVideo)
		CloseStream(&VideoStream);
	if (bHasAudio)
		CloseStream(&AudioStream);
	if (!(OutputFormat->flags & AVFMT_NOFILE))
		/* Close the output file. */
		avio_closep(&FormatContext->pb);

	/* free the stream */
	avformat_free_context(FormatContext);


}


