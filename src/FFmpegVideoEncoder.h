#pragma once
#include "FFmpegCommon.h"
#include "FFmpegMuxing.h"
#include <string>

class FFmpegVideoEncoder
{
public:
	/**
	 * @brief Create encoder to encode audio & video to file
	 * @param Width Width of the video
	 * @param Height Height of the video
	 * @param FilePath File path of the video file to write.
	 */
	FFmpegVideoEncoder(int Width, int Height, std::string FilePath);

	/**
	 * @brief add video buffer data of one frame.
	 * @param Buffer Packed BGRA bytes, each channel has 8 bits. It size must be Width*Height*4 bytes.
	 * @param Width Video width. if it's 0, use video file width that been set in constructor
	 * @param Height Video height. if it's 0, use video file height that been set in constructor
	 */
	void AddVideoBuffer(const char* Buffer, int Width, int Height, double Time);

	/**
	 * @brief add audio buffer data. The buffer is packed audio data. The format is defined by AUDIO_ENC_CHANNELS, AUDIO_ENC_CHANNEL_LAYOUT,
	 *	AUDIO_ENC_FORMAT, AUDIO_SAMPLE_RATE macros.
	 */
	void AddAudioBuffer(const void* Buffer, const size_t Size);

	// finish the encoding
	void EndEncoder();

private:
	// file path
	std::string VideoFilePath;

	// output format
	AVOutputFormat* OutputFormat{nullptr};
	AVFormatContext* FormatContext{nullptr};

	// video & audio stream
	FOutputStream VideoStream = { nullptr };
	FOutputStream AudioStream = { nullptr };

	// video & audio codec
	AVCodec* AudioCodec{nullptr}, *VideoCodec{nullptr};

	// internal ret code
	int Ret = 0;

	// whether has video stream to encode
	bool bHasVideo = false;

	// whether has audio stream to encode
	bool bHasAudio = false;

	// encode code for video or audio
	int EncodeVideo = 0;
	int EncodeAudio = 0;

	// audio buffer to encode
	char* AudioEncodeBuffer = nullptr;

	// used/cached audio buffer size
	int UsedAudioBufferSize = 0;

	// max/total audio buffer size
	int TotalAudioBufferSize = 0;

	// number of audio's bytes already encoded. 
	int64 ReceivedAudioBufferSize = 0;
	
};
