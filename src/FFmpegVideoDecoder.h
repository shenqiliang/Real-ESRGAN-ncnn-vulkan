#pragma once

#include "FFmpegCommon.h"
#include <string>

// frame type enum
enum class FFrameType
{
	None = 0,
	Video = 1, // video type
	Audio = 2 // audio type
};

// video or audio frame structure
struct FMediaFrame
{
	// frame type. audio or video
	FFrameType Type = FFrameType::None;

	// frame index in video file
	int64 Index = 0;

	// frame time in video
	double Time = 0;

	// frame data
	uint8* Buffer = nullptr;

	// frame data size
	size_t Size = 0;

	// video width. The value is 0 for audio type
	int Width = 0;

	// video height. The value is 0 for audio type
	int Height = 0;

	// frame pts
	int64 Pts = 0;
};

// reset frame structure. Set all field of the MediaFrame to 0.
static inline void ResetMediaFrame(FMediaFrame &MediaFrame)
{
	MediaFrame.Type = FFrameType::None;
	if (MediaFrame.Buffer != nullptr)
	{
		free(MediaFrame.Buffer);
		MediaFrame.Buffer = nullptr;
	}
	MediaFrame.Size = 0;
	MediaFrame.Height = 0;
	MediaFrame.Width = 0;
	MediaFrame.Time = 0;
	MediaFrame.Pts = 0;
}

// video decoder class
class FFmpegVideoDecoder
{
public:
	// constructor & destructor
	FFmpegVideoDecoder(const std::string FilePath);
	~FFmpegVideoDecoder();

	// get width of the video
	int GetWidth() const;

	// get height of the video
	int GetHeight() const;

	// seek to time
	bool Seek(const float Time) const;

	// decode a video or audio frame.
	int DecodeMedia(FMediaFrame &MediaFrame);
	
private:

	// C string representation for video file path.
	char* VideoFilePath;
	
	// format context
	AVFormatContext* DecodeFormatCtx = nullptr;

	// decoders
	AVCodecContext* VideoDecodeCtx = nullptr;
	AVCodecContext* AudioDecodeCtx = nullptr;

	// pixel format
	AVPixelFormat PixelFormat = AV_PIX_FMT_NONE;

	// streams
	AVStream* VideoStream = nullptr;
	AVStream* AudioStream = nullptr;
	int VideoStreamIndex = -1;
	int AudioStreamIndex = -1;

	// temporary flag for one frame decoding 
	bool bDecodeFinished = false;

	// temporary structure to save decoding data & info
	FMediaFrame* DestMediaFrame = nullptr;

	// video size
	int VideoWidth = 0;
	int VideoHeight = 0;

	// internal error code
	int ErrorCode = 0;

	// reusable frame & packet structure 
	AVFrame* DecodingFrame = nullptr;
	AVPacket* DecodingPacket = nullptr;

	// frame counters & indexes
	int64 VideoFrameNum = 0;
	int64 AudioFrameNum = 0;

	// video & audio format converter contexts
	SwsContext* SwsCtx = nullptr;
	SwrContext* SwrCtx = nullptr;

	// flag that be marked as init failed
	bool bInitFailed = false;

	// Decode a packet
	int DecodePacket(AVCodecContext* DecodeCtx, const AVPacket* Pkt);


	// output a video frame
	int OutputVideoFrame(const AVFrame* Frame);

	// output an audio frame
	int OutputAudioFrame(const AVFrame* Frame);

	// Open codec context
	int OpenCodecContext(int* StreamIndex, AVCodecContext** DecCtxPtr, AVFormatContext* FormatCtx, const AVMediaType Type);


	
};
