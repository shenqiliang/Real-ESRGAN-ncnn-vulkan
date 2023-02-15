#pragma once

#include "FFmpegCommon.h"

// Log packet's information
void LogPacket(const AVFormatContext* FormatCtx, const AVPacket* Pkt);

// write frame to encoder
int WriteFrame(AVFormatContext* FormatCtx, AVCodecContext* Codec, const AVStream* Stream, const AVFrame* Frame);

// flush cached frame to file
int FlushFrames(AVFormatContext* FormatCtx, AVCodecContext* Codec,
	const AVStream* Stream, AVPacket* Pkt);

// add an stream to file
void AddStream(FOutputStream* Ost, AVFormatContext* FormatCtx, AVCodec **Codec, const AVCodecID CodecId, const int Width, const int Height, const int UseHardware, const int Bitrate);

// Alloc a picture data structure
AVFrame* AllocPicture(const AVPixelFormat PixelFormat, const int Width, const int Height);

// open video codec
void OpenVideo(const AVCodec* Codec, FOutputStream* Ost, const AVDictionary* Option);

// open audio codec
void OpenAudio(const AVCodec* Codec, FOutputStream* Ost, const AVDictionary* OptArg);

// fill a video frame
AVFrame* GetVideoFrame(FOutputStream* Ost, const char* Buffer, const int Width, const int Height, double Time);

// write video frame. It will finally call WriteFrame
int WriteVideoFrame(AVFormatContext* FormatCtx, FOutputStream* Ost, const void* Buffer, int Width, int Height, double Time);

// write audio frame. It will finally call WriteFrame
int WriteAudioFrame(AVFormatContext* FormatCtx, FOutputStream* Ost, const void* Buffer, size_t Size);

// flush output stream
int FlushOutputStream(AVFormatContext* FormatCtx, FOutputStream* Ost);

// close output stream
void CloseStream(FOutputStream* Ost);
