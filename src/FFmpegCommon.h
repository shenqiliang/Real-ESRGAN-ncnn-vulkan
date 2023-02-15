
#pragma once
#define INT8_C(x)    (x)
#define INT16_C(x)   (x)
#define INT32_C(x)   (x)
#define INT64_C(x)   (x ## LL)

#define UINT8_C(x)   (x)
#define UINT16_C(x)  (x)
#define UINT32_C(x)  (x ## U)
#define UINT64_C(x)  (x ## ULL)

#define INTMAX_C(x)  INT64_C(x)
#define UINTMAX_C(x) UINT64_C(x)

typedef long long int64;
typedef unsigned char uint8;

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include "libswscale/swscale.h"
#include "libavformat/avformat.h"
}

// error codes
enum class EFFmpegResult: int
{
	FormatNotMatch = 50000
};

// convert ffmpeg enums to string desc
static char ErrorStrBuffer[AV_TS_MAX_STRING_SIZE] = {0};
#define AV_TS2STR(ts) av_ts_make_string(ErrorStrBuffer, ts)
#define AV_TS2TIME_STR(ts, tb) av_ts_make_time_string(ErrorStrBuffer, ts, tb)
#define AV_ERR2STR(ErrorNumber) av_make_error_string(ErrorStrBuffer, AV_ERROR_MAX_STRING_SIZE, ErrorNumber)

// print logs function
void FFmpegLog(const char* Format, ...);

// copy c string to heap
char* CopyCString(const char* CStr);

// a wrapper around a single output AVStream
struct FOutputStream {
	
	// audio or video stream
	AVStream* Stream = nullptr;

	// encoder context
	AVCodecContext* EncodeCtx = nullptr;

	/* pts of the next frame that will be generated */
	int64_t NextPts;

	// number of sample encoded
	int SampleNum;

	// reused frame structure
	AVFrame* Frame;

	// reused temporary frame structure
	AVFrame* TmpFrame;

	// reused temporary packet structure
	AVPacket* TmpPkt;

	// video format convert context
	SwsContext* SwsCtx;

	// audio format convert context
	SwrContext* SwrCtx;
};

// input audio data format defines for encoding
#define AUDIO_ENC_CHANNELS 2
#define AUDIO_ENC_CHANNEL_LAYOUT AV_CH_LAYOUT_STEREO
#define AUDIO_ENC_FORMAT AV_SAMPLE_FMT_S16
#define AUDIO_SAMPLE_RATE 44100

// video encode format defines
#define STREAM_FRAME_RATE 30 /* 25 images/s */
#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P /* default pix_fmt */
#define SCALE_FLAGS SWS_BICUBIC // video convert flag
