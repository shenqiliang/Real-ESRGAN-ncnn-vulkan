#include "FFmpegCommon.h"

void FFmpegLog(const char* Format, ...)
{
	va_list List; 
	va_start(List, Format);
	char Buffer[1024] = {0};
	vsnprintf(Buffer, 1024, Format, List);
	va_end(List);
	printf("[ffmpeg]%s", Buffer);
}

char* CopyCString(const char* CStr)
{
	const size_t Len = strlen(CStr);
	char *Result = static_cast<char*>(malloc(Len + 1));
#if PLATFORM_WINDOWS
	strncpy_s(Result, Len + 1, CStr, Len);
#else
	strncpy(Result, CStr, Len);
#endif
	Result[Len] = 0;
	return Result;
}