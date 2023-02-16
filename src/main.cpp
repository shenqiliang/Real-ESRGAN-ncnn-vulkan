// realesrgan implemented with ncnn library
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>
#include <codecvt>
#include <filesystem>
#include <thread>

#include "FFmpegVideoDecoder.h"
#include "FFmpegVideoEncoder.h"

namespace fs = std::filesystem;


std::string wstr2str(std::wstring string_to_convert)
{

    //setup converter
    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;

    //use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    std::string converted_str = converter.to_bytes( string_to_convert );
    return converted_str;
}

#if _WIN32
#include <wchar.h>
static wchar_t* optarg = NULL;
static int optind = 1;
static wchar_t getopt(int argc, wchar_t* const argv[], const wchar_t* optstring)
{
    if (optind >= argc || argv[optind][0] != L'-')
        return -1;

    wchar_t opt = argv[optind][1];
    const wchar_t* p = wcschr(optstring, opt);
    if (p == NULL)
        return L'?';

    optarg = NULL;

    if (p[1] == L':')
    {
        optind++;
        if (optind >= argc)
            return L'?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}

static std::vector<int> parse_optarg_int_array(const wchar_t* optarg)
{
    std::vector<int> array;
    array.push_back(_wtoi(optarg));

    const wchar_t* p = wcschr(optarg, L',');
    while (p)
    {
        p++;
        array.push_back(_wtoi(p));
        p = wcschr(p, L',');
    }

    return array;
}
#else // _WIN32
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}
#endif // _WIN32

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "realesrgan.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stderr, "Usage: realesrgan-ncnn-vulkan -i infile -o outfile [options]...\n\n");
    fprintf(stderr, "  -h                   show this help\n");
    fprintf(stderr, "  -i input-path        input image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) or directory\n");
    fprintf(stderr, "  -s scale             upscale ratio (can be 2, 3, 4. default=4)\n");
    fprintf(stderr, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stderr, "  -m model-path        folder path to the pre-trained models. default=models\n");
    fprintf(stderr, "  -n model-name        model name (default=realesr-animevideov3, can be realesr-animevideov3 | realesrgan-x4plus | realesrgan-x4plus-anime | realesrnet-x4plus)\n");
    fprintf(stderr, "  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stderr, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stderr, "  -x                   enable tta mode\n");
    fprintf(stderr, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
    fprintf(stderr, "  -v                   verbose output\n");
}

enum class TaskStatus
{
    None,
    Decoded,
    EsrDone,
    Encoded
};

class Task
{
public:
    int id = 0;

    double Time = 0;

    TaskStatus status = TaskStatus::None;
    
    ncnn::Mat inimage;
    ncnn::Mat outimage;
};

template <typename T>
class TaskQueue
{
public:
    const int cap;
    
    TaskQueue(const int cap): cap(cap)
    {
    }

    void put(const T& v)
    {
        lock.lock();

        while (tasks.size() >= cap) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void pop(T& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

    int pop_no_block(T& v)
    {
        int ret = 0;
        lock.lock();

        if (tasks.size() == 0)
        {
            ret = -1;
        }
        else
        {
            v = tasks.front();
            tasks.pop();
            ret = 1;
        }


        lock.unlock();
        
        return ret;
    }


private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<T> tasks;
};

TaskQueue<std::shared_ptr<Task>> toproc(8);
TaskQueue<std::shared_ptr<Task>> tosave(40);
TaskQueue<FMediaFrame> audioqueue(INT32_MAX);
FFmpegVideoDecoder *VideoDecoder;


class LoadThreadParams
{
public:
    int scale;
};

void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    const int scale = ltp->scale;

    int index = 0;
    FMediaFrame media_frame;
    while (VideoDecoder->DecodeMedia(media_frame) == 0 && media_frame.Time < 0.1)
    {
        if (media_frame.Type == FFrameType::Video)
        {
            unsigned char* pixeldata = media_frame.Buffer;
            if (pixeldata)
            {
                auto v = std::make_shared<Task>();
                v->id = index++;
                v->Time = media_frame.Time;

                fprintf(stderr, "decode video time: %f\n", v->Time);

                v->inimage = ncnn::Mat(media_frame.Width, media_frame.Height, (void*)pixeldata, (size_t)3, 3);
                v->outimage = ncnn::Mat(media_frame.Width * scale, media_frame.Height * scale, (size_t)3, 3);

                v->status = TaskStatus::Decoded;
                
                toproc.put(v);
                tosave.put(v);

            }
        }
        else if (media_frame.Type == FFrameType::Audio)
        {
            audioqueue.put(media_frame);
        }
    }
    
    return 0;
}

class ProcThreadParams
{
public:
    const RealESRGAN* realesrgan;
};

void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const RealESRGAN* realesrgan = ptp->realesrgan;

    for (;;)
    {
        std::shared_ptr<Task> v;

        toproc.pop(v);

        if (v->id == -233)
            break;

        realesrgan->process(v->inimage, v->outimage);
        void *buffer = v->inimage.data;
        v->inimage.release();
        if (buffer != nullptr)
        {
            free(buffer);
        }
        v->status = TaskStatus::EsrDone;
    }


    return 0;
}

class SaveThreadParams
{
public:
    int verbose;

    int width;
    int height;
    
    std::string save_path;
};

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;

    FFmpegVideoEncoder encoder(stp->width, stp->height, stp->save_path);
    
    for (;;)
    {
        std::shared_ptr<Task> v;

        tosave.pop(v);

        if (v->id == -233)
            break;

        while (v->status != TaskStatus::EsrDone)
        {
            encoder.Flush();
            if (v->status != TaskStatus::EsrDone)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        FMediaFrame audio_frame;
        while (audioqueue.pop_no_block(audio_frame) > 0)
        {
            encoder.AddAudioBuffer(audio_frame.Buffer, audio_frame.Size);
        }
        
        encoder.AddVideoBuffer((const char*)v->outimage.data, v->outimage.w, v->outimage.h, v->Time);
        v->outimage.release();

    }

    encoder.EndEncoder();

    return 0;
}


#if _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    auto begin_time = std::chrono::system_clock::now();
    path_t inputpath;
    path_t outputpath;
    int scale = 2;
    std::vector<int> tilesize;
    path_t model = PATHSTR("models");
    path_t modelname = PATHSTR("realesr-animevideov3");
    std::vector<int> gpuid;
    std::vector<int> jobs_proc;
    int verbose = 0;
    int tta_mode = 0;

#if _WIN32
    setlocale(LC_ALL, "");
    wchar_t opt;
    while ((opt = getopt(argc, argv, L"i:o:s:t:m:n:g:j:f:vxh")) != (wchar_t)-1)
    {
        switch (opt)
        {
        case L'i':
            inputpath = optarg;
            break;
        case L'o':
            outputpath = optarg;
            break;
        case L's':
            scale = _wtoi(optarg);
            break;
        case L't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case L'm':
            model = optarg;
            break;
        case L'n':
            modelname = optarg;
            break;
        case L'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case L'j':
            jobs_proc = parse_optarg_int_array(wcschr(optarg, L':') + 1);
            break;
        case L'v':
            verbose = 1;
            break;
        case L'x':
            tta_mode = 1;
            break;
        case L'h':
        default:
            print_usage();
            return -1;
        }
    }
#else // _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "i:o:s:t:m:n:g:j:f:vxh")) != -1)
    {
        switch (opt)
        {
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 's':
            scale = atoi(optarg);
            break;
        case 't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'n':
            modelname = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        case 'f':
            format = optarg;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }
#endif // _WIN32

    if (inputpath.empty() || outputpath.empty())
    {
        print_usage();
        return -1;
    }

    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) && !tilesize.empty())
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    for (int i=0; i<(int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "invalid tilesize argument\n");
            return -1;
        }
    }


    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }


    int prepadding = 0;

    if (model.find(PATHSTR("models")) != path_t::npos
        || model.find(PATHSTR("models2")) != path_t::npos)
    {
        prepadding = 10;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    // if (modelname.find(PATHSTR("realesrgan-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("realesrnet-x4plus")) != path_t::npos
    //     || modelname.find(PATHSTR("esrgan-x4")) != path_t::npos)
    // {}
    // else
    // {
    //     fprintf(stderr, "unknown model name\n");
    //     return -1;
    // }

#if _WIN32
    wchar_t parampath[256];
    wchar_t modelpath[256];

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        swprintf(parampath, 256, L"%s/%s-x%s.param", model.c_str(), modelname.c_str(), std::to_wstring(scale).c_str());
        swprintf(modelpath, 256, L"%s/%s-x%s.bin", model.c_str(), modelname.c_str(), std::to_wstring(scale).c_str());
    }
    else{
        swprintf(parampath, 256, L"%s/%s.param", model.c_str(), modelname.c_str());
        swprintf(modelpath, 256, L"%s/%s.bin", model.c_str(), modelname.c_str());
    }

#else
    char parampath[256];
    char modelpath[256];

    if (modelname == PATHSTR("realesr-animevideov3"))
    {
        sprintf(parampath, "%s/%s-x%s.param", model.c_str(), modelname.c_str(), std::to_string(scale).c_str());
        sprintf(modelpath, "%s/%s-x%s.bin", model.c_str(), modelname.c_str(), std::to_string(scale).c_str());
    }
    else{
        sprintf(parampath, "%s/%s.param", model.c_str(), modelname.c_str());
        sprintf(modelpath, "%s/%s.bin", model.c_str(), modelname.c_str());
    }
#endif

    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

// #if _WIN32
//     CoInitializeEx(NULL, COINIT_MULTITHREADED);
// #endif

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 1);
    }

    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < 0 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        int gpu_queue_count = ncnn::get_gpu_info(gpuid[i]).compute_queue_count();
        jobs_proc[i] = std::min(jobs_proc[i], gpu_queue_count);
        total_jobs_proc += jobs_proc[i];
    }

    for (int i=0; i<use_gpu_count; i++)
    {
        if (tilesize[i] != 0)
            continue;

        uint32_t heap_budget = ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

        // more fine-grained tilesize policy here
        if (model.find(PATHSTR("models")) != path_t::npos)
        {
            if (heap_budget > 8000)
            {
                tilesize[i] = 1920;
            }
            else if (heap_budget > 1900)
                tilesize[i] = 200;
            else if (heap_budget > 550)
                tilesize[i] = 100;
            else if (heap_budget > 190)
                tilesize[i] = 64;
            else
                tilesize[i] = 32;
        }
    }

    {
        std::vector<RealESRGAN*> realesrgan(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            realesrgan[i] = new RealESRGAN(gpuid[i], tta_mode);

            realesrgan[i]->load(paramfullpath, modelfullpath);

            realesrgan[i]->scale = scale;
            realesrgan[i]->tilesize = tilesize[i];
            realesrgan[i]->prepadding = prepadding;
        }

        // main routine
        {
            // load image
            LoadThreadParams ltp;
            ltp.scale = scale;
#if _WIN32
            VideoDecoder = new FFmpegVideoDecoder(wstr2str(inputpath));
#else
            VideoDecoder = new FFmpegVideoDecoder(inputpath);
#endif
            

            ncnn::Thread load_thread(load, (void*)&ltp);

            // realesrgan proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].realesrgan = realesrgan[i];
            }

            
            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    for (int j=0; j<jobs_proc[i]; j++)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;
            stp.width = VideoDecoder->GetWidth();
            stp.height = VideoDecoder->GetHeight();
#if _WIN32
            stp.save_path = wstr2str(outputpath);
#else
            stp.save_path = outputpath;
#endif

            ncnn::Thread *save_thread = new ncnn::Thread(save, (void*)&stp);


            // end
            load_thread.join();

            auto end = std::make_shared<Task>();
            end->id = -233;

            for (int i=0; i<total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }
            tosave.put(end);
            
            save_thread->join();
            delete save_thread;
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete realesrgan[i];
        }
        realesrgan.clear();
    }

    ncnn::destroy_gpu_instance();

    auto end_time = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time);

    fprintf(stderr, "Convert Total Cost: %lld\n", diff.count());
    return 0;
}
