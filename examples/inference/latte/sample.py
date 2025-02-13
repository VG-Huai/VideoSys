from videosys import LatteConfig, VideoSysEngine
import time

def run_base():
    # change num_gpus for multi-gpu inference
    config = LatteConfig("maxin-cn/Latte-1", num_gpus=1)
    engine = VideoSysEngine(config)

    # prompt = "Sunset over the sea."
    prompt = "goldfish in glass"
    # prompt = "a horse"
    # video size is fixed to 16 frames, 512x512.
    # seed=-1 means random seed. >0 means fixed seed.
    start_time = time.time()
    video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=123,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f}s")


def run_low_mem():
    config = LatteConfig("maxin-cn/Latte-1", cpu_offload=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = LatteConfig("maxin-cn/Latte-1", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_low_mem()
    # run_pab()
