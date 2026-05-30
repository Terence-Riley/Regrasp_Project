import pyrealsense2 as rs
import time

# 1. 创建配置对象
pipeline = rs.pipeline()
config = rs.config()

# 💡 关键：显式指定低分辨率和低帧率，减少带宽占用
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

print("正在启动相机...")
try:
    # 2. 启动流水线
    pipeline.start(config)
    
    # 给硬件一点预热时间
    time.sleep(1)

    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        width, height = depth_frame.get_width(), depth_frame.get_height()
        dist = depth_frame.get_distance(width // 2, height // 2)
        
        print(f"当前中心点距离: {dist:.3f} 米", end="\r")

except Exception as e:
    print(f"\n发生错误: {e}")

finally:
    pipeline.stop()
    print("\n相机已关闭")