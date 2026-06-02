import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import os

def main():
    # 1. 创建保存目录
    save_dir = "pointclouds"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 配置 RealSense 流水线
    pipeline = rs.pipeline()
    config = rs.config()

    # 显式指定较低分辨率，防止虚拟机 USB 带宽爆炸
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    print("正在启动相机...")
    profile = pipeline.start(config)

    # 3. 创建对齐对象 (将深度图对齐到彩色图)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取相机内参（Open3D 生成点云时需要）
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy
    )

    print("相机启动成功！")
    print("操作指南：")
    print(" - 按下 's' 键：抓取当前帧并保存为 3D 点云 (.ply)")
    print(" - 按下 'q' 键：退出程序")

    try:
        # 给相机的自动曝光一点时间来稳定
        for _ in range(10):
            pipeline.wait_for_frames()

        while True:
            # 4. 等待最新的一帧数据
            frames = pipeline.wait_for_frames()

            # 5. 将深度图对齐到彩色图
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # 将 RealSense 的数据格式转换成 Numpy 矩阵，方便 OpenCV 显示
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 将深度图渲染成彩色伪彩图，方便肉眼观察
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 将彩色图和深度图横向拼接在一起显示
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('RealSense Camera (Press S to Save, Q to Quit)', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Camera (Press S to Save, Q to Quit)', images)

            # 6. 处理键盘输入
            key = cv2.waitKey(1)
            
            # 按 'q' 退出
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

            # 按 's' 键保存点云
            elif key & 0xFF == ord('s'):
                print("\n正在生成 3D 点云...")
                
                # 将 Numpy 矩阵转为 Open3D 可以识别的格式
                # 注意：OpenCV 默认是 BGR，Open3D 需要 RGB
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                o3d_color = o3d.geometry.Image(color_image_rgb)
                o3d_depth = o3d.geometry.Image(depth_image)

                # 将彩色和深度合成 RGBD 图像
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color, o3d_depth,
                    depth_scale=1000.0, # D435 的深度单位默认是毫米 (1/1000米)
                    depth_trunc=3.0,    # 砍掉 3 米外的背景，不要它们
                    convert_rgb_to_intensity=False
                )

                # 根据 RGBD 图像和相机内参生成点云
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, o3d_intrinsics
                )
                
                # 翻转点云方向（因为 Open3D 和 RealSense 的坐标系定义方向不同）
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                # 保存为 PLY 文件
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"scene_{timestamp}.ply")
                o3d.io.write_point_cloud(filename, pcd)
                print(f"✅ 点云已成功保存至: {filename}")

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 关闭相机
        pipeline.stop()
        print("程序已结束。")

if __name__ == "__main__":
    main()