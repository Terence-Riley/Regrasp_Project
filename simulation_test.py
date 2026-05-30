import pybullet as p
import pybullet_data
import time

# 1. 启动 PyBullet 的图形界面
physicsClient = p.connect(p.GUI)

# 2. 设置搜索路径，方便加载默认的地面
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 3. 加载地面
planeId = p.loadURDF("plane.urdf")

# 4. 加载你刚刚处理好的 Gen3 Lite 机械臂
# 注意这里的路径要根据你实际存放 URDF 的位置来写
robot_urdf_path = "kortex_description/robots/gen3_lite.urdf"
print(f"正在加载机械臂: {robot_urdf_path}")

# useFixedBase=True 确保机械臂底座钉死在地上，不会乱跑
robotId = p.loadURDF(robot_urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

print("加载成功！虚拟世界已启动。")

# 5. 保持仿真运行
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.) # 仿真步长
except KeyboardInterrupt:
    p.disconnect()
    print("仿真已关闭。")