#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.serialization import serialize_message
import tf2_ros
import socket
import numpy as np
import threading
import struct

class UdpBridgeBNode(Node):
    def __init__(self):
        super().__init__('udp_bridge_b_node')

        # --- 请替换为电脑 A 的真实 IP ---
        self.A_IP_PORT = ("192.168.8.103", 5002)
        # self.A_IP_PORT = ("127.0.0.1", 5002)
        self.LOCAL_PORT = 5001

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(("0.0.0.0", self.LOCAL_PORT))

        # 用于向建图算法发布合并后的点云
        self.lidar_pub = self.create_publisher(
            PointCloud2, '/LIDAR_POINT_CLOUD_MERGED', 10)

        # 用于在电脑 B 本地恢复 TF 树
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 监听建图算法产生的内部结果
        self.grid_map_sub = self.create_subscription(
            GridMap, '/elevation_mapping_node/elevation_map_raw', self.gridmap_callback, 10)

        self.recv_thread = threading.Thread(target=self.udp_receive_loop, daemon=True)
        self.recv_thread.start()

        self.get_logger().info("Node B started. Reconstructing TF & Lidar from UDP...")

    def gridmap_callback(self, msg: GridMap):
        try:
            serialized_msg = serialize_message(msg)
            self.sock_send.sendto(serialized_msg, self.A_IP_PORT)
            self.get_logger().info("⬆️ Sent full GridMap to Node A.", throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f"UDP Send Map Error: {e}")

    def udp_receive_loop(self):
        while True:
            try:
                data, addr = self.sock_recv.recvfrom(65535)

                # 校验：至少要有 28 字节的 TF 头
                if len(data) <= 28:
                    continue

                # 1. 拆解头部 28 字节，提取 TF 坐标
                tx, ty, tz, qx, qy, qz, qw = struct.unpack('<7f', data[:28])

                # 2. 剩余部分为纯点云二进制流
                payload = data[28:]
                if len(payload) % 12 != 0:
                    continue

                points_array = np.frombuffer(payload, dtype=np.float32).reshape(-1, 3)

                # 获取电脑 B 的统一本地时间！(彻底解决双机时钟不同步的噩梦)
                current_time = self.get_clock().now().to_msg()

                # 3. 在电脑 B 本地恢复并广播 TF 树
                t = TransformStamped()
                t.header.stamp = current_time
                t.header.frame_id = 'odom'
                t.child_frame_id = 'base_link'
                t.transform.translation.x = tx
                t.transform.translation.y = ty
                t.transform.translation.z = tz
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw
                self.tf_broadcaster.sendTransform(t)

                # 4. 构造雷达消息并发布
                header = Header()
                header.stamp = current_time
                header.frame_id = 'base_link'

                pc2_msg = pc2.create_cloud_xyz32(header, points_array.tolist())
                self.lidar_pub.publish(pc2_msg)

            except Exception as e:
                self.get_logger().error(f"UDP Recv Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UdpBridgeBNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()