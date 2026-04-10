#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from grid_map_msgs.msg import GridMap
import sensor_msgs_py.point_cloud2 as pc2
import socket
import numpy as np
import threading
from rclpy.serialization import serialize_message

class UdpBridgeBNode(Node):
    def __init__(self):
        super().__init__('udp_bridge_b_node')

        self.A_IP_PORT = ("127.0.0.1", 5002)
        self.LOCAL_PORT = 5001

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(("0.0.0.0", self.LOCAL_PORT))

        self.lidar_pub = self.create_publisher(
            PointCloud2, '/LIDAR_POINT_CLOUD_MERGED', 10)

        self.grid_map_sub = self.create_subscription(
            GridMap,
            '/elevation_mapping_node/elevation_map_raw',
            self.gridmap_callback,
            10)

        self.recv_thread = threading.Thread(target=self.udp_receive_loop, daemon=True)
        self.recv_thread.start()

        self.get_logger().info("Node B started. Processing live point chunks...")

    def gridmap_callback(self, msg: GridMap):
        try:
            # 核心黑科技：直接序列化完整的 ROS2 消息（包含中心位姿和分辨率）
            serialized_msg = serialize_message(msg)
            self.sock_send.sendto(serialized_msg, self.A_IP_PORT)
            self.get_logger().info("⬆️ Sent full GridMap via UDP.", throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f"UDP Send Map Error: {e}")

    def udp_receive_loop(self):
        while True:
            try:
                data, addr = self.sock_recv.recvfrom(65535)
                # 保护机制：如果字节数不是 12 的倍数 (x,y,z 各 4 字节)，说明数据损坏
                if len(data) % 12 != 0:
                    continue

                # 直接将收到的纯字节块恢复为 Numpy 矩阵
                points_array = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)

                # 立即构造 ROS2 消息并发布 (完全无感知的碎片化发布)
                header = Header()
                header.stamp = self.get_clock().now().to_msg()

                # 修复关键点：必须是 base_link 才能在建图节点中正确配准！
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