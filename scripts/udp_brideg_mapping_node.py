#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from grid_map_msgs.msg import GridMap
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.serialization import deserialize_message
import tf2_ros
import socket
import numpy as np
import threading
import time
import struct

def recvall(sock, n):
    """辅助函数：确保从TCP流中完整读取 n 个字节"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

class TcpBridgeANode(Node):
    def __init__(self):
        super().__init__('tcp_bridge_a_node')

        self.LISTEN_PORT = 5002
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", self.LISTEN_PORT))
        self.server_sock.listen(1)
        self.client_sock = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.lidar_sub = self.create_subscription(
            PointCloud2, '/LIDAR_SIM_RAW', self.lidar_callback, 10)

        self.map_pub = self.create_publisher(GridMap, '/elevation_map_remote', 10)

        self.get_logger().info(f"Node A (Server) started. Listening on TCP port {self.LISTEN_PORT}...")
        
        self.accept_thread = threading.Thread(target=self.tcp_accept_loop, daemon=True)
        self.accept_thread.start()

    def lidar_callback(self, msg: PointCloud2):
        if self.client_sock is None:
            return # 没有客户端连接时不发数据

        raw_points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if raw_points is None or len(raw_points) == 0:
            return

        if isinstance(raw_points, np.ndarray):
            points = np.column_stack((raw_points['x'], raw_points['y'], raw_points['z'])).astype(np.float32)
        else:
            points = np.array(list(raw_points), dtype=np.float32)
            if len(points.shape) == 1:
                return

        try:
            t = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            tx, ty, tz = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            qx, qy, qz, qw = t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w
        except Exception:
            tx, ty, tz, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

        tf_header = struct.pack('<7f', tx, ty, tz, qx, qy, qz, qw)
        
        # TCP可以传更大的包，这里把切片调大以提高效率
        CHUNK_SIZE = 5000 
        np.random.shuffle(points)

        for i in range(0, points.shape[0], CHUNK_SIZE):
            chunk_points = points[i : i+CHUNK_SIZE]
            payload = tf_header + chunk_points.tobytes()
            
            # 【TCP核心协议】：先发4字节长度，再发数据
            msg_length = struct.pack('<I', len(payload))
            try:
                self.client_sock.sendall(msg_length + payload)
            except Exception as e:
                self.get_logger().error(f"TCP Send Error (Disconnecting): {e}")
                self.client_sock.close()
                self.client_sock = None
                break

    def tcp_accept_loop(self):
        while True:
            client, addr = self.server_sock.accept()
            self.get_logger().info(f"✅ Node B connected from {addr}")
            self.client_sock = client
            
            # 接收高程图数据的循环
            while True:
                try:
                    # 1. 先读 4 字节的包长
                    raw_msglen = recvall(self.client_sock, 4)
                    if not raw_msglen:
                        break
                    msglen = struct.unpack('<I', raw_msglen)[0]
                    
                    # 2. 按照包长读取完整的数据
                    data = recvall(self.client_sock, msglen)
                    if not data:
                        break
                        
                    msg = deserialize_message(data, GridMap)
                    self.map_pub.publish(msg)
                    self.get_logger().info("✅ Received full GridMap from Node B", throttle_duration_sec=1.0)
                except Exception as e:
                    self.get_logger().error(f"TCP Receive Error: {e}")
                    break
                    
            self.get_logger().warn(f"❌ Node B disconnected.")
            if self.client_sock:
                self.client_sock.close()
            self.client_sock = None

def main(args=None):
    rclpy.init(args=args)
    node = TcpBridgeANode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()