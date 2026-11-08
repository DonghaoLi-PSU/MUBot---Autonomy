import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldControl


class GazeboConnection:
    def __init__(self, node: Node, world_name: str = 'default', timeout: float = 5.0):
        self.node = node
        self.timeout = timeout
        self.cli = node.create_client(ControlWorld, f'/world/{world_name}/control')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            node.get_logger().info(f'Waiting for /world/{world_name}/control ...')

        self.pause_sim()

    def _call(self, **fields) -> bool:
        req = ControlWorld.Request()
        req.world_control = WorldControl(**fields)
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout)
        if not future.done() or future.result() is None:
            self.node.get_logger().error('WorldControl call failed / timed out')
            return False
        return future.result().success

    def pause_sim(self) -> bool:
        return self._call(pause=True)

    def unpause_sim(self) -> bool:
        return self._call(pause=False)

    def step(self, n: int = 1) -> bool:
        """Advance n physics steps, then stay paused."""
        return self._call(pause=True, multi_step=n)

    def close(self):
        self.node.destroy_client(self.cli)