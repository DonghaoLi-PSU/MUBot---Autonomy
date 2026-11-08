import time, rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import WorldControl
from ros_gz_interfaces.srv import ControlWorld


class GazeboConnection:
    def __init__(self, node: Node, world_name: str = 'default',
                 timeout: float = 5.0, connect_timeout: float = 30.0,
                 retries: int = 3, pause_on_connect: bool = False):
        self.node = node
        self.timeout = timeout
        self.retries = retries
        self.srv_name = f'/world/{world_name}/control'
        self.cli = node.create_client(ControlWorld, self.srv_name)

        # Bounded wait: if the sim or bridge died we must fail, not hang forever.
        deadline = time.monotonic() + connect_timeout
        while not self.cli.wait_for_service(timeout_sec=0.5):
            if time.monotonic() > deadline:
                node.destroy_client(self.cli)
                raise TimeoutError(
                    f'{self.srv_name} not available after {connect_timeout:.0f}s '
                    '(is the sim running and is the bridge started?)')

        if pause_on_connect:
            self.pause_sim()

    def _call(self, retry: bool = True, **fields) -> bool:
        req = ControlWorld.Request()
        req.world_control = WorldControl(**fields)
        attempts = self.retries if retry else 1
        for attempt in range(1, attempts + 1):
            future = self.cli.call_async(req)
            rclpy.spin_until_future_complete(
                self.node, future, timeout_sec=self.timeout)
            if future.done() and future.result() is not None:
                if future.result().success:
                    return True
            else:
                # drop the dead request so it cannot leak / fire later
                future.cancel()
                try:
                    self.cli.remove_pending_request(future)
                except Exception:
                    pass
            if attempt < attempts:
                time.sleep(0.5)
        self.node.get_logger().error(
            f'WorldControl {fields} failed after {attempts} attempt(s)')
        return False

    def pause_sim(self) -> bool:
        return self._call(pause=True)

    def unpause_sim(self) -> bool:
        return self._call(pause=False)

    def step(self, n: int = 1) -> bool:
        """Advance n physics steps, then stay paused (not retried: not idempotent)."""
        return self._call(retry=False, pause=True, multi_step=n)

    def close(self):
        try:
            self.node.destroy_client(self.cli)
        except Exception:
            pass