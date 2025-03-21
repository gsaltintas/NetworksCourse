class DynamicPrecisionScheduler:
    def __init__(self, policy, network_monitor, model_info, total_steps: int):
        self.policy = policy
        self.network_monitor = network_monitor
        self.model_info = model_info
        self.training_step = 0
        self.total_steps = total_steps

    def get_precision(self, src_device, dst_device, tensor_info, is_backward=False):
        """
        Determine the precision to use for communication.

        Args:
            src_device: Source device ID
            dst_device: Destination device ID
            tensor_info: Dict containing metadata about the tensor being communicated
                         (e.g., layer_name, layer_type, tensor_shape)
            is_backward: Whether this is during backward pass

        Returns:
            Precision enum (e.g., FP32, FP16, FP8)
        """
        # Collect relevant information
        network_stats = self.network_monitor.get_link_stats(src_device, dst_device)
        model_context = {
            "tensor_info": tensor_info,
            "is_backward": is_backward,
            "training_progress": self.training_step / self.total_steps,
        }

        # Get decision from policy
        precision = self.policy.select_precision(
            network_stats=network_stats,
            model_context=model_context,
            src_dst_pair=(src_device, dst_device),
        )

        return precision

    def update_step(self, update: int = 1):
        self.training_step += update
