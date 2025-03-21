class PrecisionPolicy:
    def select_precision(self, network_stats, model_context, src_dst_pair):
        """
        Select precision based on network conditions and model context.

        Args:
            network_stats: Network statistics for the relevant link
            model_context: Information about the model and current training state
            src_dst_pair: (source_device, destination_device) tuple

        Returns:
            Precision enum value
        """
        raise NotImplementedError
