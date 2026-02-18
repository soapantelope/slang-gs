import slangpy as spy


WAVE_SIZE = 32


class PrefixSum:
    def __init__(self, device: spy.Device):
        self._device = device
        # Load modules.
        self._utils_module = spy.Module.load_from_file(
            device=device, path="utils.slang"
        )

    def scan(self, values: spy.NDBuffer):
        wave_scan = self._utils_module.find_function(f"waveScan<{values.dtype.name}>")
        offset_scan = self._utils_module.find_function(
            f"offsetScan<{values.dtype.name}>"
        )

        num_values = values.shape[0]
        # (src_buf, src_size, partial_buf, partial_size)
        stack = []
        cur_buf = values
        cur_size = num_values

        # Upsweep phase
        while True:
            num_waves = (cur_size + WAVE_SIZE - 1) // WAVE_SIZE
            partial_buf = spy.NDBuffer(
                device=self._device, dtype=values.dtype, shape=(num_waves,)
            )
            wave_scan(
                tid=spy.grid(shape=(cur_size,)),
                numElements=cur_size,
                src=cur_buf,
                partial=partial_buf,
            )
            stack.append((cur_buf, cur_size, partial_buf, num_waves))
            if num_waves == 1:
                break
            cur_buf = partial_buf
            cur_size = num_waves

        # Downsweep phase
        for src_buf, src_size, partial_buf, _ in reversed(stack):
            offset_scan(
                tid=spy.grid(shape=(src_size,)),
                numElements=src_size,
                src=src_buf,
                partial=partial_buf,
            )

    def segmented_scan(self, values: spy.NDBuffer, flags: spy.NDBuffer):
        if flags.shape[0] != values.shape[0]:
            raise ValueError("values and flags must have the same length")

        wave_scan = self._utils_module.find_function(
            f"waveSegmentedScan<{values.dtype.name}>"
        )
        offset_scan = self._utils_module.find_function(
            f"offsetSegmentedScan<{values.dtype.name}>"
        )

        num_values = values.shape[0]
        stack = []
        cur_buf = values
        cur_flags = flags
        cur_size = num_values

        while True:
            num_waves = (cur_size + WAVE_SIZE - 1) // WAVE_SIZE
            partial_buf = spy.NDBuffer(
                device=self._device, dtype=values.dtype, shape=(num_waves,)
            )
            partial_flags = spy.NDBuffer(
                device=self._device, dtype=flags.dtype, shape=(num_waves,)
            )
            wave_scan(
                tid=spy.grid(shape=(cur_size,)),
                numElements=cur_size,
                src=cur_buf,
                flags=cur_flags,
                partial=partial_buf,
                partialFlags=partial_flags,
            )
            stack.append(
                (cur_buf, cur_flags, cur_size, partial_buf, partial_flags, num_waves)
            )
            if num_waves == 1:
                break
            cur_buf = partial_buf
            cur_flags = partial_flags
            cur_size = num_waves

        for src_buf, src_flags, src_size, partial_buf, _, _ in reversed(stack):
            offset_scan(
                tid=spy.grid(shape=(src_size,)),
                numElements=src_size,
                src=src_buf,
                flags=src_flags,
                partial=partial_buf,
            )
