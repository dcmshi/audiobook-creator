import numpy as np

from audiobook_creator.synthesize.base import register_backend


class StubBackend:
    """Deterministic tone generator: 10 ms of 440 Hz per character. For tests/CI."""

    name = "stub"
    sample_rate = 24000

    def synthesize(self, text: str, voice: str) -> bytes:
        n_samples = max(1, int(0.010 * len(text) * self.sample_rate))
        t = np.arange(n_samples, dtype=np.float64)
        samples = 0.2 * np.sin(2 * np.pi * 440.0 * t / self.sample_rate)
        return (samples * 32767).astype("<i2").tobytes()


register_backend("stub", StubBackend, is_local=True)
