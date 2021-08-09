from contextlib import AbstractContextManager
from pathlib import Path

import ffmpeg
import numpy as np


class VAEVideo(AbstractContextManager):
    """Create a video of the observation and the VAE reconstructed image side-by-side.
        
        Parameters
        ----------
        output : Path
            Output file path and name

        video_width : int
            Full final width of video (must be greater than the sum of the two arrays)

        video_height : int
            Full final height of video (must be greater than the sum of the two arrays)

        Example
        -------
        ```
        with VAEVideo("output.mp4", 620, 480) as video:
            arr = (np.random.uniform((620,480,3)) * 255).astype("uint8")
            video.write_frame(observation=arr, reconstructed=arr)
        ```
    """

    def __init__(self, output: Path, video_width: int, video_height: int) -> None:
        """[summary]

        Parameters
        ----------
        output : Path
            [description]
        video_width : int
            [description]
        video_height : int
            [description]
        """
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.video_width = video_width
        self.video_height = video_height

    def __enter__(self):
        self.stream = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                framerate="10",
                s=f"{self.video_width}x{self.video_height}",
            )
            .drawtext(
                text="Observation",
                fontcolor="white",
                x=f"w/4-text_w/2",
                y=f"h/4-2*text_h",
            )
            .drawtext(
                text="Reconstructed",
                fontcolor="white",
                x="w*3/4-text_w/2",
                y="h/4-2*text_h",
            )
            .output(str(self.output), pix_fmt="yuv420p", vcodec="libx264", r="10")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        self.background = np.zeros((self.video_height, self.video_width, 3), np.uint8)
        return self

    def __exit__(self, *exc_details) -> None:
        self.stream.stdin.close()
        self.stream.wait()

    def write_frame(self, observation: np.ndarray, reconstructed: np.ndarray) -> None:
        """Write a frame to the video file.

        Parameters
        ----------
        observation : np.ndarray
             The observation array (from gym)
            
        reconstructed : np.ndarray
            The reconstructed image from the VAE
        """

        frame = self.background.copy()
        offset = (int((self.video_height - observation.shape[0]) / 2), 0)
        frame[
            offset[0] : offset[0] + observation.shape[0],
            offset[1] : offset[1] + observation.shape[1],
        ] = observation
        offset = (
            int((self.video_height - observation.shape[0]) / 2),
            observation.shape[1],
        )
        frame[
            offset[0] : offset[0] + reconstructed.shape[0],
            offset[1] : offset[1] + reconstructed.shape[1],
        ] = reconstructed
        self.stream.stdin.write(frame.tobytes())
