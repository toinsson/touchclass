import numpy as np

import pyrealsense2 as rs


class Camera(object):
    """docstring for Camera"""
    def __init__(self):
        super(Camera, self).__init__()
        self.pipeline = None
        self.frames = None

    def start(self):
        self.__enter__()
    def stop(self):
        self.__exit__(0,0,0)

    def __enter__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_all_streams()
        self.profile = self.pipeline.start(self.config)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.pipeline.stop()

    def wait_for_frames(self):
        self.frames = self.pipeline.wait_for_frames()

    @property
    def color(self):
        return np.asanyarray(self.frames.get_color_frame().get_data())

    @property
    def depth(self):
        return np.asanyarray(self.frames.get_depth_frame().get_data())

    @property
    def infrared(self):
        return np.asanyarray(self.frames.get_infrared_frame().get_data())

    @property
    def cad(self):
        align = rs.align(rs.stream.depth)
        frames_aligned = align.process(self.frames)
        data = frames_aligned.get_color_frame().get_data()
        return np.asanyarray(data)

    @property
    def dac(self):
        # TODO: provide uncoloured version
        align = rs.align(rs.stream.depth)
        frames_aligned = align.process(self.frames)
        colorizer = rs.colorizer()
        data = colorizer.colorize(frames_aligned.get_depth_frame()).get_data()
        return np.asanyarray(data)

    @property
    def points(self):
        pc = rs.pointcloud()
        points = rs.points
        pc.map_to(self.frames.get_color_frame())
        points = pc.calculate(self.frames.get_depth_frame())
        vtx = np.asanyarray(points.get_vertices())
        # texture = np.asarray(points.get_texture_coordinates())

        return vtx.view(np.float32).reshape(vtx.shape + (-1,))

    @property
    def pointcloud(self):
        # TODO: condition on depth resolution
        return self.points.reshape((480,640,3))

    def project_point_to_pixel(self, point):
        depth_intrin = self.frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
        return np.array(rs.rs2_project_point_to_pixel(depth_intrin, list(point)))

    def deproject_pixel_to_point(self, coord, value):
        depth_intrin = self.frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        return np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, list(coord), value*depth_scale))
