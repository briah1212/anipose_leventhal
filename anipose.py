import numpy as np
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames

vidnames = [['R0530_20240401_13-04-14_calibration-charuco-camA.mp4'],
            ['R0530_20240401_13-04-14_calibration-charuco-camB.mp4'],
            ['R0530_20240401_13-04-14_calibration-charuco-camC.mp4'],
            ['R0530_20240401_13-04-14_calibration-charuco-camD.mp4']]

cam_names = ['A', 'B', 'C','D']

n_cams = len(vidnames)

board = CharucoBoard(7, 10,
                     square_length=25, # here, in mm but any unit works
                     marker_length=18.75,
                     marker_bits=4, dict_size=50)

cgroup = CameraGroup.from_names(cam_names, fisheye=False)

cgroup.calibrate_videos(vidnames, board)
cgroup.dump('calibration.toml')

# cgroup = CameraGroup.load('calibration.toml')


# --------------------------------------------------------------------

# start breaking down code
def get_rows_videos(self, videos, board, verbose=True):
    all_rows = []

    for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
        rows_cam = []
        for vnum, vidname in enumerate(cam_videos):
            if verbose: print(vidname)
            rows = board.detect_video(vidname, prefix=vnum, progress=verbose)
            if verbose: print("{} boards detected".format(len(rows)))
            rows_cam.extend(rows)
        all_rows.append(rows_cam)

    return all_rows

def set_camera_sizes_videos(self, videos):
    for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
        rows_cam = []
        for vnum, vidname in enumerate(cam_videos):
            params = get_video_params(vidname)
            size = (params['width'], params['height'])
            cam.set_size(size)

def calibrate_videos(self, videos, board,
                     init_intrinsics=True, init_extrinsics=True, verbose=True,
                     **kwargs):
    """Takes as input a list of list of video filenames, one list of each camera.
    Also takes a board which specifies what should be detected in the videos"""

    all_rows = self.get_rows_videos(videos, board, verbose=verbose)
    if init_extrinsics:
        self.set_camera_sizes_videos(videos)

    error = self.calibrate_rows(all_rows, board,
                                init_intrinsics=init_intrinsics,
                                init_extrinsics=init_extrinsics,
                                verbose=verbose, **kwargs)
    return error, all_rows
