import numpy as np
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames

videos = [['R0530_20240401_13-04-14_calibration-charuco-camA.mp4'],
          ['R0530_20240401_13-04-14_calibration-charuco-camB.mp4'],
          ['R0530_20240401_13-04-14_calibration-charuco-camC.mp4'],
          ['R0530_20240401_13-04-14_calibration-charuco-camD.mp4']]

cam_names = ['A', 'B', 'C','D']

n_cams = len(videos)

board = CharucoBoard(7, 10,
                     square_length=25, # here, in mm but any unit works
                     marker_length=18.75,
                     marker_bits=4, dict_size=50)

# make list of cameras
cameras = []
for name in cam_names:
    cam = Camera(name=name)
    cameras.append(cam)

cgroup = CameraGroup(cameras)

all_rows = cgroup.get_rows_videos(videos, board, verbose=verbose)

cgroup.set_camera_sizes_videos(videos)

# stuff in calibrate_rows (changed from self.cameras into my own cameras list constructed from list of names)
for rows, camera in zip(all_rows, cameras):
    size = camera.get_size()

    assert size is not None, \
        "Camera with name {} has no specified frame size".format(camera.get_name())

    if init_intrinsics:
        objp, imgp = board.get_all_calibration_points(rows)
        mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 9]
        objp, imgp = zip(*mixed)
        matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
        camera.set_camera_matrix(matrix.copy())
        camera.zero_distortions()



# further break down calibrate rows to only get the imgp thing that I need to check plots
# can probably download aniposelib locally to test
# check and run more easily on VS code


cgroup.calibrate_videos(videos, board)
cgroup.dump('calibration.toml')

# cgroup = CameraGroup.load('calibration.toml')


# --------------------------------------------------------------------

# start breaking down code

# the imgp should have the necessary points and can recreate it in pyplot??
# the result should be a calibrated plane without any outliers
def calibrate_rows(self, all_rows, board,
                   init_intrinsics=True, init_extrinsics=True, verbose=True,
                   **kwargs):
    assert len(all_rows) == len(self.cameras), \
        "Number of camera detections does not match number of cameras"

    for rows, camera in zip(all_rows, self.cameras):
        size = camera.get_size()

        assert size is not None, \
            "Camera with name {} has no specified frame size".format(camera.get_name())

        if init_intrinsics:
            objp, imgp = board.get_all_calibration_points(rows)
            mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 9]
            objp, imgp = zip(*mixed)
            matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
            camera.set_camera_matrix(matrix.copy())
            camera.zero_distortions()

    print(self.get_dicts())

    for i, (row, cam) in enumerate(zip(all_rows, self.cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, row)

    new_rows = [[r for r in rows if r['ids'].size >= 8] for rows in all_rows]
    merged = merge_rows(new_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)

    if init_extrinsics:
        rtvecs = extract_rtvecs(merged)
        if verbose:
            pprint(get_connections(rtvecs, self.get_names()))
        rvecs, tvecs = get_initial_extrinsics(rtvecs, self.get_names())
        self.set_rotations(rvecs)
        self.set_translations(tvecs)

    error = self.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)

    return error

# what calibrate row does
# Camera Calibration:
#
#     The function iterates over each camera and its corresponding detected rows.
#     For each camera:
#         It retrieves the size of the camera frame.
#         If init_intrinsics is True, it initializes the intrinsic parameters of the camera using calibration points from board.
#         It then calls set_camera_matrix and zero_distortions methods, presumably to set the camera matrix and zero out any distortions.
#
# Pose Estimation:
#
#     After calibrating each camera's intrinsic parameters, it estimates the pose (rotation and translation) of the calibration board in each camera's frame using estimate_pose_rows method.
#
# Data Processing:
#
#     It filters out rows with less than 8 detected points.
#     It merges the filtered rows.
#
# Extrinsic Calibration:
#
#     It extracts rotation and translation vectors from the merged data.
#     If init_extrinsics is True, it gets initial extrinsic parameters and sets them using set_rotations and set_translations methods.
#
# Bundle Adjustment:
#
#     It performs bundle adjustment, presumably refining the camera parameters to minimize reprojection error.
#
# Return:
#
#     It returns the error from the bundle adjustment step.

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

#board class. the
def get_all_calibration_points(self, rows):
    rows = self.fill_points_rows(rows)

    objpoints = self.get_object_points()
    objpoints = objpoints.reshape(-1, 3)

    all_obj = []
    all_img = []

    for row in rows:
        filled_test = row['filled'].reshape(-1, 2)
        good = np.all(~np.isnan(filled_test), axis=1)
        filled_app = row['filled'].reshape(-1, 2)
        objp = np.copy(objpoints)
        all_obj.append(np.float32(objp[good]))
        all_img.append(np.float32(filled_app[good]))

    # all_obj = np.vstack(all_obj)
    # all_img = np.vstack(all_img)

    # all_obj = np.array(all_obj, dtype='float64')
    # all_img = np.array(all_img, dtype='float64')

    return all_obj, all_img
