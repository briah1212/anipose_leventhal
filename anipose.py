import numpy as np
import cv2  # Added to support visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from matplotlib.animation import FuncAnimation
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup, triangulate_simple
from aniposelib.utils import load_pose2d_fnames, get_initial_extrinsics
from aniposelib.boards import merge_rows, extract_points, extract_rtvecs

videos = [['R0530_20240401_13-04-14_calibration-charuco-camA.mp4'],
          ['R0530_20240401_13-04-14_calibration-charuco-camC.mp4'],
          ['R0530_20240401_13-04-14_calibration-charuco-camD.mp4']]

cam_names = ['A', 'C','D']

n_cams = len(videos)

board = CharucoBoard(10, 7,
                     square_length=25, # here, in mm but any unit works
                     marker_length=18.75,
                     marker_bits=4, dict_size=50)

# make list of cameras
cameras = []
for name in cam_names:
    cam = Camera(name=name)
    cameras.append(cam)

cgroup = CameraGroup(cameras)

all_rows = cgroup.get_rows_videos(videos, board, verbose=True)

cgroup.set_camera_sizes_videos(videos)

# stuff in calibrate_rows (changed from self.cameras into my own cameras list constructed from list of names)
for rows, camera in zip(all_rows, cameras):
    size = camera.get_size()

    # Added check for size
    assert size is not None, \
        "Camera with name {} has no specified frame size".format(camera.get_name())

    objp, imgp = board.get_all_calibration_points(rows)
    mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 9]

    # Added check for mixed being empty
    if not mixed:
        print(f"No valid calibration points found for camera {camera.get_name()}")
        continue

    objp, imgp = zip(*mixed)
    matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
    camera.set_camera_matrix(matrix.copy())
    camera.zero_distortions()


    print(cgroup.get_dicts())

    for i, (row, cam) in enumerate(zip(all_rows, cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, row)

    new_rows = [[r for r in rows if r['ids'].size >= 8] for rows in all_rows]
    merged = merge_rows(new_rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)

    # if init_extrinsics:
    rtvecs = extract_rtvecs(merged)
    # # if verbose:
    # pprint(get_connections(rtvecs, cgroup.get_names()))

    rvecs, tvecs = get_initial_extrinsics(rtvecs, cgroup.get_names())
    cgroup.set_rotations(rvecs)
    cgroup.set_translations(tvecs)

    # error = cgroup.bundle_adjust_iter(imgp, extra)

print("Calibration complete")

print("visualizing the imgp points")

points3d = cgroup.triangulate(imgp)

def visualize_calibration_points(all_obj, all_img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for obj, img in zip(all_obj, all_img):
        ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], c='b', marker='o')
        ax.scatter(img[:, 0], img[:, 1], np.zeros_like(img[:, 0]), c='r', marker='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Calibration Points')
    ax.legend(['Object Points', 'Image Points'])
    plt.show()

def visualize_cameras(all_img):
    num_cameras = len(all_img)
    for cam_idx, img_points in enumerate(all_img):
        plt.figure()
        plt.scatter(img_points[:, 0], img_points[:, 1], c='b', marker='o')
        plt.title(f"Camera {cam_idx + 1}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

def visualize_triangulated_points(triangulated_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(triangulated_points[:, 0], triangulated_points[:, 1], triangulated_points[:, 2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangulated Points')
    plt.show()

def visualize_triangulated_points_in_batches(triangulated_points, batch_size=54):
    num_points = triangulated_points.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size  # Ceiling division

    def plot_batch(batch_num):
        start = batch_num * batch_size
        end = min(start + batch_size, num_points)
        points_batch = triangulated_points[start:end]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_batch[:, 0], points_batch[:, 1], points_batch[:, 2], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Triangulated Points (Batch {batch_num + 1}/{num_batches})')
        plt.show()

    batch_num = 0
    while True:
        plot_batch(batch_num)
        user_input = input("Enter 'n' for next batch, 'p' for previous batch, or 'q' to quit: ").strip().lower()
        if user_input == 'n':
            if batch_num < num_batches - 1:
                batch_num += 1
            else:
                print("Already at the last batch.")
        elif user_input == 'p':
            if batch_num > 0:
                batch_num -= 1
            else:
                print("Already at the first batch.")
        elif user_input == 'q':
            break
        else:
            print("Invalid input. Please enter 'n', 'p', or 'q'.")

visualize_triangulated_points_in_batches(points3d)
visualize_triangulated_points(points3d) # should be a plane
# visualize_cameras(imgp) # each camera should have a set of points
# visualize_calibration_points(objp, imgp)

# further break down calibrate rows to only get the imgp thing that I need to check plots
# can probably download aniposelib locally to test
# check and run more easily on VS code


# cgroup.calibrate_videos(videos, board)
# cgroup.dump('calibration.toml')
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
