import numpy as np
import matplotlib.pyplot as plt

# -- Robot parameters --  
ROBOT_JOINT_1 = [0, 0, 1]
ROBOT_JOINT_2 = [0, 0, 1]
ROBOT_JOINT_3 = [0, 0, 1]
MOTOR_AXIS = ['Z', 'Y', 'Y']
ORIGIN = np.array([0, 0, 0])

# -- Goal -- 
GOAL = np.array([1, 1, 0.5])

class HomogeneousTransform():
    """maps cords from one frame into another"""
    def __init__(self) -> None:
        self.H = np.identity(4)

    def translate(self, V):
        """tranlate frame through vector V"""
        self.H[0:3, 3] = V
        return self.H

    def rotate(self, axis, theta):
        """rotate frame about axis, through theta"""
        if axis == 'X': 
            self.xRot(theta)
        elif axis == 'Y': 
            self.yRot(theta)
        elif axis == 'Z': 
            self.zRot(theta)

        return self.H

    def xRot(self, theta):
        """Rotate coordinate frame about x-axis"""
        self.H[0:3, 0:3] = np.array([[1, 0 ,0 ],
                                    [0, np.cos(theta), -1*np.sin(theta)],
                                    [0, np.sin(theta), np.cos(theta)]])

    def yRot(self, theta):
        """Rotate coordinate frame about y-axis"""
        self.H[0:3, 0:3] = np.array([[np.cos(theta), 0 , np.sin(theta)],
                                    [0, 1, 0],
                                    [-1*np.sin(theta), 0, np.cos(theta)]])

    def zRot(self, theta):
        """Rotate coordinate frame about z-axis"""
        self.H[0:3, 0:3] = np.array([[np.cos(theta), -1*np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])


def orthographic_projection(points_3d, camera=1):
    """
    Applies an orthographic projection to a 3D point cloud.
    
    Parameters:
    point_cloud (numpy.ndarray): A 2D array of shape (num_points, 3) to be projected.
    camera: defines the projective palne of the camera 
    
    Returns:
    numpy.ndarray: A 2D vector resulting from the orthographic projection.
    """

    # Define the projection matrices
    if camera == 1:
        P = np.array([[1, 0, 0],
                      [0, 0, 1]])

    else:
        P = np.array([[0, 1, 0],
                      [0, 0, 1]])

    projected_points = [] 
    
    for i in range(len(points_3d)):
        v_projected = P @ np.array(points_3d[i]).reshape(-1, 1)
        projected_points.append(v_projected)

    result = np.concatenate((projected_points[0][:2], projected_points[1][:2])).flatten()
    
    return result


def robot_forward_kin(rotation):

    # -- translation from origin to base of joint 0 -- 
    Tb = HomogeneousTransform().translate(ORIGIN) 

    # -- translation, rotation corresponding to joint 0 -- 
    R0 = HomogeneousTransform().rotate(MOTOR_AXIS[0], rotation[0]) 
    T0 = HomogeneousTransform().translate(ROBOT_JOINT_1) 

    # -- translation, rotation corresponding to joint 1 -- 
    R1 = HomogeneousTransform().rotate(MOTOR_AXIS[1], rotation[1]) 
    T1 = HomogeneousTransform().translate(ROBOT_JOINT_2)

    # -- translation, rotation corresponding to joint 2 -- 
    R2 = HomogeneousTransform().rotate(MOTOR_AXIS[2], rotation[2]) 
    T2 = HomogeneousTransform().translate(ROBOT_JOINT_3) 

    H_0 = Tb @ R0 @ T0
    H_1 = H_0 @ R1 @ T1 
    H_2 = H_1 @ R2 @ T2

    """Returns the (x,y,z) postion of the robot w.r.t the robots base"""
    result_0 = H_0 @ np.array([0,0,0,1]).T
    result_1 = H_1 @ np.array([0,0,0,1]).T
    result_2 = H_2 @ np.array([0,0,0,1]).T

    return [result_0[:3], result_1[:3], result_2[:3]]


def visual_servoing(threshold = 0.1):
    # Create the 3D plot and the two 2D subplots
    fig, (ax_3d, ax_2d_xz, ax_2d_yz) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [2, 1, 1]})
    ax_3d = fig.add_subplot(131, projection='3d')

    # -- initialise robot -- 
    joint_angles = [np.pi/6, np.pi/6, np.pi/6]
    J = initialise_jacobian()

    # -- get initial error -- 
    robot_points = robot_forward_kin(joint_angles)
    end_effector = robot_points[2]

    ex0, ey0, gx0, gy0 = orthographic_projection([end_effector ,GOAL], camera=1)
    ex1, ey1, gx1, gy1 = orthographic_projection([end_effector ,GOAL], camera=2)

    current = np.array([ex0, ey0, ex1, ey1])
    goal = np.array([gx0, gy0, gx1, gy1])

    error = np.linalg.norm(goal - current)

    while error > threshold:
        print("error: ", error)

        # -- update joint angles -- 
        f = current - goal 
        s = -1 * np.linalg.pinv(J) @ f
        joint_angles += 0.1*s

        # -- move robot get points -- 
        robot_points = robot_forward_kin(joint_angles)
        end_effector = robot_points[2]

        # -- get camera points --
        ex0, ey0, _, _ = orthographic_projection([end_effector ,GOAL], camera=1)
        ex1, ey1, _, _ = orthographic_projection([end_effector ,GOAL], camera=2)

        current = np.array([ex0, ey0, ex1, ey1])
        error = np.linalg.norm(goal - current)

        # -- broyden update --
        y = current - goal - f
        print(((y - J@s)))
        J += 0.1 * np.outer((y - J @ s), s.T) / np.dot(s.T, s)

        # -- plot data -- 
        plot_3d(robot_points, [ex0, ey0, gx0, gy0], [ex1, ey1, gx1, gy1], ax_3d, ax_2d_xz, ax_2d_yz)

    plt.show()


def initialise_jacobian():
    # -- initialise robot -- 
    joint_angles = [np.pi/6, np.pi/6, np.pi/6]
    robot_points = robot_forward_kin(joint_angles)
    end_effector_start = robot_points[2]

    # -- get inital image points -- 
    ex0_start, ey0_start, _, _ = orthographic_projection([end_effector_start, GOAL], camera=1)
    ex1_start, ey1_start, _, _ = orthographic_projection([end_effector_start, GOAL], camera=2)

    camera_start = np.array([ex0_start, ey0_start, ex1_start, ey1_start])

    J = np.zeros((4, 3))
    delta_size = 0.15

    for i in range(len(MOTOR_AXIS)):
        # -- set new joint angles -- 
        delta = np.zeros((3,))
        delta[i] = 1
        delta *= delta_size
        joint_angles_new = joint_angles + delta

        # -- move robot --
        robot_points_new = robot_forward_kin(joint_angles_new)
        end_effector_new = robot_points_new[2]

        # -- get images --
        ex0_new, ey0_new, _, _ = orthographic_projection([end_effector_new, GOAL], camera=1)
        ex1_new, ey1_new, _, _ = orthographic_projection([end_effector_new, GOAL], camera=2)
        camera_new = np.array([ex0_new, ey0_new, ex1_new, ey1_new])

        # -- calculate partial derivatives -- 
        J[:, i] = (camera_new - camera_start) / delta_size

    return J


def plot_3d(points, camera1_points, camera2_points , ax_3d, ax_2d_xz, ax_2d_yz):
    point_A, point_B, point_C = points

    # Clear the 3D plot from the previous iteration
    ax_3d.clear()

    ax_3d.scatter(*GOAL, c='g', marker='o', label='GOAL')
    ax_3d.scatter(*ORIGIN, c='r', marker='o')
    ax_3d.scatter(*point_A, c='r', marker='o')
    ax_3d.scatter(*point_B, c='r', marker='o')
    ax_3d.scatter(*point_C, c='k', marker='o', label='End Effector')

    # Plot the lines
    ax_3d.plot([ORIGIN[0], point_A[0]], [ORIGIN[1], point_A[1]], [ORIGIN[2], point_A[2]], c='b', linestyle='-', linewidth=1)
    ax_3d.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], [point_A[2], point_B[2]], c='b', linestyle='-', linewidth=1)
    ax_3d.plot([point_B[0], point_C[0]], [point_B[1], point_C[1]], [point_B[2], point_C[2]], c='b', linestyle='-', linewidth=1)

    # Set axes labels
    ax_3d.set_xlabel('X axis')
    ax_3d.set_ylabel('Y axis')
    ax_3d.set_zlabel('Z axis')

    # Set the limits for x, y, and z axes
    ax_3d.set_xlim(-1, 2)
    ax_3d.set_ylim(-1, 2)
    ax_3d.set_zlim(0, 3)

    # Add legend
    ax_3d.legend()

    # Update the plot
    plt.draw()
    plt.pause(0.1)

    # Clear the 2D subplots from the previous iteration
    ax_2d_xz.clear()
    ax_2d_yz.clear()

    # Update the 2D subplots (example with random data)
    xz_points = np.random.rand(10, 2)
    yz_points = np.random.rand(10, 2)

    # -- plot camera 1 -- 
    ax_2d_xz.scatter(camera1_points[0], camera1_points[1], c='k', label='End Effector')
    ax_2d_xz.scatter(camera1_points[2], camera1_points[3], c='g', label='GOAL')

    # -- plot camera 2 -- 
    ax_2d_yz.scatter(camera2_points[0], camera2_points[1], c='k', label='End Effector')
    ax_2d_yz.scatter(camera2_points[2], camera2_points[3], c='g', label='GOAL')

    # Set the limits for the 2D subplots
    ax_2d_xz.set_xlim(-4, 4)
    ax_2d_xz.set_ylim(-4, 4)
    ax_2d_yz.set_xlim(-4, 4)
    ax_2d_yz.set_ylim(-4, 4)

    # Set the labels for the 2D subplots
    ax_2d_xz.set_xlabel('X axis')
    ax_2d_xz.set_ylabel('Z axis')
    ax_2d_yz.set_xlabel('Y axis')
    ax_2d_yz.set_ylabel('Z axis')

    # Update the 2D subplots
    plt.draw()
    plt.pause(0.1)


visual_servoing()
