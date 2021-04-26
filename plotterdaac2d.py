import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import gtsam
import matplotlib
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



def plotPoint2OnAxes(ax, point, linespec):
    ax.plot([point[0]], [point[1]], linespec)


def plotPoint2(fignum, point, linespec, ax=None):
    if ax is None:
        fig = plt.figure(fignum)
        ax = fig.gca()
    plotPoint2OnAxes(ax, point, linespec)


def plot2DPoints(fignum, values, linespec, marginals=None, ax=None):
    # PLOT3DPOINTS Plots the Point3's in a values, with optional covariances
    #    Finds all the Point3 objects in the given Values object and plots them.
    #  If a Marginals object is given, this function will also plot marginal
    #  covariance ellipses for each point.

    # Plot points and covariance matrices
    keys = values.keys()

    # Plot points and covariance matrices
    for key in keys:
        try:
            p = values.atPoint3(key)
            # if haveMarginals
            #     P = marginals.marginalCovariance(key);
            #     gtsam.plotPoint3(p, linespec, P);
            # else
            plotPoint2(fignum, p, linespec, ax=ax)
        except RuntimeError:
            continue
            # I guess it's not a Point3


def plotPose2OnAxes(ax, pose, axisLength=0.01): #WIP
    # get rotation and translation (center)
    # gRp = pose.rotation().matrix()  # rotation from pose to global
    # C = pose.translation().vector()

    angles = pose.rotation().matrix()
    psi = np.arctan2(angles[1,0], angles[0,0])

    gRp = np.array([[np.cos(psi), -np.sin(psi)],[ np.sin(psi), np.cos(psi)]])
    C = np.array([pose.x(), pose.y()])

    # draw the camera axes
    xAxis = C + gRp[:, 0] * axisLength
    L = np.append(C[np.newaxis], xAxis[np.newaxis], axis=0)
    ax.plot(L[:, 0], L[:, 1], 'r-')

    yAxis = C + gRp[:, 1] * axisLength
    L = np.append(C[np.newaxis], yAxis[np.newaxis], axis=0)
    ax.plot(L[:, 0], L[:, 1], 'b-')

    # # plot the covariance
    # if (nargin>2) && (~isempty(P))
    #     pPp = P(4:6,4:6); % covariance matrix in pose coordinate frame
    #     gPp = gRp*pPp*gRp'; % convert the covariance matrix to global coordinate frame
    #     gtsam.covarianceEllipse3D(C,gPp);
    # end

def plotPose2OnAxesGT(ax, pose, axisLength=0.02): #WIP
    # get rotation and translation (center)
    # gRp = pose.rotation().matrix()  # rotation from pose to global
    # C = pose.translation().vector()

    angles = pose.rotation().matrix()
    psi = np.arctan2(angles[1,0], angles[0,0])

    gRp = np.array([[np.cos(psi), -np.sin(psi)],[ np.sin(psi), np.cos(psi)]])
    C = np.array([pose.x(), pose.y()])

    # draw the camera axes
    xAxis = C + gRp[:, 0] * axisLength
    L = np.append(C[np.newaxis], xAxis[np.newaxis], axis=0)
    ax.plot(L[:, 0], L[:, 1], 'g-')

    yAxis = C + gRp[:, 1] * axisLength
    L = np.append(C[np.newaxis], yAxis[np.newaxis], axis=0)
    ax.plot(L[:, 0], L[:, 1], 'y-')

    # # plot the covariance
    # if (nargin>2) && (~isempty(P))
    #     pPp = P(4:6,4:6); % covariance matrix in pose coordinate frame
    #     gPp = gRp*pPp*gRp'; % convert the covariance matrix to global coordinate frame
    #     gtsam.covarianceEllipse3D(C,gPp);
    # end


def plotPose2(fignum, pose, axisLength=0.1, GT=False, ax=None):
    # get figure object
    if ax is None:
        fig = plt.figure(fignum)
        ax = fig.gca()
    if GT is False:
        plotPose2OnAxes(ax, pose, axisLength)
    else:
        plotPose2OnAxesGT(ax, pose, axisLength)


def plot_ellipse(ax, mu, sigma, color="k", alpha=0.2, linewidth=1, edgecolor='k', fill=False, enlarge=1):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals) * enlarge

    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(mu, w, h, theta, facecolor=color, linewidth=linewidth, edgecolor=edgecolor, fill=fill)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(alpha)
    ax.add_artist(ellipse)


def extract_mu_cov_2d(values, marginals, key):

    full_point = values.atPose3(key)
    point_2d = np.array([full_point.x(), full_point.y()])

    full_cov = marginals.marginalCovariance(key)
    cov_2d = full_cov[3:5, 3:5]

    return point_2d, cov_2d


# Plot ground truth poses
def GT_plotter(GT_poses, GT_objects, fig_num=0, show_plot=True, ax=None, plt_save_name=None, pause_time=None,
               limitx=None, limity=None, multistart=False, start_poses=None, red_point=None):


    # Figure ID
    if ax is None:
        fig = plt.figure(fig_num)
        ax = fig.gca()
    plt.gca()

    idx = 0
    # Plot cameras
    for pose in GT_poses:
        plotPose2(fig_num, pose, 0.1, GT=True, ax=ax)

        # Annotate the point
        # ax.text(pose.x(), pose.y(), 'X' + str(idx))
        idx += 1

    idx = 1
    # Plot objects
    for pose in GT_objects:
        plotPose2(fig_num, pose, 0.15, GT=True, ax=ax)
        plotPoint2(fig_num, pose.translation(), 'go', ax=ax)

        # Annotate the point
        ax.text(pose.x(), pose.y(), 'O' + str(idx))

        idx += 1

    robot_idx = 1
    if multistart is True:
        for pose in start_poses:
            plotPoint2(fig_num, pose.translation(), 'ro', ax=ax)

            # Annotate the point
            ax.text(pose.x(), pose.y(), 'R' + str(robot_idx))
            robot_idx += 1

    if red_point is not None:
        plotPoint2(fig_num, GT_poses[red_point].translation(), 'ro', ax=ax)


    if show_plot is True:
        # plt.title('Figure number: ' + str(fig_num) + ' Ground truth')
        ax.set_xlabel('X axis [m]')
        ax.set_ylabel('Y axis [m]')
        if limitx is not None:
            ax.set_xlim(limitx[0],limitx[1])
        if limity is not None:
            ax.set_ylim(limity[0],limity[1])
        plt.tight_layout()
        if plt_save_name is not None:
            plt.savefig(plt_save_name + '.eps', format='eps')
        if pause_time is not None:
            plt.show(block=False)
            plt.pause(pause_time)
            plt.close()
        else:
            plt.show()
    return ax

# Plot ground truth poses, line
def GT_plotter_line(GT_poses, GT_objects, fig_num=0, show_plot=True, ax=None, color='r', limitx=None, limity=None,
                    start_pose=None, red_point=None, robot_id=None, plt_save_name=None,
                    jpg_save=False, pause_time=None, alpha=1):

    # Figure ID
    if ax is None:
        fig = plt.figure(fig_num)
        ax = fig.gca()
    plt.gca()

    idx = 0
    # Plot cameras
    previous_pose = GT_poses[0]
    for pose in GT_poses:
        # plotPose2(fig_num, pose, 0.1, GT=True, ax=ax)
        ax.plot([previous_pose.x(), pose.x()],[previous_pose.y(), pose.y()],color=color, alpha=alpha)
        previous_pose = pose
        # Annotate the point
        # ax.text(pose.x(), pose.y(), 'X' + str(idx))
        idx += 1

    idx = 1
    # Plot objects
    for pose in GT_objects:
        plotPose2(fig_num, pose, 0.5, GT=True, ax=ax)
        plotPoint2(fig_num, pose.translation(), 'go', ax=ax)

        # Annotate the point
        ax.text(pose.x(), pose.y(), 'O' + str(idx))

        idx += 1

    if start_pose is not None:
        plotPoint2(fig_num, start_pose.translation(), 'ro', ax=ax)
        ax.text(start_pose.x(), start_pose.y(), str(robot_id))

    if red_point is not None:
        plotPoint2(fig_num, GT_poses[red_point].translation(), 'ro', ax=ax)

    if show_plot is True:
        # plt.title('Figure number: ' + str(fig_num) + ' Ground truth')
        ax.set_xlabel('X axis [m]')
        ax.set_ylabel('Y axis [m]')
        if limitx is not None:
            ax.set_xlim(limitx[0],limitx[1])
        if limity is not None:
            ax.set_ylim(limity[0],limity[1])
        plt.tight_layout()
        if plt_save_name is not None and jpg_save is False:
            plt.savefig(plt_save_name + '.eps', format='eps')
        elif plt_save_name is not None and jpg_save is True:
            plt.savefig(plt_save_name + '.jpg', format='jpg')
        if pause_time is not None:
            plt.show(block=False)
            plt.pause(pause_time)
            plt.close()
        else:
            plt.show()

    return ax

def prior_plotter(prior_pose, prior_noise, prior_objects, prior_obj_noise, fig_num=0, show_plot=True,
                  limitx=None, limity=None):
    # Figure ID
    fig = plt.figure(fig_num)
    ax = fig.gca()
    plt.gca()

    prior_noise_2d = np.array([[prior_noise[3],0],[0,prior_noise[4]]])
    prior_obj_noise_2d = np.array([[prior_obj_noise[3],0],[0,prior_obj_noise[4]]])

    # Extract psi
    angles = prior_pose.rotation().matrix()
    psi = np.arctan2(angles[1, 0], angles[0, 0])
    gRp = np.array([[np.cos(psi), np.sin(psi)], [np.sin(psi), -np.cos(psi)]])
    multi1 = np.matmul(gRp, prior_noise_2d)
    prior_noise_2d_rotated = np.matmul(multi1, np.transpose(gRp))

    # Plot cameras
    plotPose2(fig_num, prior_pose, 0.3, GT=True)
    mu = np.array([prior_pose.x(), prior_pose.y()])
    plot_ellipse(ax, mu, prior_noise_2d_rotated)

    # Annotate the point
    ax.text(prior_pose.x(), prior_pose.y(), 'X0')

    idx = 1
    # Plot objects
    for pose in prior_objects:
        # Extract psi from objects
        angles = pose.rotation().matrix()
        psi = np.arctan2(angles[1, 0], angles[0, 0])
        gRp = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        multi1 = np.matmul(gRp, prior_obj_noise_2d)
        prior_obj_noise_2d_rotated = np.matmul(multi1, np.transpose(gRp))


        plotPoint2(fig_num, pose.translation(), 'go')
        mu = np.array([pose.x(), pose.y()])
        plot_ellipse(ax, mu, prior_obj_noise_2d_rotated)

        # Annotate the point
        ax.text(pose.x(), pose.y(), 'O' + str(idx))

        idx += 1

    # plt.title('Figure number: ' + str(fig_num) + ', Priors')
    #ax.set_ylim(bottom=-5,top=7) # REMOVE WHEN IRRELEVANT
    #ax.set_xlim(right=8, left =-8) # REMOVE WHEN IRRELEVANT
    ax.set_xlabel('X axis [m]')
    ax.set_ylabel('Y axis [m]')

    if show_plot is True:
        if limitx is not None:
            ax.set_xlim(limitx[0], limitx[1])
        if limity is not None:
            ax.set_ylim(limity[0], limity[1])
        plt.show()

# Rotate covariance
def rotate_covariance(pose, cov):

    angles = pose.rotation().matrix()
    psi = np.arctan2(angles[1, 0], angles[0, 0])
    gRp = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    multi1 = np.matmul(gRp, cov)
    cov_rotated = np.matmul(multi1, np.transpose(gRp))

    return cov_rotated

# Class probability for multiple classes
def plot_weight_bars(ax, log_weight_vector, log_weight_flag=True, labels=None, bottom=None):

    if bottom is None:
        lower_limit = np.zeros(len(log_weight_vector))
    else:
        lower_limit = bottom

    if log_weight_flag is True:
        weight_vector = np.exp(log_weight_vector)
    else:
        weight_vector = log_weight_vector

    x = np.arange(len(log_weight_vector))
    bars = ax.bar(x, weight_vector, bottom=lower_limit)
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        #ax.set_xticks(x, labels)
    ax.set_ylabel('Probability of weight')
    ax.set_ylim(0, 1)

    return bars

# Class probability with error for multiple classes; lambda representation
def plot_weight_bars_multi(ax, log_weight_vector_list, log_weight_flag=True, labels=None, bottom=None,
                           plot_individual_points=False, color='blue', width_offset=0):

    # For class probabilities with more than 2 candidate classes
    if bottom is None:
        lower_limit = np.zeros(len(log_weight_vector_list[0]))
    else:
        lower_limit = bottom

    # Check whether log data or not
    weight_vector_list = list()
    if log_weight_flag is True:
        for log_weight_vector in log_weight_vector_list:
            weight_vector_list.append(np.exp(log_weight_vector))
    else:
        for log_weight_vector in log_weight_vector_list:
            weight_vector_list.append(log_weight_vector)

    ind_class_probabilities = list()

    # Create multiple class probabilities
    for hybridb_idx in range(len(weight_vector_list[0])):
        ind_class_probabilities.append(np.zeros(len(weight_vector_list)))
        for idx in range(len(weight_vector_list)):
            ind_class_probabilities[-1][idx] = weight_vector_list[idx][hybridb_idx]

    mean_vector = np.zeros(len(ind_class_probabilities))
    error_vector = np.zeros(len(ind_class_probabilities))

    for idx in range(len(ind_class_probabilities)):
        mean_vector[idx] = np.mean(ind_class_probabilities[idx])
        error_vector[idx] = np.std(ind_class_probabilities[idx])

    width = 0.18

    x = np.arange(len(weight_vector_list[0]))
    bars = ax.bar(x + width_offset, mean_vector, width, yerr=error_vector, bottom=lower_limit, alpha=0.5, ecolor='black', color=color)

    # Plot individual points
    if plot_individual_points is True:
        idx = 0
        for rect in bars:
            for hybridb_idx in range(len(weight_vector_list)):
                plt.text(rect.get_x() + rect.get_width() / 2.7, ind_class_probabilities[idx][hybridb_idx], '*',
                         color='red')
            idx += 1

    if labels is not None:
        ax.set_xticks(x + width_offset / 2)
        ax.set_xticklabels(labels)
    ax.set_ylabel('Probability of weight')
    ax.set_ylim(0, 1)

    return bars

# Plot bars from dictionary
def plot_weight_bars_multi_dict(ax, log_weight_vector_dict, log_weight_flag=True, labels=None, bottom=None,
                           plot_individual_points=False, color='blue', width_offset=0):

    # For class probabilities with more than 2 candidate classes
    if bottom is None:
        lower_limit = np.zeros(len(log_weight_vector_dict))
    else:
        lower_limit = bottom

    # Check whether log data or not
    weight_vector_dict = dict()
    if log_weight_flag is True:
        for key in log_weight_vector_dict:
            weight_vector_dict[key] = np.exp(log_weight_vector_dict[key])
    else:
        for key in log_weight_vector_dict:
            weight_vector_dict[key] = log_weight_vector_dict[key]

    mean_dict = dict()
    error_dict = dict()
    for key in weight_vector_dict:
        mean_dict[key] = np.mean(weight_vector_dict[key])
        error_dict[key] = np.std(weight_vector_dict[key])

    mean_data = mean_dict.values()
    error_data = error_dict.values()
    key_data = mean_dict.keys()

    char_key = list()
    for key in key_data:
        char_key.append('Object ' + str(int(key)))

    x_pos_center = [i for i, _ in enumerate(char_key)]


    width = 0.18

    print(list(ax.get_xticks()))
    # x_pos = [i + width_offset for i in list(ax.get_xticks())]
    x_pos = [i + width_offset for i, _ in enumerate(char_key)]
    print(x_pos)

    bars = ax.bar(x_pos, mean_data, width, yerr=error_data, bottom=lower_limit,
                  alpha=0.5, ecolor='black', color=color)

    #
    #
    # ax.bar(x_pos, mean_data, width, yerr=error_data, bottom=lower_limit,
    #        alpha=0.5, ecolor='black', color=color)

    #ax.set_xticks(list(ax.get_xticks()) + x_pos_center)
    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Bunch of code for accumilated objects
    labels_total = list()
    for label in labels:
        if label != '':
            labels_total.append(label)

    for key in char_key:
        if key not in labels_total:
            labels_total.append(key)

    print(labels_total)

    # print(x_pos)
    # ax.set_xticks(x_pos + list(ax.get_xticks()))
    # print(list(ax.get_xticks()))
    # print(list(ax.get_xticklabels()))
    # print(labels)
    ax.set_xticks(x_pos_center)
    ax.set_xticklabels(labels_total)

    # Plot individual points
    if plot_individual_points is True:
        idx = 0
        for rect in bars:
            for key in weight_vector_dict:
                plt.text(rect.get_x() + rect.get_width() / 2.7, weight_vector_dict[key][idx], '*',
                         color='red')
            idx += 1

    ax.set_ylabel('Probability of weight')
    ax.set_ylim(0, 1)

    return bars




