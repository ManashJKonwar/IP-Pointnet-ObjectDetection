__author__ = "konwar.m"
__copyright__ = "Copyright 2022, AI R&D"
__credits__ = ["konwar.m"]
__license__ = "Individual Ownership"
__reference__ = "https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/prepare_data.py"
__version__ = "1.0.1"
__maintainer__ = "konwar.m"
__email__ = "rickykonwar@gmail.com"
__status__ = "Development"

def extract_frustum_data(idx_filename, split, output_filename, viz=False,
                        perturb_box2d=False, augmentX=1, type_whitelist=['Car'], dataset_folder=r'datasets\KITTI'):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)
        
    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(dataset_folder, split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist :continue

            # 2D BOX: Get pts rect backprojected 
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
                    print(box2d)
                    print(xmin,ymin,xmax,ymax)
                else:
                    xmin,ymin,xmax,ymax = box2d
                box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                    (pc_image_coord[:,0]>=xmin) & \
                    (pc_image_coord[:,1]<ymax) & \
                    (pc_image_coord[:,1]>=ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds,:]
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                uvdepth = np.zeros((1,3))
                uvdepth[0,0:2] = box2d_center
                uvdepth[0,2] = 20 # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                    box2d_center_rect[0,0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
                _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if ymax-ymin<25 or np.sum(label)==0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
    
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]
        
    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)
    
    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i] 
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def generate_data(**kwargs):
    """
    This method generates the frustum level point cloud data for objects inside each of the rgb image
    Parameters: 
        dataset_directory (str): Path for KITTI dataset
        generate_train_data (bool): Decides whether frustum data is to be extracted for training set
        generate_val_data (bool): Decides whether frustum data is to be extarcted for validation set
        generate_rgb_val_data (bool): Decides whether frustum data is to be extarcted for rgb validation set
    Returns: 
        None
    """
    dataset_directory=kwargs.get('dataset_directory')
    generate_train_data=kwargs.get('generate_train_data')
    generate_val_data=kwargs.get('generate_val_data')
    generate_rgb_val_data=kwargs.get('generate_rgb_val_data')
    object_list=kwargs.get('object_list')
    output_prefix = 'frustum_'+ ''.join([obj[:4] for obj in object_list]) + '_'

    if generate_train_data:
        extract_frustum_data(os.path.join(dataset_directory, r'image_sets/train.txt'),
                        'training',
                        os.path.join(dataset_directory, output_prefix+'train.pickle'), 
                        viz=False, 
                        perturb_box2d=True, 
                        augmentX=5,
                        type_whitelist=object_list)

    if generate_val_data:
        extract_frustum_data(os.path.join(dataset_directory, r'image_sets/val.txt'),
                        'training',
                        os.path.join(dataset_directory, output_prefix+'val.pickle'),
                        viz=False, 
                        perturb_box2d=False, 
                        augmentX=1,
                        type_whitelist=object_list)

    if generate_rgb_val_data:
        extract_frustum_data_rgb_detection(os.path.join(dataset_directory, r'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(dataset_directory, output_prefix+'val_rgb_detection.pickle'),
            viz=False,
            type_whitelist=object_list) 