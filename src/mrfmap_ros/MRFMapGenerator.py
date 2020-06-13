#!/usr/bin/env python
import numpy as np
import os,errno
from mrfmap_ros.GVDBPyModules import GVDBInference, GVDBImage, gvdb_params, KeyframeSelector
import pdb

dtype = np.float32
class MRFMapGenerator:
    def __init__(self, params_file, dir_path, title):
        print 'Loading params from file {0}'.format(params_file)
        self.params = gvdb_params()
        self.params.load_from_file(params_file)
        self.params.set_from_python()
        print 'Creating Inference object...'
        self.inference = GVDBInference(True, False)
        # self.inference = None
        self.keyframe_selector = KeyframeSelector(0.2, 0.2)
        self.dir_path = dir_path
        self.title = title

    def add_data(self, p, i_ptr):
        self.inference.add_camera(p.astype(dtype), i_ptr)
        self.inference.perform_inference()

    def load_brick_data(self):
        bricks = self.inference.get_occupied_bricks(0)
        self.alphas = np.zeros((bricks.shape[0], 8*8*8), dtype=np.float32)

        brick_xs, brick_ys, brick_zs = np.mgrid[0:8, 0:8, 0:8]

        self.global_xs = np.array([brick_xs + b[0] - 4 for b in bricks])
        self.global_ys = np.array([brick_ys + b[1] - 4 for b in bricks])
        self.global_zs = np.array([brick_zs + b[2] - 4 for b in bricks])

        self.inference.get_indices_and_alphas(self.alphas)
        self.alphas = self.alphas.reshape(self.alphas.shape[0], 8, 8, 8)

    def save_map(self, save_img=False):
        filename = self.dir_path + self.title + '.npy'
        self.load_brick_data()       
        temp = np.concatenate((self.global_xs.flatten()[:, None], self.global_ys.flatten()[
            :, None], self.global_zs.flatten()[:, None], self.alphas.flatten()[:, None]), axis=1)
        # Check if path exists
        try:
            os.makedirs(self.dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        np.save(filename, temp)
        if save_img:
            try:
                self.show_map(None, False)
            except:
                print 'Well, that did not work out'

    def show_map(self, didx, thresh=None, show=True, closeup=False):
        if show:
            from mayavi import mlab
            # For the heaavvvyyy runs...
            # mlab.options.offscreen = True
        if thresh == None:
            thresh = 0.1  # self.params.prior + 0.1*self.params.prior
        subselected = self.alphas.flatten() > thresh
        fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(
            0, 0, 0), size=(800, 600))
        scaled_xs = (self.global_xs*self.params.res -
                     self.params.dims[0]/2.0).flatten()[subselected]
        scaled_ys = (self.global_ys*self.params.res -
                     self.params.dims[1]/2.0).flatten()[subselected]
        scaled_zs = (self.global_zs*self.params.res).flatten()[subselected]

        if didx == 0:
            # Rotate by 180 about Z since mocap arena has x facing backwards
            scaled_xs = -scaled_xs
            scaled_ys = -scaled_ys

        if didx == 1 or didx == 2:
            self.alphas = self.global_zs*self.params.res

        obj = mlab.points3d(scaled_xs,
                            scaled_ys,
                            scaled_zs, self.alphas.flatten()[subselected], figure=fig, mode='cube', scale_mode='none', scale_factor=self.params.res, colormap='viridis', vmin=0.0, vmax=1.0)
        obj.glyph.color_mode = 'color_by_scalar'

        # if didx==0:
        #     cb = mlab.colorbar(object=obj, orientation='horizontal', title='Occupancy')
        #     cb.scalar_bar.unconstrained_font_size = True
        #     cb.label_text_property.font_size=16
        if not closeup:
            outline = mlab.outline(color=(0, 0, 0))
            outline.outline_mode = 'cornered'
            axes = mlab.axes(color=(0, 0, 0), nb_labels=5, ranges=[-self.params.dims[0]/2.0, self.params.dims[0]/2.0,
                                                                -self.params.dims[1]/2.0, self.params.dims[1]/2.0,
                                                                0, self.params.dims[2]])
            axes.axes.label_format = '%.2f'
            axes.label_text_property.bold = False
        
        filename = self.dir_path + self.title + '.png'
        # mlab.title(self.title, height=0.9)
        cam = fig.scene.camera

        if didx == 0:
            cam.zoom(0.8)
            cam.focal_point = (0,0,0.5)
            cam.position = cam.position +(0,0,-1.5)

            if closeup:
                # For Hand zoom
                cam.position = [1.2778361880731004, 1.0320858725903652, 1.78619247887134]
                cam.focal_point = [-0.6277706322975143, 0.09905530113369865, 1.3639411406184958]
                cam.view_angle = 37.5
                cam.view_up = [-0.16874950695747495, -0.09915466969595163, 0.9806589393765275]
                cam.clipping_range = [0.007254798749460739, 7.254798749460739]            

        if didx==1:

            cam.position = [3.633748241697305, 2.5587483132228788, 4.68374820612078]
            cam.focal_point = [-0.04999995231628418, -1.1249998807907104, 1.0000000121071935]
            cam.view_angle = 30.0
            cam.view_up = [0.0, 0.0, 1.0]
            cam.clipping_range = [0.016383043070992025, 16.383043070992024]

            # For chair zoom
            if closeup:
                cam.position = [2.319401891144527, -1.1163001192205257, 2.143165504968805]
                cam.focal_point = [1.1456461918469256, -2.290055818518119, 0.9694098056712085]
                cam.view_angle = 30.0
                cam.view_up = [0.0, 0.0, 1.0]
                cam.clipping_range = [0.011199111522278649, 11.199111522278649]

        if didx == 2 :
            if closeup:
                cam.position = [10.956703963028406, 10.956703963028406, 3.866704193214095]
                cam.focal_point = [7.489999771118164, 7.489999771118164, 0.4000000013038516]
                cam.view_angle = 30.0
                cam.view_up = [0.0, 0.0, 1.0]
                cam.clipping_range = [0.03297374234716125, 32.97374234716125]
        mlab.savefig(filename, figure=fig)
        if show:
            mlab.show()
        mlab.close(all=True)
